# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Generator, Optional

import ray
import torch

from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.import_utils import import_external_libs
from verl.utils.ray_utils import auto_await
from verl.workers.config import CheckpointEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.rollout import BaseRollout, RolloutReplica, get_rollout_class
from verl.workers.rollout.fault_tolerance import (
    BuildGroupPartialFailure,
    FaultToleranceConfig,
    lenient_gather,
    split_refs_by_timeout,
)
from verl.workers.rollout.utils import ensure_async_iterator

logger = logging.getLogger(__name__)


@dataclass
class TensorMeta:
    name: str
    """The name of the weight tensor."""
    shape: torch.Size
    """The shape of the weight tensor."""
    dtype: torch.dtype
    """The dtype of the weight tensor."""
    chunk_offset: int
    """The chunk offset of the weight tensor."""
    chunk_size: int
    """The chunk size of the weight tensor."""
    offset: int
    """The offset of the weight tensor in the bucket."""


class CheckpointEngineRegistry:
    """Checkpoint engine registry."""

    _registry: dict[str, type["CheckpointEngine"]] = {}

    def register(backend: str):
        """Register a checkpoint engine.

        Args:
            backend: The backend of the checkpoint engine.
        """

        def wrapper(cls: type["CheckpointEngine"]):
            CheckpointEngineRegistry._registry[backend] = cls
            return cls

        return wrapper

    @classmethod
    def get(cls, backend: str) -> type["CheckpointEngine"]:
        """Get the checkpoint engine class.

        Args:
            backend: The backend of the checkpoint engine.

        Returns:
            The checkpoint engine class.
        """
        return cls._registry[backend]

    @classmethod
    def new(cls, backend: str, *args, **kwargs) -> "CheckpointEngine":
        """Create a new checkpoint engine instance.

        Args:
            backend: The backend of the checkpoint engine.
            *args: Variable length argument pass to the checkpoint engine constructor.
            **kwargs: Arbitrary keyword arguments pass to the checkpoint engine constructor.

        Returns:
            A new checkpoint engine instance.
        """
        if backend not in cls._registry:
            raise ValueError(f"Checkpoint engine {backend} not registered")
        return cls._registry[backend](*args, **kwargs)


class CheckpointEngine(ABC):
    """CheckpointEngine is an abstraction to transfer weights from trainer to rollout.

    In trainer process:
    >>> trainer = EngineRegistry.new(...) # FSDP, Megatron, VeOmini, TorchTitan, ...
    >>> engine = CheckpointEngine.new(...) # NCCLCheckpointEngine, NIXLCheckpointEngine, ...
    >>> await engine.send_weights(trainer.get_per_tensor_param())

    In rollout process:
    >>> engine = CheckpointEngine.new(...)
    >>> server_adapter = ServerAdapter()
    >>> await server_adapter.update_weights(engine.get_weights()) # update weights via cuda ipc
    """

    @abstractmethod
    def prepare(self) -> dict[str, Any]:
        """Prepare checkpoint engine before each step send_weights/receive_weights.

        1. Allocate weight bucket.
        2. [Optional] Register weight bucket for RDMA.
        3. Return metadata to build communication topology: master ip:port, register RDMA description, etc.

        Args:
            worker_group: The worker group that the checkpoint engine will be used.

        Returns:
            A dictionary that contains the metadata of the worker group.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_topology(
        cls, trainer_world_size: int, rollout_world_size: int, metadata: list[dict]
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        """Build communication topology between all workers.

        Args:
            trainer_world_size: The world size of the trainer worker group.
            rollout_world_size: The world size of the rollout replica.
            metadata: A list of metadata `prepare` from all workers.

        Returns:
            A tuple of two dictionaries that contains the communication topology for trainer and rollout worker group.
            Each dict value should be a list argument equal to the world size of the worker group to dispatch to
            `init_process_group`.

            ```
            world_size = rollout.world_size + trainer.world_size
            kwargs = {
                "rank": list(range(world_size)),
                "world_size": [world_size] * world_size,
                "master_metadata": [metadata[0]] * world_size,
            }
            ```
        """
        raise NotImplementedError

    @abstractmethod
    def init_process_group(self, **kwargs):
        """Init process group for checkpoint engine.

        Args:
            **kwargs: Keyword arguments from `build_topology`.
        """
        raise NotImplementedError

    @abstractmethod
    def finalize(self):
        """Finalize checkpoint engine after each step send_weights/receive_weights.

        1. Free weight bucket.
        1. [Optional] Deregister weight bucket for RDMA.
        2. [Optional] Destroy process group.
        """
        raise NotImplementedError

    def destroy_nccl_group(self) -> bool:
        """Best-effort local destroy of this engine's NCCL collective group."""
        group_name = getattr(self, "group_name", None)
        if group_name is None:
            return True
        try:
            import ray.util.collective as collective

            if collective.is_group_initialized(group_name):
                collective.destroy_collective_group(group_name)
            setattr(self, "_group_initialized", False)
            return True
        except Exception as exc:
            logger.warning("[FT] destroy_nccl_group(%s) failed: %r", group_name, exc)
            setattr(self, "_group_initialized", False)
            return False

    @abstractmethod
    async def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
        """Send the weights of the model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        raise NotImplementedError

    @abstractmethod
    async def receive_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Receive the weights of the model.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        raise NotImplementedError


class CheckpointEngineWithCache(CheckpointEngine):
    """Checkpoint engine with local cache: shm, disk, etc. This allow to synchronize weights without interrupting
    rollout ongoing requests (partial rollout). After requests exhausted, rollout can get weights from local cache.

    Laminar: https://arxiv.org/abs/2510.12633
    """

    @abstractmethod
    async def get_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get the weights of the model from local cache.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        raise NotImplementedError


@CheckpointEngineRegistry.register("naive")
class ColocatedCheckpointEngine(CheckpointEngine):
    """Checkpoint engine for trainer and rollout colocated on same GPU.

    In trainer process:
    >>> engine = ColocatedCheckpointEngine()
    >>> trainer = Trainer()
    >>> server_adapter = ServerAdapter()
    >>> engine.send_weights(trainer.get_per_tensor_param())
    >>> server_adapter.update_weights(engine.receive_weights())
    """

    def __init__(self, bucket_size: int, is_master: bool = False) -> None:
        self.bucket_size = bucket_size
        self.is_master = is_master

    def prepare(self):
        raise NotImplementedError

    def init_process_group(self, **kwargs):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError

    @classmethod
    def build_topology(cls, *args, **kwargs):
        raise NotImplementedError

    def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
        """Send the weights of the model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        self.weights = weights

    def receive_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Receive the weights of the model.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        yield from self.weights
        self.weights = None


class CheckpointEngineWorker(Worker):
    """CheckpointEngineWorker colocated with inference engine's WorkerProc on same GPU.

    Args:
        rollout_config: The rollout configuration.
        model_config: The model configuration.
        server_adapter: The server adapter to update weights.
    """

    def __init__(
        self,
        rollout_config: RolloutConfig,
        model_config: HFModelConfig,
        server_adapter: BaseRollout = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.rollout_config = rollout_config
        self.model_config = model_config

        self.server_adapter: BaseRollout = server_adapter
        backend = self.rollout_config.checkpoint_engine.backend
        bucket_size = self.rollout_config.checkpoint_engine.update_weights_bucket_megabytes << 20
        engine_kwargs = self.rollout_config.checkpoint_engine.engine_kwargs.get(backend, {})
        # If custom_backend_module is set, import it so plugins can register
        # in CheckpointEngineRegistry before the backend is instantiated.
        import_external_libs(self.rollout_config.checkpoint_engine.custom_backend_module or None)
        self.checkpoint_engine: CheckpointEngine = CheckpointEngineRegistry.new(
            backend, bucket_size=bucket_size, **engine_kwargs
        )
        self.extra_rollout_args = args
        self.extra_rollout_kwargs = kwargs
        if self.server_adapter is None:
            self.server_adapter = get_rollout_class(self.rollout_config.name, self.rollout_config.mode)(
                *self.extra_rollout_args,
                config=self.rollout_config,
                model_config=self.model_config,
                device_mesh=None,
                **self.extra_rollout_kwargs,
            )
        # sglang and trt-llm need device_mesh for internal communication
        initialize_global_process_group_ray(timeout_second=None, backend="cpu:gloo")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None):
        weights = self.checkpoint_engine.receive_weights()
        await self.server_adapter.update_weights(weights, global_steps=global_steps)

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def destroy_nccl_group(self) -> bool:
        return self.checkpoint_engine.destroy_nccl_group()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_replica_rank(self) -> int:
        """Get replica rank from the underlying rollout server adapter."""
        return self.server_adapter.replica_rank

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def is_leader_rank(self) -> bool:
        """Get leader rank flag from the underlying rollout server adapter."""
        return self.server_adapter.is_leader_rank


_worker_cls = ray.remote(CheckpointEngineWorker)


class CheckpointEngineManager:
    """Checkpoint engine manager to coordinate weight synchronization between trainer and rollout replicas.

    - ME: model engine, FSDP, MCore, VeOmni, export full tensor generator `get_per_tensor_param`
    - CE: checkpoint engine, NCCL, NIXL, etc

    In trainer, model engine and checkpoint engine are in same process.
    In rollout, checkpoint engine and rollout worker are in separate process, update weights via cuda ipc.

    ```
    ┌────────┬────────┬─────┬────────┐         ┌───────────────────┬───────────────────┐
    │ ┌────┐ │ ┌────┐ │     │ ┌────┐ │         │     Replica 0     │     Replica 1     │
    │ │ ME0│ │ │ ME1│ │     │ │ MEn│ │         ├────┬────┬────┬────┼────┬────┬────┬────┤
    │ └──┬─┘ │ └────┘ │ ... │ └────┘ │         │ 0  │ 1  │ 2  │ 3  │ 0  │ 1  │ 2  │ 3  │
    │    v   |        |     |        |         └──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┘
    | ┌──┴─┐ │ ┌────┐ │     │ ┌────┐ │            ^    ^    ^   cuda ipc   ^    ^    ^
    │ │ CE │ │ │ CE │ │     │ │ CE │ │         ┌──┴─┬──┴─┬──┴─┬──┴─┬──┴─┬──┴─┬──┴─┬──┴─┐
    │ └──┬─┘ │ └────┘ │     │ └────┘ │         │ CE │ CE │ CE │ CE │ CE │ CE │ CE │ CE |
    └────┼───┴────────┴─────┴────────┘         └──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┘
         v                                        |    |    |    |    |    |    |    |
         └─────────────(nccl/nixl/..)─────────────┴────┴────┴────┴────┴────┴────┴────┘
    ```

    Args:
        config: The checkpoint engine config.
        trainer: The trainer worker group.
        replicas: The list of rollout replicas.
    """

    def __init__(
        self,
        config: CheckpointEngineConfig,
        trainer: RayWorkerGroup,
        replicas: list[RolloutReplica],
        fault_tolerance: Optional[FaultToleranceConfig] = None,
        load_balancer_handle: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.backend = config.backend
        import_external_libs(self.config.custom_backend_module or None)
        self.backend_cls = CheckpointEngineRegistry.get(config.backend)
        self.trainer = trainer
        self.replicas = replicas
        self.fault_tolerance = fault_tolerance
        self.load_balancer_handle = load_balancer_handle
        self.membership_changed: bool = False
        self._dead_lock: threading.Lock = threading.Lock()
        # Bumped on each _force_reset; broadcast so workers share group_name@resetN.
        self._reset_generation: int = 0

    def _ft_enabled(self) -> bool:
        return self.fault_tolerance is not None and self.fault_tolerance.enabled

    def _build_rollout_worker_group(self) -> RayWorkerGroup:
        with self._dead_lock:
            replicas_snapshot = list(self.replicas)
        workers = []
        for replica in replicas_snapshot:
            workers.extend(replica.workers)
        return RayWorkerGroup(
            worker_handles=workers,
            ray_cls_with_init=RayClassWithInitArgs(cls=_worker_cls),
        )

    def _replicas_for_dead_rollout_workers(self, dead_workers: list) -> list[RolloutReplica]:
        dead_rollout_workers = {idx for side, idx in dead_workers if side == "rollout"}
        if not dead_rollout_workers:
            return []

        dead_replicas = []
        offset = 0
        for replica in self.replicas:
            next_offset = offset + len(replica.workers)
            if any(offset <= idx < next_offset for idx in dead_rollout_workers):
                dead_replicas.append(replica)
            offset = next_offset
        return dead_replicas

    def _server_ids_for_replicas(self, replicas: list[RolloutReplica]) -> list[str]:
        server_ids = []
        for replica in replicas:
            server_id = getattr(replica, "_server_address", None)
            if server_id is not None:
                server_ids.append(server_id)
        return server_ids

    async def _remove_replicas_and_servers(self, replicas: list[RolloutReplica]):
        server_ids = self._server_ids_for_replicas(replicas)
        self.remove_replicas(replicas)
        if server_ids and self.load_balancer_handle is not None:
            await self.load_balancer_handle.remove_servers.remote(server_ids=server_ids)

    def _force_reset_nccl_groups(self) -> None:
        """FT: broadcast force_destroy_nccl_group to all surviving CheckpointEngineWorkers.

        Called when CKE.membership_changed=True (Supervisor pruned a replica).
        Each worker locally destroys its NCCL communicator without peer coordination
        (uses ncclCommAbort path indirectly via Ray's destroy_collective_group).
        After this, the next build_process_group will init a fresh N-1 NCCL group.

        Bounded by 30s; on timeout, logs and proceeds (build_process_group will surface
        any unresolved state).
        """
        rollout = self._build_rollout_worker_group()
        self._reset_generation += 1
        gen = self._reset_generation
        refs = (
            self.trainer.execute_checkpoint_engine(
                method=["force_destroy_nccl_group"] * self.trainer.world_size,
                generation=[gen] * self.trainer.world_size,
            )
            + rollout.execute_checkpoint_engine(
                method=["force_destroy_nccl_group"] * rollout.world_size,
                generation=[gen] * rollout.world_size,
            )
        )
        if not refs:
            return
        logger.warning(
            "[FT] _force_reset_nccl_groups: broadcasting destroy to %d surviving CKE workers",
            len(refs),
        )
        ready, not_ready = ray.wait(refs, num_returns=len(refs), timeout=30.0)
        if not_ready:
            logger.error(
                "[FT] _force_reset_nccl_groups: %d/%d workers did not ack within 30s; proceeding anyway",
                len(not_ready), len(refs),
            )
            return
        try:
            results = ray.get(ready)
            failures = [i for i, r in enumerate(results) if r is False]
            if failures:
                logger.warning(
                    "[FT] _force_reset_nccl_groups: workers %s returned False; proceeding anyway",
                    failures,
                )
        except Exception as e:
            logger.error("[FT] _force_reset_nccl_groups: ray.get on ready refs raised: %r", e)

    def build_process_group(self, rollout: RayWorkerGroup):
        """Build process group for trainer and rollout replicas.

        With FT enabled, each `ray.get` is bounded by
        `fault_tolerance.weight_sync_member_timeout_s`. On timeout, any
        timed-out refs are classified (trainer vs rollout) and surfaced via
        `BuildGroupPartialFailure` so the caller (P2 Supervisor) can prune
        dead members and retry with the alive subset.
        """
        trainer = self.trainer
        ft_on = self._ft_enabled()
        timeout_s = self.fault_tolerance.weight_sync_member_timeout_s if ft_on else None

        # 1. prepare all workers
        prepare_refs = (
            trainer.execute_checkpoint_engine(["prepare"] * trainer.world_size)
            + rollout.execute_checkpoint_engine(["prepare"] * rollout.world_size)
        )
        if ft_on:
            metadata, trainer_dead, rollout_dead = split_refs_by_timeout(
                prepare_refs,
                trainer_world_size=trainer.world_size,
                timeout_s=timeout_s,
                ray_wait_fn=ray.wait,
                ray_get_fn=ray.get,
            )
            if trainer_dead or rollout_dead:
                logger.warning(
                    "[FT] build_process_group prepare timeout: trainer_dead=%s rollout_dead=%s",
                    trainer_dead,
                    rollout_dead,
                )
                raise BuildGroupPartialFailure(
                    dead_workers=[("trainer", i) for i in trainer_dead]
                    + [("rollout", i) for i in rollout_dead],
                    stage="prepare",
                )
        else:
            metadata = ray.get(prepare_refs)

        # 2. build communication topology between all workers
        trainer_kwargs, rollout_kwargs = self.backend_cls.build_topology(
            trainer.world_size, rollout.world_size, metadata
        )
        for k, v in trainer_kwargs.items():
            assert len(v) == trainer.world_size, f"trainer_kwargs[{k}] must have length of {trainer.world_size}"
        for k, v in rollout_kwargs.items():
            assert len(v) == rollout.world_size, f"rollout_kwargs[{k}] must have length of {rollout.world_size}"

        trainer_kwargs["method"] = ["init_process_group"] * trainer.world_size
        rollout_kwargs["method"] = ["init_process_group"] * rollout.world_size

        # 3. init process group between all workers
        init_refs = trainer.execute_checkpoint_engine(**trainer_kwargs) + rollout.execute_checkpoint_engine(
            **rollout_kwargs
        )
        if ft_on:
            _, trainer_dead, rollout_dead = split_refs_by_timeout(
                init_refs,
                trainer_world_size=trainer.world_size,
                timeout_s=timeout_s,
                ray_wait_fn=ray.wait,
                ray_get_fn=ray.get,
            )
            if trainer_dead or rollout_dead:
                # NCCL partially initialized; treat as terminal (cannot safely retry).
                logger.error(
                    "[FT] build_process_group init_process_group timeout (terminal): "
                    "trainer_dead=%s rollout_dead=%s",
                    trainer_dead,
                    rollout_dead,
                )
                raise BuildGroupPartialFailure(
                    dead_workers=[("trainer", i) for i in trainer_dead]
                    + [("rollout", i) for i in rollout_dead],
                    stage="init_process_group",
                )
        else:
            ray.get(init_refs)

    def add_replicas(self, replicas: list[RolloutReplica]):
        """Add rollout replicas to the manager for elastic scale up, will rebuild process group.

        Args:
            replicas: The list of rollout replicas to add.
        """
        self.replicas.extend(replicas)

    def remove_replicas(self, replicas: list[RolloutReplica]):
        """Remove rollout replicas from the manager for elastic scale down, will rebuild process group.

        Args:
            replicas: The list of rollout replicas to remove.
        """
        replicas_set = set(replicas)
        self.replicas[:] = [r for r in self.replicas if r not in replicas_set]

    async def on_replica_dead(self, replica_id: str) -> None:
        """Remove a dead rollout replica and mark NCCL membership dirty.

        Called from the ThreadedSupervisor's sub-thread. The lock pairs with
        the read/clear in ``update_weights`` so the (replicas, membership_changed)
        pair stays consistent across threads.
        """
        with self._dead_lock:
            dead_replicas = [
                replica
                for replica in self.replicas
                if replica_id in self._server_ids_for_replicas([replica])
                or str(getattr(replica, "replica_rank", "")) == replica_id
            ]
            if not dead_replicas:
                logger.warning("[FT] on_replica_dead(%s): no matching rollout replica", replica_id)
                return
            replicas_set = set(dead_replicas)
            self.replicas[:] = [r for r in self.replicas if r not in replicas_set]
            self.membership_changed = True
        logger.warning(
            "[FT] on_replica_dead(%s): pruned %d replica(s); membership_changed=True",
            replica_id, len(dead_replicas),
        )

    @auto_await
    async def sleep_replicas(self):
        """Sleep all rollout replicas: free weight and kv_cache device memory.

        FT path: lenient gather so a dead replica doesn't crash weight sync
        before abort_all_requests / build_process_group get a chance to react.
        """
        if self._ft_enabled():
            await lenient_gather(
                [r.sleep() for r in self.replicas],
                op_name="sleep_replicas",
                logger=logger,
            )
        else:
            await asyncio.gather(*[r.sleep() for r in self.replicas])

    @auto_await
    async def wake_up_replicas(self):
        """Resume all rollout replicas: recover kv_cache and weights device memory."""
        if self._ft_enabled():
            await lenient_gather(
                [r.wake_up() for r in self.replicas],
                op_name="wake_up_replicas",
                logger=logger,
            )
        else:
            await asyncio.gather(*[r.wake_up() for r in self.replicas])

    @auto_await
    async def update_weights(self, global_steps: int = None):
        """Update weights from trainer to rollout replicas.

        Args:
            global_steps: The global steps of the trainer.
        """

        # 0. update weights for sync training with colocated trainer and rollout
        if self.backend == "naive":
            ray.get(self.trainer.update_weights(global_steps=global_steps))
            return

        # FT: if membership changed (replica died), force-reset NCCL groups on all
        # surviving workers before re-initializing. Must happen BEFORE abort/sleep so
        # any leftover NCCL collective state is cleared first.
        #
        # Lock is held only for the read+clear, not across the slow reset RPC, so the
        # Supervisor sub-thread isn't blocked for the full ray.wait timeout. A 2nd
        # death during reset will re-set the flag; the next update_weights catches it.
        if self._ft_enabled():
            with self._dead_lock:
                needs_reset = self.membership_changed
                self.membership_changed = False
            if needs_reset:
                self._force_reset_nccl_groups()

        # 1. abort and save all unfinished requests for partial rollout
        if self._ft_enabled():
            await lenient_gather(
                [r.abort_all_requests() for r in self.replicas],
                op_name="abort_all_requests",
                logger=logger,
            )
        else:
            await asyncio.gather(*[r.abort_all_requests() for r in self.replicas])

        # 2. create a temporay worker group for all replicas
        rollout = self._build_rollout_worker_group()
        trainer = self.trainer

        # 3. sleep replicas to free kv_cache before weight sync (if free_cache_engine is enabled)
        await self.sleep_replicas()

        # 4. build process group
        self.build_process_group(rollout)

        # 5. update weights of all workers
        rollout = self._build_rollout_worker_group()
        ray.get(trainer.update_weights(global_steps=global_steps) + rollout.update_weights(global_steps=global_steps))

        # 6. finalize all workers
        ray.get(
            trainer.execute_checkpoint_engine(["finalize"] * trainer.world_size)
            + rollout.execute_checkpoint_engine(["finalize"] * rollout.world_size)
        )

        # 7. resume replicas to recover kv_cache (for free_cache_engine scenarios)
        await self.wake_up_replicas()

        # 8. resume all unfinished requests for partial rollout
        if self._ft_enabled():
            await lenient_gather(
                [r.resume_generation() for r in self.replicas],
                op_name="resume_generation",
                logger=logger,
            )
        else:
            await asyncio.gather(*[r.resume_generation() for r in self.replicas])


async def split_weight_chunks(
    weights: Generator[tuple[str, torch.Tensor], None, None], bucket_size: int
) -> AsyncGenerator[tuple[TensorMeta, torch.Tensor], None]:
    """Split the weight into chunks.

    Args:
        weights: The weights generator.
        bucket_size: Max bucket size in bytes.

    Yields:
        A tuple of the weight chunk metadata and the buffer.
    """
    async for name, weight in ensure_async_iterator(weights):
        buffer = weight.view(-1).view(torch.uint8)
        chunk_offset = 0
        while chunk_offset < weight.nbytes:
            chunk_size = min(bucket_size, weight.nbytes - chunk_offset)
            tensor_meta = TensorMeta(
                name=name,
                shape=weight.shape,
                dtype=weight.dtype,
                chunk_offset=chunk_offset,
                chunk_size=chunk_size,
                offset=None,
            )
            yield (tensor_meta, buffer[chunk_offset : chunk_offset + chunk_size])
            chunk_offset += chunk_size


async def merge_weight_chunks(
    chunks: Generator[tuple[TensorMeta, torch.Tensor], None, None], bucket_size: int
) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
    """Merge the weight chunks into the original weight.

    Args:
        chunks: The chunks generator.
        bucket_size: Max bucket size in bytes.

    Yields:
        A tuple of the name of the weight tensor and the tensor itself.
    """
    merge_name, merge_weight, merge_buffer, merge_offset = None, None, None, 0
    async for tensor_meta, chunk in chunks:
        assert chunk.dtype == torch.uint8, f"Chunk dtype must be uint8, but got {chunk.dtype}"
        nbytes = tensor_meta.shape.numel() * tensor_meta.dtype.itemsize

        # weight is small enough to fit in one bucket
        if nbytes <= bucket_size:
            assert merge_weight is None, f"Weight must be None, but got {merge_name}"
            name, weight = tensor_meta.name, chunk.view(tensor_meta.dtype).view(tensor_meta.shape)
            yield (name, weight)
            continue

        if merge_weight is None:
            assert tensor_meta.chunk_offset == 0, f"Chunk offset must be 0, but got {tensor_meta}"
            merge_name, merge_weight = (
                tensor_meta.name,
                torch.empty(tensor_meta.shape, dtype=tensor_meta.dtype, device=chunk.device),
            )
            merge_buffer = merge_weight.view(-1).view(torch.uint8)
            merge_offset = 0

        assert tensor_meta.name == merge_name
        assert merge_offset == tensor_meta.chunk_offset
        merge_buffer[tensor_meta.chunk_offset : tensor_meta.chunk_offset + tensor_meta.chunk_size] = chunk
        merge_offset += tensor_meta.chunk_size
        if tensor_meta.chunk_offset + tensor_meta.chunk_size == nbytes:
            yield (merge_name, merge_weight)
            merge_name, merge_weight, merge_buffer, merge_offset = None, None, None, 0
