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
import inspect
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
    FaultToleranceConfig,
    ReplicaSnapshot,
    StageFailure,
    StageRef,
    SyncMember,
    WeightSyncAttempt,
    WeightSyncStage,
    WeightSyncStageFailure,
    lenient_gather,
    wait_ray_stage,
    wait_replica_stage,
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
            self._group_initialized = False
            return True
        except Exception as exc:
            logger.warning("[FT] destroy_nccl_group(%s) failed: %r", group_name, exc)
            self._group_initialized = False
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
    async def update_weights(
        self,
        global_steps: int = None,
        attempt_id: int | None = None,
        target_version: int | None = None,
    ):
        weights = self.checkpoint_engine.receive_weights()
        if attempt_id is None:
            # Preserve the pre-FT adapter call shape so legacy sync paths keep
            # deriving the visible version from global_steps.
            await self.server_adapter.update_weights(weights, global_steps=global_steps)
        else:
            await self.server_adapter.update_weights(
                weights,
                global_steps=global_steps,
                attempt_id=attempt_id,
                target_version=target_version,
            )

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
        sync_failure_reporter: Optional[Any] = None,
        replica_promotion_reporter: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.backend = config.backend
        import_external_libs(self.config.custom_backend_module or None)
        self.backend_cls = CheckpointEngineRegistry.get(config.backend)
        self.trainer = trainer
        self.replicas = replicas
        self.fault_tolerance = fault_tolerance
        self.load_balancer_handle = load_balancer_handle
        self.sync_failure_reporter = sync_failure_reporter
        self.replica_promotion_reporter = replica_promotion_reporter
        self._validate_fault_tolerance_backend()
        self.membership_changed: bool = False
        self._dead_lock: threading.Lock = threading.Lock()
        # This lock serializes complete sync transactions even when auto_await
        # is called from different event loops/threads.
        self._sync_lock: threading.Lock = threading.Lock()
        self._sync_in_progress: bool = False
        # Epoch changes only when an active member is removed.  An attempt
        # carrying an older epoch is stale and cannot commit its weights.
        self._membership_epoch: int = 0
        self._next_attempt_id: int = 0
        self._pending_replicas: list[RolloutReplica] = []
        # Bumped on each _force_reset; broadcast so workers share group_name@resetN.
        self._reset_generation: int = 0

    def _ft_enabled(self) -> bool:
        return self.fault_tolerance is not None and self.fault_tolerance.enabled

    def _validate_fault_tolerance_backend(self) -> None:
        """Reject unvalidated HCCL recovery instead of claiming it is safe."""
        if self._ft_enabled() and self.backend_cls.__name__ == "HCCLCheckpointEngine":
            raise RuntimeError(
                "HCCL weight-sync fault recovery is disabled until an aborted collective is "
                "proven not to block creation of the next communicator"
            )

    def set_sync_failure_reporter(self, reporter: Optional[Any]) -> None:
        """Set the non-blocking Supervisor notification callback."""
        self.sync_failure_reporter = reporter

    def set_replica_promotion_reporter(self, reporter: Optional[Any]) -> None:
        """Set the Supervisor-owned serving-admission callback."""
        self.replica_promotion_reporter = reporter

    def add_pending_replicas(self, replicas: list[RolloutReplica]) -> None:
        """Register healthy replacements without admitting them to sync/LB yet."""
        with self._dead_lock:
            known_ids = {
                self._replica_id(replica)
                for replica in tuple(self.replicas) + tuple(self._pending_replicas)
            }
            for replica in replicas:
                replica_id = self._replica_id(replica)
                if replica_id not in known_ids:
                    self._pending_replicas.append(replica)
                    known_ids.add(replica_id)

    def _replica_id(self, replica: RolloutReplica) -> str:
        """Return the identity shared by CKE, Supervisor, and the load balancer."""
        server_id = getattr(replica, "_server_address", None)
        if server_id is not None:
            return str(server_id)
        return f"replica-{getattr(replica, 'replica_rank', 'unknown')}"

    def _snapshot_replica(self, replica: RolloutReplica) -> ReplicaSnapshot:
        """Freeze worker handles so a later list mutation cannot remap ranks."""
        return ReplicaSnapshot(
            replica_id=self._replica_id(replica),
            replica=replica,
            workers=tuple(replica.workers),
        )

    def _replicas_from_members(self, members) -> tuple[ReplicaSnapshot, ...]:
        if members is None:
            with self._dead_lock:
                replicas = tuple(self.replicas) + tuple(self._pending_replicas)
            return tuple(self._snapshot_replica(replica) for replica in replicas)
        return tuple(
            member if isinstance(member, ReplicaSnapshot) else self._snapshot_replica(member)
            for member in members
        )

    def _create_attempt(self, global_steps: int | None, retry_members=None) -> WeightSyncAttempt:
        """Capture the target version, epoch, and fixed rollout workers."""
        members = self._replicas_from_members(retry_members)
        with self._dead_lock:
            membership_epoch = self._membership_epoch
            self._next_attempt_id += 1
            attempt_id = self._next_attempt_id
        return WeightSyncAttempt(
            attempt_id=attempt_id,
            target_version=global_steps,
            membership_epoch=membership_epoch,
            replicas=members,
        )

    def _is_attempt_valid(self, attempt: WeightSyncAttempt) -> bool:
        """Reject work if membership changed or a fixed member disappeared."""
        with self._dead_lock:
            if self._membership_epoch != attempt.membership_epoch:
                return False
            current_by_id = {
                self._replica_id(replica): replica
                for replica in tuple(self.replicas) + tuple(self._pending_replicas)
            }
        return all(
            current_by_id.get(snapshot.replica_id) is snapshot.replica
            for snapshot in attempt.replicas
        )

    def _build_rollout_worker_group(self, replicas=None) -> RayWorkerGroup:
        if replicas is None:
            with self._dead_lock:
                replicas_snapshot = tuple(self.replicas)
            workers = [worker for replica in replicas_snapshot for worker in replica.workers]
        elif isinstance(replicas, WeightSyncAttempt):
            workers = [worker for snapshot in replicas.replicas for worker in snapshot.workers]
        else:
            members = self._replicas_from_members(replicas)
            workers = [worker for snapshot in members for worker in snapshot.workers]
        return RayWorkerGroup(
            worker_handles=workers,
            ray_cls_with_init=RayClassWithInitArgs(cls=_worker_cls),
        )

    def _replicas_for_dead_rollout_workers(self, dead_workers: list, replicas=None) -> list[RolloutReplica]:
        dead_rollout_workers = {idx for side, idx in dead_workers if side == "rollout"}
        if not dead_rollout_workers:
            return []

        dead_replicas = []
        offset = 0
        members = self._replicas_from_members(replicas)
        for snapshot in members:
            next_offset = offset + len(snapshot.workers)
            if any(offset <= idx < next_offset for idx in dead_rollout_workers):
                dead_replicas.append(snapshot.replica)
            offset = next_offset
        return dead_replicas

    async def _remove_replicas_and_servers(self, replicas: list[RolloutReplica]):
        """Prune local sync membership; Supervisor exclusively owns LB isolation."""
        self.remove_replicas(replicas)

    async def _force_reset_nccl_groups(self, attempt: WeightSyncAttempt) -> None:
        """FT: broadcast force_destroy_nccl_group to all surviving CheckpointEngineWorkers.

        Called when CKE.membership_changed=True (Supervisor pruned a replica).
        Each worker invalidates its local communicator generation without calling
        a peer-coordinated destroy.  The next build_process_group then initializes
        a fresh N-1 communicator group.

        Every surviving worker must acknowledge the same generation before a
        new attempt can build a communicator. Proceeding after a partial reset
        would allow old and new collectives to share mutable engine state.
        """
        rollout = self._build_rollout_worker_group(attempt)
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
        stage_refs = self._stage_refs(refs, attempt)
        result = await wait_ray_stage(
            stage_refs,
            stage=WeightSyncStage.RESET_GROUP,
            timeout_s=self.fault_tolerance.weight_sync_member_timeout_s,
            ray_wait_fn=ray.wait,
            ray_get_fn=ray.get,
            attempt_valid=lambda: self._is_attempt_valid(attempt),
        )
        self._raise_stage_failure(attempt, WeightSyncStage.RESET_GROUP, result)
        rejected = tuple(
            StageFailure(
                member=entry.member,
                kind="reset_rejected",
                error_type="ResetRejected",
                error_message="checkpoint engine rejected communicator reset",
                retryable=entry.member.side == "rollout",
            )
            for entry, value in zip(stage_refs, result.values, strict=True)
            if value is False
        )
        if rejected:
            raise WeightSyncStageFailure(
                WeightSyncStage.RESET_GROUP.value,
                attempt.attempt_id,
                rejected,
            )

    def _stage_refs(self, refs: list, attempt: WeightSyncAttempt) -> list[StageRef]:
        """Attach fixed replica IDs to the flattened Ray worker order."""
        members = [
            StageRef(ref, SyncMember("trainer", index))
            for index, ref in enumerate(refs[: self.trainer.world_size])
        ]
        rollout_refs = refs[self.trainer.world_size :]
        rollout_index = 0
        for snapshot in attempt.replicas:
            for worker_index in range(len(snapshot.workers)):
                if rollout_index >= len(rollout_refs):
                    break
                members.append(
                    StageRef(
                        rollout_refs[rollout_index],
                        SyncMember("rollout", rollout_index, snapshot.replica_id),
                    )
                )
                rollout_index += 1
        if rollout_index != len(rollout_refs):
            raise RuntimeError(
                "weight-sync rollout refs do not match the fixed replica worker snapshot: "
                f"mapped={rollout_index}, refs={len(rollout_refs)}"
            )
        return members

    def _raise_stage_failure(
        self, attempt: WeightSyncAttempt, stage: WeightSyncStage, result
    ) -> None:
        if result.failures or result.membership_changed:
            raise WeightSyncStageFailure(
                stage.value,
                attempt.attempt_id,
                result.failures,
                membership_changed=result.membership_changed,
            )

    async def _wait_worker_stage(
        self,
        refs: list,
        *,
        attempt: WeightSyncAttempt,
        stage: WeightSyncStage,
        timeout_s: float,
    ) -> tuple[Any, ...]:
        result = await wait_ray_stage(
            self._stage_refs(refs, attempt),
            stage=stage,
            timeout_s=timeout_s,
            ray_wait_fn=ray.wait,
            ray_get_fn=ray.get,
            attempt_valid=lambda: self._is_attempt_valid(attempt),
        )
        self._raise_stage_failure(attempt, stage, result)
        return result.values

    async def build_process_group(
        self,
        rollout: RayWorkerGroup | None = None,
        attempt: WeightSyncAttempt | None = None,
    ):
        """Build a process group, using member-aware bounded waits in FT mode."""
        trainer = self.trainer
        ft_on = self._ft_enabled()
        if rollout is None:
            rollout = self._build_rollout_worker_group(attempt)
        if not ft_on:
            prepare_refs = (
                trainer.execute_checkpoint_engine(["prepare"] * trainer.world_size)
                + rollout.execute_checkpoint_engine(["prepare"] * rollout.world_size)
            )
            metadata = ray.get(prepare_refs)
            trainer_kwargs, rollout_kwargs = self.backend_cls.build_topology(
                trainer.world_size, rollout.world_size, metadata
            )
            trainer_kwargs["method"] = ["init_process_group"] * trainer.world_size
            rollout_kwargs["method"] = ["init_process_group"] * rollout.world_size
            init_refs = trainer.execute_checkpoint_engine(**trainer_kwargs) + rollout.execute_checkpoint_engine(
                **rollout_kwargs
            )
            ray.get(init_refs)
            return

        if attempt is None:
            attempt = self._create_attempt(global_steps=None)
        timeout_s = self.fault_tolerance.weight_sync_member_timeout_s

        # Prepare and init are separate stages so a failure is attributed to
        # the exact remote member before the transaction is retried.
        prepare_refs = (
            trainer.execute_checkpoint_engine(["prepare"] * trainer.world_size)
            + rollout.execute_checkpoint_engine(["prepare"] * rollout.world_size)
        )
        metadata = await self._wait_worker_stage(
            prepare_refs,
            attempt=attempt,
            stage=WeightSyncStage.PREPARE,
            timeout_s=timeout_s,
        )

        trainer_kwargs, rollout_kwargs = self.backend_cls.build_topology(
            trainer.world_size, rollout.world_size, list(metadata)
        )
        for key, values in trainer_kwargs.items():
            assert len(values) == trainer.world_size, f"trainer_kwargs[{key}] must have length of {trainer.world_size}"
        for key, values in rollout_kwargs.items():
            assert len(values) == rollout.world_size, f"rollout_kwargs[{key}] must have length of {rollout.world_size}"

        trainer_kwargs["method"] = ["init_process_group"] * trainer.world_size
        rollout_kwargs["method"] = ["init_process_group"] * rollout.world_size
        init_refs = trainer.execute_checkpoint_engine(**trainer_kwargs) + rollout.execute_checkpoint_engine(
            **rollout_kwargs
        )
        await self._wait_worker_stage(
            init_refs,
            attempt=attempt,
            stage=WeightSyncStage.INIT_PROCESS_GROUP,
            timeout_s=timeout_s,
        )

    def add_replicas(self, replicas: list[RolloutReplica]):
        """Register rollout replicas, deferring arrivals during a sync attempt.

        Args:
            replicas: The list of rollout replicas to add.
        """
        with self._dead_lock:
            if self._ft_enabled() and self._sync_in_progress:
                known_ids = {
                    self._replica_id(replica)
                    for replica in tuple(self.replicas) + tuple(self._pending_replicas)
                }
                for replica in replicas:
                    replica_id = self._replica_id(replica)
                    if replica_id in known_ids:
                        continue
                    self._pending_replicas.append(replica)
                    known_ids.add(replica_id)
                logger.info(
                    "[FT] queued %d replacement replica(s) until the next successful weight sync",
                    len(replicas),
                )
            else:
                self.replicas.extend(replicas)

    def remove_replicas(self, replicas: list[RolloutReplica]):
        """Remove rollout replicas from the manager for elastic scale down, will rebuild process group.

        Args:
            replicas: The list of rollout replicas to remove.
        """
        remove_ids = {self._replica_id(replica) for replica in replicas}
        with self._dead_lock:
            active_before = {self._replica_id(replica) for replica in self.replicas}
            self.replicas[:] = [r for r in self.replicas if self._replica_id(r) not in remove_ids]
            self._pending_replicas[:] = [r for r in self._pending_replicas if self._replica_id(r) not in remove_ids]
            if self._ft_enabled() and active_before - {self._replica_id(replica) for replica in self.replicas}:
                self._membership_epoch += 1
                self.membership_changed = True

    async def on_replica_dead(self, replica_id: str) -> None:
        """Remove a dead rollout replica and mark NCCL membership dirty.

        Called from the ThreadedSupervisor's sub-thread. The lock pairs with
        the read/clear in ``update_weights`` so the (replicas, membership_changed)
        pair stays consistent across threads.
        """
        with self._dead_lock:
            matches = lambda replica: (
                self._replica_id(replica) == str(replica_id)
                or str(getattr(replica, "replica_rank", "")) == str(replica_id)
            )
            dead_replicas = [replica for replica in self.replicas if matches(replica)]
            pending_replicas = [replica for replica in self._pending_replicas if matches(replica)]
            if not dead_replicas and not pending_replicas:
                logger.warning("[FT] on_replica_dead(%s): no matching rollout replica", replica_id)
                return
            dead_ids = {self._replica_id(replica) for replica in dead_replicas + pending_replicas}
            self.replicas[:] = [r for r in self.replicas if self._replica_id(r) not in dead_ids]
            self._pending_replicas[:] = [r for r in self._pending_replicas if self._replica_id(r) not in dead_ids]
            if dead_replicas:
                self._membership_epoch += 1
                self.membership_changed = True
            epoch = self._membership_epoch
        logger.warning(
            "[FT] on_replica_dead(%s): pruned %d active and %d pending replica(s); epoch=%d",
            replica_id,
            len(dead_replicas),
            len(pending_replicas),
            epoch,
        )

    def _membership_stage_failure(
        self, attempt: WeightSyncAttempt, stage: WeightSyncStage
    ) -> WeightSyncStageFailure:
        """Build a stale-attempt error without attributing it to a wrong worker."""
        return WeightSyncStageFailure(
            stage.value,
            attempt.attempt_id,
            (),
            membership_changed=True,
        )

    async def _collect_replica_stage(
        self,
        attempt: WeightSyncAttempt,
        stage: WeightSyncStage,
        operation_factory,
        timeout_s: float,
    ):
        """Run one operation per fixed replica and retain all member outcomes."""
        operations = []
        for index, snapshot in enumerate(attempt.replicas):
            try:
                operation = operation_factory(snapshot.replica)
            except BaseException as error:
                async def failed_operation(error=error):
                    raise error

                operation = failed_operation()
            operations.append((SyncMember("rollout", index, snapshot.replica_id), operation))
        result = await wait_replica_stage(
            operations,
            stage=stage,
            timeout_s=timeout_s,
            attempt_valid=lambda: self._is_attempt_valid(attempt),
        )
        return result

    async def _run_replica_stage(
        self,
        attempt: WeightSyncAttempt,
        stage: WeightSyncStage,
        operation_factory,
        timeout_s: float,
    ) -> tuple[Any, ...]:
        """Run one fixed-membership replica stage and raise on partial success."""
        result = await self._collect_replica_stage(
            attempt,
            stage,
            operation_factory,
            timeout_s,
        )
        self._raise_stage_failure(attempt, stage, result)
        return result.values

    async def _execute_attempt(self, attempt: WeightSyncAttempt) -> None:
        """Execute all pre-commit stages for one fixed membership snapshot."""
        member_timeout_s = self.fault_tolerance.weight_sync_member_timeout_s
        await self._run_replica_stage(
            attempt,
            WeightSyncStage.ABORT_REQUESTS,
            lambda replica: replica.abort_all_requests(),
            member_timeout_s,
        )
        await self._run_replica_stage(
            attempt,
            WeightSyncStage.SLEEP,
            lambda replica: replica.sleep(),
            member_timeout_s,
        )

        rollout = self._build_rollout_worker_group(attempt)
        await self.build_process_group(rollout=rollout, attempt=attempt)
        await self._begin_target_version(attempt)

        trainer_refs = self.trainer.update_weights(global_steps=attempt.target_version)
        rollout_refs = rollout.update_weights(
            global_steps=attempt.target_version,
            attempt_id=attempt.attempt_id,
            target_version=attempt.target_version,
        )
        transfer_refs = trainer_refs + rollout_refs
        transfer_result = await wait_ray_stage(
            self._stage_refs(transfer_refs, attempt),
            stage=WeightSyncStage.TRANSFER,
            timeout_s=self.fault_tolerance.weight_sync_transfer_timeout_s,
            ray_wait_fn=ray.wait,
            ray_get_fn=ray.get,
            attempt_valid=lambda: self._is_attempt_valid(attempt),
        )
        self._raise_stage_failure(attempt, WeightSyncStage.TRANSFER, transfer_result)

        finalize_kwargs = {}
        if self.backend_cls.__name__ in {"NCCLCheckpointEngine", "HCCLCheckpointEngine"}:
            # A late finalize from an older generation must not destroy the
            # communicator opened for the current retry.
            generation = self._reset_generation
            finalize_kwargs = {"reset_generation": [generation] * self.trainer.world_size}
            trainer_finalize_refs = self.trainer.execute_checkpoint_engine(
                method=["finalize"] * self.trainer.world_size,
                **finalize_kwargs,
            )
            rollout_finalize_kwargs = {"reset_generation": [generation] * rollout.world_size}
            rollout_finalize_refs = rollout.execute_checkpoint_engine(
                method=["finalize"] * rollout.world_size,
                **rollout_finalize_kwargs,
            )
        else:
            trainer_finalize_refs = self.trainer.execute_checkpoint_engine(["finalize"] * self.trainer.world_size)
            rollout_finalize_refs = rollout.execute_checkpoint_engine(["finalize"] * rollout.world_size)
        finalize_refs = trainer_finalize_refs + rollout_finalize_refs
        finalize_result = await wait_ray_stage(
            self._stage_refs(finalize_refs, attempt),
            stage=WeightSyncStage.FINALIZE,
            timeout_s=member_timeout_s,
            ray_wait_fn=ray.wait,
            ray_get_fn=ray.get,
            attempt_valid=lambda: self._is_attempt_valid(attempt),
        )
        self._raise_stage_failure(attempt, WeightSyncStage.FINALIZE, finalize_result)
        await self._commit_target_version(attempt)
        await self._verify_target_version(attempt)

    def _report_sync_failure(self, replica_ids: tuple[str, ...], stage: str) -> None:
        """Notify Supervisor asynchronously after local sync-side pruning."""
        if self.sync_failure_reporter is None:
            return
        for replica_id in replica_ids:
            try:
                result = self.sync_failure_reporter(replica_id, f"weight_sync:{stage}")
            except Exception as error:
                logger.warning("[FT] sync failure reporter rejected %s: %r", replica_id, error)
                continue
            if inspect.isawaitable(result):
                async def drain_report(result=result, replica_id=replica_id):
                    try:
                        await result
                    except Exception as error:
                        logger.warning("[FT] async sync failure report for %s failed: %r", replica_id, error)

                asyncio.create_task(drain_report())

    def _handle_stage_failure(
        self, attempt: WeightSyncAttempt, failure: WeightSyncStageFailure
    ) -> tuple[ReplicaSnapshot, ...]:
        """Prune failed rollout replicas and return the next retry snapshot."""
        if failure.trainer_failed or not failure.retryable:
            raise failure

        membership_changed = failure.membership_changed or any(
            item.kind == "membership_changed" for item in failure.failures
        )
        failed_ids = set(failure.failed_replica_ids)
        if not failed_ids and not membership_changed:
            raise failure

        with self._dead_lock:
            if membership_changed:
                # Supervisor may have removed the member already.  Reuse only
                # members still registered, without deleting healthy replicas.
                current_ids = {
                    self._replica_id(replica)
                    for replica in tuple(self.replicas) + tuple(self._pending_replicas)
                }
            else:
                active_before = {self._replica_id(replica) for replica in self.replicas}
                self.replicas[:] = [
                    replica for replica in self.replicas if self._replica_id(replica) not in failed_ids
                ]
                self._pending_replicas[:] = [
                    replica for replica in self._pending_replicas if self._replica_id(replica) not in failed_ids
                ]
                active_removed = active_before - {
                    self._replica_id(replica) for replica in self.replicas
                }
                if active_removed:
                    self._membership_epoch += 1
                    self.membership_changed = True
                current_ids = {
                    self._replica_id(replica)
                    for replica in tuple(self.replicas) + tuple(self._pending_replicas)
                }

            surviving = tuple(
                snapshot
                for snapshot in attempt.replicas
                if snapshot.replica_id in current_ids
                and (membership_changed or snapshot.replica_id not in failed_ids)
            )

        if not surviving:
            raise failure
        return surviving

    async def _reset_surviving_groups(self, attempt: WeightSyncAttempt) -> None:
        """Destroy stale communicators once before replaying the next attempt."""
        await self._force_reset_nccl_groups(attempt=attempt)
        with self._dead_lock:
            if self._membership_epoch == attempt.membership_epoch:
                self.membership_changed = False

    def _server_operations(self, attempt: WeightSyncAttempt, method_name: str, *args, **kwargs):
        """Build stable server operations for every node in the attempt snapshot."""
        operations = []
        for snapshot in attempt.replicas:
            for server_index, server in enumerate(getattr(snapshot.replica, "servers", ())):
                method = getattr(server, method_name, None)
                if method is None:
                    continue
                operations.append(
                    (
                        SyncMember("rollout", server_index, snapshot.replica_id),
                        method.remote(*args, **kwargs),
                    )
                )
        return operations

    async def _begin_target_version(self, attempt: WeightSyncAttempt) -> None:
        """Fence every node-level server before any bucket for this attempt."""
        operations = self._server_operations(
            attempt,
            "begin_weight_sync",
            attempt.attempt_id,
            attempt.target_version,
        )
        result = await wait_replica_stage(
            operations,
            stage=WeightSyncStage.TRANSFER,
            timeout_s=self.fault_tolerance.weight_sync_member_timeout_s,
            attempt_valid=lambda: self._is_attempt_valid(attempt),
        )
        self._raise_stage_failure(attempt, WeightSyncStage.TRANSFER, result)
        rejected = tuple(
            StageFailure(
                member=member,
                kind="stale_attempt",
                error_type="StaleWeightSyncAttempt",
                error_message="server rejected the weight-sync attempt",
                retryable=True,
            )
            for (member, _), accepted in zip(operations, result.values, strict=True)
            if accepted is False
        )
        if rejected:
            raise WeightSyncStageFailure(
                WeightSyncStage.TRANSFER.value,
                attempt.attempt_id,
                rejected,
            )

    async def _commit_target_version(self, attempt: WeightSyncAttempt) -> None:
        """Expose the target version only after all checkpoint engines finalize."""
        if attempt.target_version is None:
            return
        operations = self._server_operations(
            attempt,
            "set_global_steps",
            attempt.target_version,
            attempt_id=attempt.attempt_id,
        )
        result = await wait_replica_stage(
            operations,
            stage=WeightSyncStage.VERIFY_VERSION,
            timeout_s=self.fault_tolerance.weight_sync_member_timeout_s,
            attempt_valid=lambda: self._is_attempt_valid(attempt),
        )
        self._raise_stage_failure(attempt, WeightSyncStage.VERIFY_VERSION, result)

    async def _verify_target_version(self, attempt: WeightSyncAttempt) -> None:
        """Confirm every available vLLM server applied the requested version."""
        if attempt.target_version is None:
            if not self._is_attempt_valid(attempt):
                raise self._membership_stage_failure(attempt, WeightSyncStage.VERIFY_VERSION)
            return

        operations = self._server_operations(attempt, "get_global_steps")
        result = await wait_replica_stage(
            operations,
            stage=WeightSyncStage.VERIFY_VERSION,
            timeout_s=self.fault_tolerance.weight_sync_member_timeout_s,
            attempt_valid=lambda: self._is_attempt_valid(attempt),
        )
        self._raise_stage_failure(attempt, WeightSyncStage.VERIFY_VERSION, result)
        if len(result.values) != len(operations):
            raise self._membership_stage_failure(attempt, WeightSyncStage.VERIFY_VERSION)

        failures = []
        value_index = 0
        for snapshot in attempt.replicas:
            for server_index, server in enumerate(getattr(snapshot.replica, "servers", ())):
                if getattr(server, "get_global_steps", None) is None:
                    continue
                actual = result.values[value_index]
                value_index += 1
                if actual != attempt.target_version:
                    failures.append(
                        StageFailure(
                            member=SyncMember("rollout", server_index, snapshot.replica_id),
                            kind="version_mismatch",
                            error_type="VersionMismatch",
                            error_message=(
                                f"expected version {attempt.target_version}, got {actual}"
                            ),
                            retryable=True,
                        )
                    )
        if failures:
            raise WeightSyncStageFailure(
                WeightSyncStage.VERIFY_VERSION.value,
                attempt.attempt_id,
                tuple(failures),
            )

    def _commit_attempt(self, attempt: WeightSyncAttempt) -> None:
        """Commit only an epoch-valid attempt; serving admission remains separate."""
        if not self._is_attempt_valid(attempt):
            raise self._membership_stage_failure(attempt, WeightSyncStage.VERIFY_VERSION)

        attempt_ids = {snapshot.replica_id for snapshot in attempt.replicas}
        with self._dead_lock:
            current_ids = {
                self._replica_id(replica)
                for replica in tuple(self.replicas) + tuple(self._pending_replicas)
            }
            if self._membership_epoch != attempt.membership_epoch or not attempt_ids.issubset(current_ids):
                raise self._membership_stage_failure(attempt, WeightSyncStage.VERIFY_VERSION)

            self.membership_changed = False

    async def _publish_promoted_replicas(self, attempt: WeightSyncAttempt) -> None:
        """Move synced pending replicas to active only after LB acknowledgement."""
        attempt_ids = {snapshot.replica_id for snapshot in attempt.replicas}
        with self._dead_lock:
            promoted = [
                replica
                for replica in self._pending_replicas
                if self._replica_id(replica) in attempt_ids
            ]
        if not promoted:
            return
        acknowledged = list(promoted)
        if self.replica_promotion_reporter is not None:
            acknowledged = []
            for replica in promoted:
                replica_id = self._replica_id(replica)
                address = getattr(replica, "_server_address", None)
                handle = getattr(replica, "_server_handle", None)
                servers = {address: handle} if address is not None and handle is not None else {}
                try:
                    result = self.replica_promotion_reporter(
                        replica_id,
                        servers,
                        attempt.attempt_id,
                        attempt.target_version,
                    )
                    if inspect.isawaitable(result):
                        result = await result
                except Exception as error:
                    logger.warning(
                        "[FT] keeping synced replacement %s pending after Supervisor publish failed: %r",
                        replica_id,
                        error,
                    )
                    continue
                if result:
                    acknowledged.append(replica)
        else:
            servers = {
                replica._server_address: replica._server_handle
                for replica in promoted
                if getattr(replica, "_server_address", None) is not None
                and getattr(replica, "_server_handle", None) is not None
            }
        if (
            self.replica_promotion_reporter is None
            and servers
            and self.load_balancer_handle is not None
        ):
            try:
                await self.load_balancer_handle.add_servers.remote(servers)
            except Exception as error:
                logger.warning(
                    "[FT] keeping %d synced replacement replica(s) pending after LB publish failed: %r",
                    len(promoted),
                    error,
                )
                return
        promoted_ids = {self._replica_id(replica) for replica in acknowledged}
        with self._dead_lock:
            current = [
                replica
                for replica in self._pending_replicas
                if self._replica_id(replica) in promoted_ids
            ]
            current_ids = {self._replica_id(replica) for replica in current}
            self._pending_replicas[:] = [
                replica
                for replica in self._pending_replicas
                if self._replica_id(replica) not in current_ids
            ]
            active_ids = {self._replica_id(replica) for replica in self.replicas}
            self.replicas.extend(
                replica for replica in current if self._replica_id(replica) not in active_ids
            )

    async def _wake_and_resume(self, attempt: WeightSyncAttempt) -> None:
        """Resume healthy committed members even when another member fails to wake."""
        member_timeout_s = self.fault_tolerance.weight_sync_member_timeout_s
        wake_result = await self._collect_replica_stage(
            attempt,
            WeightSyncStage.WAKE_UP,
            lambda replica: replica.wake_up(),
            member_timeout_s,
        )
        failed_ids = set(wake_result.failed_replica_ids)
        survivors = tuple(
            snapshot for snapshot in attempt.replicas if snapshot.replica_id not in failed_ids
        )
        resume_result = None
        if survivors:
            resume_attempt = WeightSyncAttempt(
                attempt_id=attempt.attempt_id,
                target_version=attempt.target_version,
                membership_epoch=attempt.membership_epoch,
                replicas=survivors,
            )
            resume_result = await self._collect_replica_stage(
                resume_attempt,
                WeightSyncStage.RESUME,
                lambda replica: replica.resume_generation(),
                member_timeout_s,
            )
        failures = wake_result.failures + (() if resume_result is None else resume_result.failures)
        if failures:
            raise WeightSyncStageFailure(
                WeightSyncStage.POST_COMMIT.value,
                attempt.attempt_id,
                failures,
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

        if not self._ft_enabled():
            # Keep the non-FT path unchanged: strict, unbounded gathers retain
            # the original failure semantics when fault tolerance is disabled.
            await asyncio.gather(*[r.abort_all_requests() for r in self.replicas])
            rollout = self._build_rollout_worker_group()
            await self.sleep_replicas()
            await self.build_process_group(rollout)
            rollout = self._build_rollout_worker_group()
            ray.get(
                self.trainer.update_weights(global_steps=global_steps)
                + rollout.update_weights(global_steps=global_steps)
            )
            ray.get(
                self.trainer.execute_checkpoint_engine(["finalize"] * self.trainer.world_size)
                + rollout.execute_checkpoint_engine(["finalize"] * rollout.world_size)
            )
            await self.wake_up_replicas()
            await asyncio.gather(*[r.resume_generation() for r in self.replicas])
            return

        if not self._sync_lock.acquire(blocking=False):
            raise RuntimeError("another weight synchronization transaction is already running")

        with self._dead_lock:
            # Capture the initial set while publishing the in-progress flag.
            # A replacement callback that arrives after this point is pending
            # for the next transaction, even if it becomes healthy immediately.
            self._sync_in_progress = True
            initial_members = tuple(self.replicas) + tuple(self._pending_replicas)
            # A pending replacement changes communicator world size even when
            # a prior shrink transaction already cleared membership_changed.
            needs_reset = self.membership_changed or bool(self._pending_replicas)
        try:
            if not initial_members:
                raise RuntimeError("cannot synchronize weights without a rollout replica")

            retry_members = tuple(self._snapshot_replica(replica) for replica in initial_members)
            max_retries = max(0, int(getattr(self.fault_tolerance, "max_weight_sync_retries", 2)))
            for retry_index in range(max_retries + 1):
                attempt = self._create_attempt(global_steps, retry_members)
                try:
                    if needs_reset:
                        await self._reset_surviving_groups(attempt)
                        needs_reset = False
                    await self._execute_attempt(attempt)
                    self._commit_attempt(attempt)
                except WeightSyncStageFailure as failure:
                    if failure.trainer_failed or not failure.retryable or retry_index >= max_retries:
                        raise

                    retry_members = self._handle_stage_failure(attempt, failure)
                    if not failure.membership_changed and not any(
                        item.kind == "membership_changed" for item in failure.failures
                    ):
                        failed_ids = set(failure.failed_replica_ids)
                        self._report_sync_failure(failure.failed_replica_ids, failure.stage)
                        failed_replicas = [
                            snapshot.replica
                            for snapshot in attempt.replicas
                            if snapshot.replica_id in failed_ids
                        ]
                        if failed_replicas:
                            await self._remove_replicas_and_servers(failed_replicas)
                    if not retry_members:
                        raise

                    # The next attempt owns reset and must receive every ACK
                    # before it can build a new communicator generation.
                    needs_reset = True
                else:
                    # Commit is the transaction boundary.  A wake/resume
                    # failure after commit must not replay the full model or
                    # create a second committed attempt; report the rollout
                    # failure and let the caller handle the post-commit error.
                    try:
                        await self._wake_and_resume(attempt)
                    except WeightSyncStageFailure as failure:
                        if failure.retryable:
                            self._report_sync_failure(failure.failed_replica_ids, failure.stage)
                            failed_ids = set(failure.failed_replica_ids)
                            failed_replicas = [
                                snapshot.replica
                                for snapshot in attempt.replicas
                                if snapshot.replica_id in failed_ids
                            ]
                            if failed_replicas:
                                await self._remove_replicas_and_servers(failed_replicas)
                            remaining_ids = {
                                self._replica_id(replica)
                                for replica in tuple(self.replicas) + tuple(self._pending_replicas)
                            }
                            if not remaining_ids:
                                raise
                        else:
                            raise
                    await self._publish_promoted_replicas(attempt)
                    return
        finally:
            with self._dead_lock:
                self._sync_in_progress = False
            self._sync_lock.release()


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
