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
"""
Utility classes for manage and request LLM servers:
- LLMServerManager: manage life-cycle of LLM servers, including launch, tear-down replicas.
- LLMServerClient: proxy client to request LLM servers, used by AgentLoopWorker.
- GlobalRequestLoadBalancer: global load balancer for LLMServerClient.
"""

import asyncio
import logging
import os
import socket
from typing import Any, Optional
from uuid import uuid4

import ray
import torch
from cachetools import LRUCache
from omegaconf import DictConfig, OmegaConf

from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup
from verl.utils.device import get_resource_name
from verl.utils.ray_utils import auto_await
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import RolloutReplica, TokenOutput, get_rollout_replica_class
from verl.workers.rollout.utils import update_prometheus_config

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_ROUTING_CACHE_SIZE = 10000


class _LoadBalancerCore:
    """Plain-Python state machine for load balancing.

    Wrapped by `GlobalRequestLoadBalancer` (Ray actor) for remote access.
    Splitting the logic from the Ray decorator makes it unit-testable without
    `ray.init()`, while keeping the Ray actor as a thin forwarder.

    Per spec §横向系统 LB:
      - state: `_server` / `_inflight` / `_dead` / `_sticky`
      - all write methods idempotent
      - `acquire_server` skips dead, clears stale sticky entries
      - `enable_fault_tolerance` toggles lenient semantics for `release_server`
        (idempotent decrement vs. original strict ValueError) — when False,
        behavior is bit-exact identical to the pre-FT implementation
    """

    def __init__(
        self,
        servers: dict,
        max_cache_size: int = DEFAULT_ROUTING_CACHE_SIZE,
        enable_fault_tolerance: bool = False,
    ) -> None:
        if not servers:
            raise ValueError("server must be non-empty")

        self._server: dict = dict(servers)
        self._inflight: dict[str, int] = {sid: 0 for sid in self._server}
        self._dead: set[str] = set()
        self._sticky: LRUCache = LRUCache(maxsize=max_cache_size)
        self._ft = enable_fault_tolerance

    # ----- helpers -----

    def _alive_ids(self) -> list[str]:
        """Return server_ids that are not in the dead set."""
        return [sid for sid in self._inflight if sid not in self._dead]

    # ----- core API (sync, matching the public Ray actor surface) -----

    def acquire_server(self, request_id: str) -> str:
        # Request-level sticky (multi-turn). If cached server is dead, evict and re-pick.
        if request_id in self._sticky:
            sid = self._sticky[request_id]
            if sid in self._dead:
                del self._sticky[request_id]
            else:
                self._inflight[sid] += 1
                return sid

        alive = self._alive_ids()
        if not alive:
            from verl.workers.rollout.fault_tolerance import AllServersFailed

            raise AllServersFailed("no alive servers in pool")

        sid = min(alive, key=lambda s: self._inflight[s])
        self._sticky[request_id] = sid
        self._inflight[sid] += 1
        return sid

    def release_server(self, server_id: str) -> None:
        if self._ft:
            # Lenient: never raise; cap at 0 to keep INV-3 (no negative inflight).
            if server_id in self._inflight:
                self._inflight[server_id] = max(0, self._inflight[server_id] - 1)
            return
        # Bit-exact original behavior when FT is off.
        if server_id not in self._inflight:
            raise ValueError(f"Invalid server_id for release: {server_id}")
        if self._inflight[server_id] <= 0:
            raise ValueError(f"Release called with no inflight requests on server {server_id}")
        self._inflight[server_id] -= 1

    def mark_failed(self, server_id: str) -> None:
        """Mark a server as dead; subsequent acquires skip it. Idempotent."""
        if server_id in self._server or server_id in self._inflight:
            self._dead.add(server_id)
        # If server_id is unknown entirely, silently no-op (idempotent on unknown ids).

    def add_servers(self, servers: dict) -> None:
        """Register new servers. Idempotent on existing ids. Resurrect dead ids."""
        for sid, handle in servers.items():
            self._server[sid] = handle
            if sid in self._dead:
                self._dead.discard(sid)
                self._inflight[sid] = 0
            elif sid not in self._inflight:
                self._inflight[sid] = 0
            # If sid already exists and alive, leave inflight count untouched (idempotent).

    def remove_servers(self, server_ids: list) -> None:
        """Drop servers from routing. Adds to dead; clears their sticky entries."""
        ids = set(server_ids)
        for sid in ids:
            if sid in self._server:
                self._dead.add(sid)
                # Keep _inflight entry so callers' release_server still finds it
                # (idempotency / no spurious ValueError mid-flight on FT=False).
        # Clear sticky entries pointing to removed servers.
        stale = [rid for rid, sid in self._sticky.items() if sid in ids]
        for rid in stale:
            del self._sticky[rid]

    def get_server_handle(self, server_id: str):
        """Return the Ray actor handle for `server_id`, or None if unknown."""
        return self._server.get(server_id)


@ray.remote
class GlobalRequestLoadBalancer:
    """Global sticky-session + in-flight load balancer shared by all AgentLoopWorkers.

    Thin Ray-actor wrapper around `_LoadBalancerCore`. See the core class for
    the actual state machine + invariants.
    """

    def __init__(
        self,
        servers: dict[str, ray.actor.ActorHandle],
        max_cache_size: int = DEFAULT_ROUTING_CACHE_SIZE,
        enable_fault_tolerance: bool = False,
    ) -> None:
        self._core = _LoadBalancerCore(
            servers=servers,
            max_cache_size=max_cache_size,
            enable_fault_tolerance=enable_fault_tolerance,
        )

    def acquire_server(self, request_id: str) -> str:
        return self._core.acquire_server(request_id)

    def release_server(self, server_id: str) -> None:
        self._core.release_server(server_id)

    def mark_failed(self, server_id: str) -> None:
        self._core.mark_failed(server_id)

    def add_servers(self, servers: dict[str, ray.actor.ActorHandle]) -> None:
        self._core.add_servers(servers)

    def remove_servers(self, server_ids: list[str]) -> None:
        self._core.remove_servers(server_ids)

    def get_server_handle(self, server_id: str):
        """Forward to core for FT spawn-back path; see _LoadBalancerCore.get_server_handle."""
        return self._core.get_server_handle(server_id)


class LLMServerClient:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least in-flight requests load balancing via global coordination
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    - (FT) Transient fault translation: '_generate_once' wraps a single server call in
      wait_for + is transient_fault; on fault it marks the server failed (awaited, so the
      dead flag is visible to the next acquire) and raise ServerUnavailable for the caller
      ('FullyLLMServerClient') or the shared '_generate_with_retry' loop to handle.
      When `ft.enabled=False`, the path is bit-exact identical to the pre-FT implementation (spec §硬规则.可关性).
    """

    def __init__(
        self,
        config: DictConfig,
        servers: dict[str, ray.actor.ActorHandle],
        load_balancer_handle: ray.actor.ActorHandle,
        run_id: Optional[str] = None,
        progress_store: Optional[ray.actor.ActorHandle] = None,
        max_model_len: Optional[int] = None,
    ):
        """Initialize the LLMServerClient.

        Args:
            config (DictConfig): whole config for main entrypoint.
            servers (dict[str, ray.actor.ActorHandle]): handle for each LLM server.
            load_balancer_handle (ray.actor.ActorHandle): shared global load balancer actor.
            run_id (Optional[str]): process-level run id (for token continuation).
                None when FT/progress is off.
            progress_store (Optional[ray.actor.ActorHandle]): RolloutProgressStoreActor
                handle. None when FT/progress is off.
            max_model_len (Optional[int]): model context length, used by client
                to resolve the per-turn token-continuation budget
        """
        self.config = config
        self._load_balancer = load_balancer_handle
        self._server_id_to_handle: dict[str, ray.actor.ActorHandle] = servers
        self._run_id = run_id
        self._progress_store = progress_store
        self.max_model_len = max_model_len

    def _ft_enabled(self) -> bool:
        try:
            return bool(self.config.async_training.fault_tolerance.enabled)
        except (AttributeError, KeyError):
            return False

    def _ft_call_timeout_s(self) -> float:
        try:
            return float(self.config.async_training.fault_tolerance.server_call_timeout_s)
        except (AttributeError, KeyError):
            return 120.0  # FaultToleranceConfig default

    def _ft_max_request_retries(self) -> int:
        try:
            return int(self.config.async_training.fault_tolerance.max_request_retries)
        except (AttributeError, KeyError):
            return 3  # retry 3 times

    async def _call_server(
        self,
        server_id: str,
        server: ray.actor.ActorHandle,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]],
        video_data: Optional[list[Any]],
        kwargs: dict[str, Any],
        call_timeout_s: Optional[float],
    ) -> TokenOutput:
        """
        Issue a single "server.generate.remote" call, optionally bounded by a timeout.
        """
        call = server.generate.remote(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=video_data,
            **kwargs,
        )
        if call_timeout_s is not None:
            return await asyncio.wait_for(call, timeout=call_timeout_s)
        return await call

    async def _generate_once(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]],
        video_data: Optional[list[Any]],
        kwargs: dict[str, Any],
        call_timeout_s: Optional[float],
        translate_fault: bool,
        progress_ctx: Optional[Any] = None,
    ) -> tuple[TokenOutput, str]:
        """Single acquire -> call -> release on one server.
        Shared single attempt primitive used by "generate", "RetryLLMServerClient"
        and the FullyLLMServerClient while-loop.
        Return "(TokenOutput, server_id)".
        when "translate_fault" is True, transient faults are translated to ServerUnavailable, otherwise, exceptions
        propagate as-is bit-exact with the pre-FT path.
        """
        from verl.workers.rollout.fault_tolerance import (
            AllServersFailed,
            ServerUnavailable,
            is_transient_fault,
        )

        try:
            server_id, server = await self._acquire_server(request_id)
        except AllServersFailed:
            # Empty/all-dead pool — caller decides whether to surface or escalate.
            raise
        except Exception as e:
            if translate_fault and is_transient_fault(e):
                raise ServerUnavailable("<lb>", cause=e) from e
            raise

        try:
            try:
                if progress_ctx is not None:
                    kwargs = {**kwargs, "progress_ctx": progress_ctx}
                output = await self._call_server(
                    server_id,
                    server,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=image_data,
                    video_data=video_data,
                    kwargs=kwargs,
                    call_timeout_s=call_timeout_s,
                )
                return output, server_id
            except Exception as e:
                if translate_fault and is_transient_fault(e):
                    await self._mark_server_failed(server_id)
                    raise ServerUnavailable(server_id, cause=e) from e
                raise
        finally:
            self._release_server(server_id)

    async def _acquire_server(self, request_id: str) -> tuple[str, ray.actor.ActorHandle]:
        server_id = await self._load_balancer.acquire_server.remote(request_id=request_id)
        handle = self._server_id_to_handle.get(server_id)
        if handle is None:
            # Spawn-back replica not yet in local cache; lazy-fetch from LB.
            handle = await self._load_balancer.get_server_handle.remote(server_id=server_id)
            if handle is None:
                raise RuntimeError(f"Unknown server_id returned by load balancer: {server_id}")
            self._server_id_to_handle[server_id] = handle
        return server_id, handle

    def _release_server(self, server_id: str) -> None:
        # Fire-and-forget: release is just a counter decrement, no need to await.
        # Awaiting here risks blocking the finally clause if the LB actor is unresponsive.
        try:
            self._load_balancer.release_server.remote(server_id=server_id)
        except Exception:
            # R10: LB itself may be unhealthy; don't let release leak into caller path.
            pass

    async def _mark_server_failed(self, server_id: str) -> None:
        """Fire-and-forget notify LB that a server is dead. Must never block."""
        try:
            await asyncio.wait_for(self._load_balancer.mark_failed.remote(server_id=server_id), timeout=3.0)
        except Exception:
            logger.warning("[FT] _mark_server_failed: mark server %s failed", server_id)

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        progress_ctx: Optional[Any] = None,
        **kwargs: Any,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.
            progress_ctx (Optional[Any]): Token-continuation context (ProgressContext)
                forwarded to the server's ``generate`` for in-stream ``ingest``.
                None when FT/progress is off (mode A/B).

        Returns:
            TokenOutput | DiffusionOutput: token or diffusion output

        Raises:
            ServerUnavailable: (FT-only) the chosen server died / hung. Caller (L3) should retry.
            AllServersFailed: (FT-only) the LB has no live servers to hand out.
            Other exceptions (e.g. ValueError): non-transient — propagated as-is.
        """
        ft_on = self._ft_enabled()
        output, _ = await self._generate_once(
            request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=video_data,
            progress_ctx=progress_ctx,
            kwargs=kwargs,
            call_timeout_s=self._ft_call_timeout_s() if ft_on else None,
            translate_fault=ft_on,
        )
        output.extra_fields["llm_generate_attempts"] = 1
        return output


class RetryLLMServerClient(LLMServerClient):
    """Cross-server retry for a single prompt, independent of fault tolerance.
    When a server dies mid-generation, the partial tokens produced so far are lost with the server.
    This client switches to a fresh server and regenerates the response from the original prompt, up to max_retries
    times before raise AllServersFailed.
    """

    def __init__(
        self,
        config: DictConfig,
        servers: dict[str, ray.actor.ActorHandle],
        load_balancer_handle: ray.actor.ActorHandle,
        run_id: Optional[str] = None,
        progress_store: Optional[ray.actor.ActorHandle] = None,
        max_model_len: Optional[int] = None,
    ):
        super().__init__(
            config,
            servers,
            load_balancer_handle,
            run_id=run_id,
            progress_store=progress_store,
            max_model_len=max_model_len,
        )
        self.max_retries = self._ft_max_request_retries()
        self.call_timeout_s = self._ft_call_timeout_s()
        self._last_global_step: Optional[int] = None

    def _progress_enabled(self) -> bool:
        if not self._ft_enabled():
            return False
        if self._run_id is None or self._progress_store is None:
            return False
        try:
            return bool(OmegaConf.select(self.config, "async_training.fault_tolerance.progress.enabled", default=False))
        except (AttributeError, KeyError, TypeError):
            return False

    def _flush_token_interval(self) -> int:
        try:
            return int(
                OmegaConf.select(
                    self.config, "async_training.fault_tolerance.progress.flush_token_interval", default=64
                )
            )
        except (AttributeError, KeyError, TypeError):
            return 64

    def _model_version_policy(self):
        from verl.workers.rollout.fault_tolerance import ModelVersionPolicy

        try:
            node = OmegaConf.select(self.config, "async_training.fault_tolerance.progress.model_version_policy")
            if node is not None:
                mode = node.get("mode", "exact") if hasattr(node, "get") else "exact"
                return ModelVersionPolicy(mode=mode)
        except (AttributeError, KeyError, TypeError):
            pass
        return ModelVersionPolicy(mode="exact")

    def _resolve_original_max_tokens(self, sampling_params: dict, original_prompt: list[int]) -> int:
        if "max_tokens" in sampling_params:
            raw = sampling_params["max_tokens"]
        elif "max_new_tokens" in sampling_params:
            raw = sampling_params["max_new_tokens"]
        else:
            rollout_cfg = self.config.actor_rollout_ref.rollout
            raw = min(
                rollout_cfg.response_length,
                rollout_cfg.prompt_length + rollout_cfg.response_length - len(original_prompt),
            )
        raw = int(raw)
        if self.max_model_len is not None:
            raw = min(raw, self.max_model_len - len(original_prompt))
        return max(0, raw)

    async def _weights_version(self) -> Optional[str]:
        if self._last_global_step is not None:
            return str(self._last_global_step)
        for handle in list(self._server_id_to_handle.values()):
            try:
                gs = await asyncio.wait_for(handle.get_global_steps.remote(), timeout=5.0)
                if gs is not None:
                    self._last_global_step = int(gs)
                    return str(gs)
            except Exception:
                continue
        return None

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        **kwargs: Any,
    ) -> TokenOutput:
        import copy

        from verl.workers.rollout.fault_tolerance import (
            AllServersFailed,
            ProgressContext,
            ServerUnavailable,
            VLLMProgressCheckPoint,
        )

        progress_on = self._progress_enabled()
        max_retries = self.max_retries

        original_prompt = normalize_token_ids(prompt_ids)
        original_sampling = copy.deepcopy(sampling_params)

        recovery_id = f"{request_id}:{uuid4().hex[:8]}"

        retries = 0
        while True:
            progress_ctx = None
            prefix_for_call = original_prompt
            call_sampling = original_sampling
            if progress_on:
                sp_for_checkpoint = dict(original_sampling)
                sp_for_checkpoint["max_tokens"] = self._resolve_original_max_tokens(original_sampling, original_prompt)
                result = await VLLMProgressCheckPoint.create_or_resume(
                    store=self._progress_store,
                    run_id=self._run_id,
                    recovery_id=recovery_id,
                    prompt_token_ids=original_prompt,
                    sampling_params=sp_for_checkpoint,
                    model_weight_version=await self._weights_version(),
                    flush_token_interval=self._flush_token_interval(),
                    model_version_policy=self._model_version_policy(),
                )
                checkpoint = result.checkpoint
                progress_ctx = ProgressContext(checkpoint=checkpoint)
                prefix_for_call = checkpoint.resume_prefix_token_ids()
                call_sampling = copy.deepcopy(original_sampling)
                call_sampling["max_tokens"] = checkpoint.remaining_max_tokens()
                logger.info(
                    "[progress] run=%s, rid=%s, attempt=%d, outcome=%s, inherited_len=%d"
                    "prefix_len=%d remaining_max_tokens=%d (%s)",
                    self._run_id,
                    recovery_id,
                    result.attempt_id,
                    result.outcome.name,
                    result.inherited_prefix_len,
                    len(prefix_for_call),
                    checkpoint.remaining_max_tokens(),
                    result.failure_detail or "-",
                )
            try:
                output, server_id = await self._generate_once(
                    request_id,
                    prompt_ids=prefix_for_call,
                    sampling_params=call_sampling,
                    image_data=image_data,
                    video_data=video_data,
                    kwargs=dict(kwargs),
                    call_timeout_s=self.call_timeout_s,
                    translate_fault=True,
                    progress_ctx=progress_ctx,
                )
            except AllServersFailed:
                raise
            except ServerUnavailable as e:
                if e.server_id == "<lb>":
                    # The LB itself is unavailable; there is no healthy server to retry
                    raise
                retries += 1
                if retries > max_retries:
                    raise AllServersFailed(
                        f"RetryLLMServerClient: retries exhausted after {retries} attempts"
                    ) from None
                logger.warning(
                    "RetryLLMServerClient: server %s failed (%s), retries %d/%d",
                    e.server_id,
                    type(e.cause).__name__ if e.cause is not None else "server-fault",
                    retries,
                    max_retries,
                )
                continue

            gs = output.extra_fields.get("global_steps", None)
            if gs is not None:
                self._last_global_step = gs
            if progress_on and progress_ctx is not None:
                cp = progress_ctx.checkpoint
                inherited_ids = list(cp.cumulative_token_ids)
                new_ids = list(output.token_ids)
                lp_inherited = list(cp.cumulative_log_probs) if cp.cumulative_log_probs else []
                lp_new = list(output.log_probs) if output.log_probs else []
                re_inherited = cp.cumulative_routed_experts
                re_new = output.routed_experts
                if re_inherited is not None and re_new is not None:
                    routed_experts = torch.cat([re_inherited, re_new], dim=0)
                elif re_new is not None:
                    routed_experts = re_new
                else:
                    routed_experts = re_inherited
                final = TokenOutput(
                    token_ids=inherited_ids + new_ids,
                    log_probs=(lp_inherited + lp_new) or None,
                    routed_experts=routed_experts,
                    num_preempted=(cp.num_preempted or 0) + (output.num_preempted or 0),
                    stop_reason=output.stop_reason,
                    extra_fields=dict(output.extra_fields),
                )
            else:
                final = output
            output.extra_fields["llm_generate_attempts"] = retries + 1
            output.extra_fields["llm_generate_retries"] = retries
            return final


class FullyLLMServerClient(LLMServerClient):
    """FullyLLMServerClient supports resume generation on partial rollout, making rollout interruption
    invisible to the AgentLoop.

    With FT enabled (spec §L3), the while loop also catches ``ServerUnavailable`` from L2 and
    retries on a fresh server. The two paths in the loop:
        - abort path (``stop_reason='aborted'``): partial tokens kept; loop continues with
          ``prompt_ids + final_output.token_ids`` so vLLM picks up where it left off.
        - fault path (``ServerUnavailable``): partial tokens are gone with the dead server;
          ``final_output`` and ``sampling_params`` reset to original; loop retries from scratch
          on a new server.
    Retries exhausted ⇒ ``AllServersFailed``. When ``ft.enabled=False``, behaviour is bit-exact
    identical to the pre-FT implementation.

    Mode C (``progress.enabled=True``): before each attempt, ``create_or_resume`` decides the
    prefix from the persisted checkpoint. ``RESUMED`` → prefix = prompt + cumulative;
    ``DEGRADED_FRESH`` → prefix = prompt (degrades to plain retry). ``progress_ctx`` is
    forwarded to ``vLLMHttpServer.generate`` for in-stream ``ingest`` + flush.
    """

    def _ft_max_request_retries(self) -> int:
        try:
            return int(self.config.async_training.fault_tolerance.max_request_retries)
        except (AttributeError, KeyError):
            return 3  # FaultToleranceConfig default

    def _progress_enabled(self) -> bool:
        """True only when FT master switch AND progress sub-switch are both on."""
        if not self._ft_enabled():
            return False
        if self._progress_store is None or self._run_id is None:
            return False
        try:
            return bool(OmegaConf.select(self.config, "async_training.fault_tolerance.progress.enabled", default=False))
        except (AttributeError, KeyError, TypeError):
            return False

    def _flush_token_interval(self) -> int:
        try:
            return int(
                OmegaConf.select(
                    self.config, "async_training.fault_tolerance.progress.flush_token_interval", default=64
                )
            )
        except (AttributeError, KeyError, TypeError):
            return 64

    def _model_version_policy(self):
        from verl.workers.rollout.fault_tolerance import ModelVersionPolicy

        try:
            node = OmegaConf.select(self.config, "async_training.fault_tolerance.progress.model_version_policy")
            if node is not None:
                mode = node.get("mode", "exact") if hasattr(node, "get") else "exact"
                return ModelVersionPolicy(mode=mode)
        except (AttributeError, KeyError, TypeError):
            pass
        return ModelVersionPolicy(mode="exact")

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session. Also serves as
                ``recovery_id`` (stable across physical retries).
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.
            image_data (Optional[List[Any]]): Image data for the chat completion.
            video_data (Optional[List[Any]]): Video data for the chat completion.

        Returns:
            TokenOutput: token output

        Raises:
            AllServersFailed: (FT-only) all retries exhausted across the live server pool.
        """
        import copy

        from verl.workers.rollout.fault_tolerance import (
            AllServersFailed,
            ServerUnavailable,
        )

        ft_on = self._ft_enabled()
        progress_on = self._progress_enabled()
        max_retries = self._ft_max_request_retries()

        original_prompt = normalize_token_ids(prompt_ids)
        original_sampling = copy.deepcopy(sampling_params)  # rescue: deepcopy for nested objects (e.g. stop_token_ids)

        limit_key = None
        if "max_tokens" in sampling_params:
            limit_key = "max_tokens"
        elif "max_new_tokens" in sampling_params:
            limit_key = "max_new_tokens"
        original_max_tokens = sampling_params.get(limit_key) if limit_key else None

        final_output = TokenOutput(
            token_ids=[],
            log_probs=[],
            num_preempted=0,
        )
        min_global_steps, max_global_steps, global_steps = None, None, None
        retries = 0
        total_llm_generate_attempts = 0
        current_request_id = request_id  # rescue: new request_id on fault to avoid server-side ID collision

        # Token-continuation state (mode C).
        recovery_id = request_id  # stable across physical retries
        model_weight_version = None  # first attempt: None; updated from output after success

        while True:
            # ----- decide prefix + progress_ctx for this attempt -----
            checkpoint = None
            progress_ctx = None
            if progress_on:
                from verl.workers.rollout.fault_tolerance import (
                    ProgressContext,
                    VLLMProgressCheckPoint,
                )

                result = await VLLMProgressCheckPoint.create_or_resume(
                    store=self._progress_store,
                    run_id=self._run_id,
                    recovery_id=recovery_id,
                    prompt_token_ids=original_prompt,
                    sampling_params=original_sampling,
                    model_weight_version=model_weight_version,
                    flush_token_interval=self._flush_token_interval(),
                    model_version_policy=self._model_version_policy(),
                )
                checkpoint = result.checkpoint
                progress_ctx = ProgressContext(checkpoint=checkpoint)
                prefix_for_call = checkpoint.resume_prefix_token_ids()
                call_sampling = copy.deepcopy(original_sampling)
                if limit_key is not None:
                    call_sampling[limit_key] = checkpoint.remaining_max_tokens()
                # Seed final_output with inherited cumulative so that after fault reset
                # (where final_output was cleared) the persisted tokens are not lost.
                # On the abort path this is idempotent (cumulative == existing final_output).
                if checkpoint.inherited_prefix_len > 0:
                    final_output = TokenOutput(
                        token_ids=list(checkpoint.cumulative_token_ids),
                        log_probs=list(checkpoint.cumulative_log_probs) if checkpoint.cumulative_log_probs else [],
                        routed_experts=checkpoint.cumulative_routed_experts,
                        num_preempted=checkpoint.num_preempted,
                    )
            else:
                # Mode B: existing logic — prefix = prompt + final_output.token_ids
                progress_ctx = None
                prefix_for_call = original_prompt + final_output.token_ids
                call_sampling = sampling_params

            # 1. generate tokens — catch ServerUnavailable to retry on a fresh server
            try:
                output, _ = await self._generate_once(
                    request_id=current_request_id,
                    prompt_ids=prefix_for_call,
                    sampling_params=call_sampling,
                    image_data=image_data,
                    video_data=video_data,
                    kwargs={},
                    call_timeout_s=self._ft_call_timeout_s() if ft_on else None,
                    translate_fault=ft_on,
                    progress_ctx=progress_ctx,
                )
                total_llm_generate_attempts += 1
            except ServerUnavailable as e:
                if not ft_on or e.server_id == "<lb>":
                    raise  # bit-exact when FT is off (L2 wouldn't raise this anyway)
                retries += 1
                total_llm_generate_attempts += 1
                if retries >= max_retries:
                    raise AllServersFailed(
                        f"FullyLLMServerClient: retries exhausted after {retries} attempts"
                    ) from None
                # Fault reset: partial gone with dead server; restore originals.
                # The checkpoint for the next attempt is decided by create_or_resume.
                final_output = TokenOutput(token_ids=[], log_probs=[], num_preempted=0)
                sampling_params = copy.deepcopy(original_sampling)
                min_global_steps, max_global_steps, global_steps = None, None, None
                current_request_id = uuid4().hex  # rescue: avoid stale-ID collision on retry
                model_weight_version = None  # reset; next load_latest uses None
                continue

            # 2. merge output into final_output
            if progress_on and checkpoint is not None:
                # Mode C: under the copy model the client-side checkpoint only
                # carries the inherited prefix (the server ingests a serialized
                # copy and does not mutate this object), so append the new tokens
                # returned by this attempt.
                final_output.token_ids = list(checkpoint.cumulative_token_ids) + list(output.token_ids)
                if checkpoint.cumulative_log_probs is not None:
                    final_output.log_probs = list(checkpoint.cumulative_log_probs) + list(output.log_probs or [])
                else:
                    final_output.log_probs = list(output.log_probs or [])
                if checkpoint.cumulative_routed_experts is not None:
                    if output.routed_experts is not None:
                        final_output.routed_experts = torch.cat(
                            [checkpoint.cumulative_routed_experts, output.routed_experts], dim=0
                        )
                    else:
                        final_output.routed_experts = checkpoint.cumulative_routed_experts
                else:
                    final_output.routed_experts = output.routed_experts
                final_output.num_preempted = (checkpoint.num_preempted or 0) + (output.num_preempted or 0)
                final_output.stop_reason = output.stop_reason
            else:
                # Mode B: extend final_output with this attempt's new tokens
                final_output.token_ids.extend(output.token_ids)
                if output.log_probs is not None:
                    final_output.log_probs.extend(output.log_probs)
                # On partial rollout resume the model version may differ, so keep
                # existing routing and only append routing for newly generated tokens.
                if output.routed_experts is not None and len(output.token_ids) > 0:
                    if final_output.routed_experts is None:
                        final_output.routed_experts = output.routed_experts
                    else:
                        final_output.routed_experts = torch.cat(
                            [final_output.routed_experts, output.routed_experts[-len(output.token_ids) :]],
                            dim=0,
                        )
                if output.num_preempted is not None:
                    final_output.num_preempted += output.num_preempted
                final_output.stop_reason = output.stop_reason

            # update model weights version (used by next create_or_resume's version check)
            global_steps = output.extra_fields.get("global_steps", None)
            if min_global_steps is None:
                min_global_steps = global_steps
            max_global_steps = global_steps
            model_weight_version = global_steps

            # 3. update max_new_tokens; truncate (FT only) to spec contract: final ≤ original_max_tokens
            if original_max_tokens is not None and limit_key is not None:
                if not progress_on:
                    # Mode B: mutate sampling_params for next iteration's budget
                    sampling_params[limit_key] = original_max_tokens - len(final_output.token_ids)
                if len(final_output.token_ids) >= original_max_tokens:
                    if ft_on and len(final_output.token_ids) > original_max_tokens:
                        # Hard cap: truncate any overshoot from the last server call (F06).
                        # Gated on ft_on so the pre-FT path is bit-exact.
                        final_output.token_ids = final_output.token_ids[:original_max_tokens]
                        if final_output.log_probs:
                            final_output.log_probs = final_output.log_probs[:original_max_tokens]
                    final_output.stop_reason = "length"
                    break

            # 4. check stop reason
            partial_rollout_enabled = False
            try:
                partial_rollout_enabled = bool(self.config.async_training.partial_rollout)
            except (AttributeError, KeyError):
                pass
            if output.stop_reason not in ("aborted", "abort") or not partial_rollout_enabled:
                break
        final_output.extra_fields["global_steps"] = global_steps
        final_output.extra_fields["min_global_steps"] = min_global_steps
        final_output.extra_fields["max_global_steps"] = max_global_steps
        final_output.extra_fields["llm_generate_attempts"] = total_llm_generate_attempts
        final_output.extra_fields["llm_generate_retries"] = retries
        return final_output


class LLMServerManager:
    """LLMServerManager is responsible for:
    - Launch server replicas
    - Launch global load balancer
    - Elastic launch/tear-down new replicas

    Args:
        config (DictConfig): Config for the trainer entrypoint.
        worker_group (RayWorkerGroup): Worker group for the server replicas. If not none, init hybrid server,
            else init standalone server with a new resource pool.
        rollout_resource_pool (RayResourcePool): Resource pool for the server replicas, only needed for TensorRT-LLM.
    """

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool: RayResourcePool = None,
    ):
        self.config = config
        self.rollout_config = config.actor_rollout_ref.rollout
        self.model_config = config.actor_rollout_ref.model
        self.worker_group = worker_group
        self.rollout_resource_pool = rollout_resource_pool

        assert worker_group is not None or self.rollout_config.nnodes > 0, "nnodes must be > 0 in standalone mode"

        # Process-level run_id for token-continuation checkpoint isolation.
        # Prevents cross-process stale checkpoint misuse (spec §3.1).
        self.run_id = f"{socket.gethostname()}_{os.getpid()}_{uuid4().hex[:8]}"
        # RolloutProgressStoreActor handle; None unless mode C is assembled.
        self._progress_store = None
        self.max_model_len = self._resolve_max_model_len()

        # for recipe to change
        if not hasattr(self, "rollout_replica_class"):
            self.rollout_replica_class = get_rollout_replica_class(
                self.rollout_config.name,
                disaggregation_enabled=self.rollout_config.disaggregation.enabled,
            )

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create the LLMServerManager."""
        instance = cls(*args, **kwargs)
        await instance._initialize_llm_servers()
        await instance._init_global_load_balancer()
        return instance

    async def _initialize_llm_servers(self):
        """Initialize the LLM server replicas."""
        rollout_world_size = (
            self.rollout_config.tensor_model_parallel_size
            * self.rollout_config.data_parallel_size
            * self.rollout_config.pipeline_model_parallel_size
        )
        # PD inflates per-replica footprint; miss this and init_hybrid slices
        # past worker_group → empty workers on replica_rank>=1.
        disagg = getattr(self.rollout_config, "disaggregation", None)
        if disagg is not None and getattr(disagg, "enabled", False):
            prefill_tp = self.rollout_config.tensor_model_parallel_size
            # Inline decode_tp default: OmegaConf/Ray serialization drops dataclass methods.
            decode_tp = (
                disagg.decode_tensor_model_parallel_size
                if disagg.decode_tensor_model_parallel_size is not None
                else prefill_tp
            )
            rollout_world_size = (
                (prefill_tp * disagg.prefill_replicas + decode_tp * disagg.decode_replicas)
                * self.rollout_config.data_parallel_size
                * self.rollout_config.pipeline_model_parallel_size
            )
        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.rollout_config.n_gpus_per_node * self.rollout_config.nnodes
        )
        num_replicas = world_size // rollout_world_size

        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=replica_rank,
                config=self.rollout_config,
                model_config=self.model_config,
                gpus_per_node=self.rollout_config.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]

        if self.worker_group and self.rollout_config.name != "trtllm":
            await asyncio.gather(*[server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        # TODO: unify trtllm to init_hybrid
        elif self.worker_group and self.rollout_config.name == "trtllm":
            await asyncio.gather(
                *[
                    server.init_hybrid_colocated(self.worker_group, self.rollout_resource_pool)
                    for server in self.rollout_replicas
                ]
            )
        else:
            await asyncio.gather(*[server.init_standalone() for server in self.rollout_replicas])

        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]
        print(f"LLMServerManager: {self.server_addresses}")

        # Update Prometheus configuration with server addresses
        if self.rollout_config.prometheus.enable:
            if self.rollout_config.disable_log_stats:
                raise ValueError("PROMETHEUS needs disable_log_stats==False, but it is currently True.")
            update_prometheus_config(self.rollout_config.prometheus, self.server_addresses, self.rollout_config.name)

    async def _init_global_load_balancer(self) -> None:
        ft_on = False
        try:
            ft_on = bool(self.config.async_training.fault_tolerance.enabled)
        except (AttributeError, KeyError):
            pass
        self.global_load_balancer = GlobalRequestLoadBalancer.remote(
            servers=dict(zip(self.server_addresses, self.server_handles, strict=True)),
            max_cache_size=DEFAULT_ROUTING_CACHE_SIZE,
            enable_fault_tolerance=ft_on,
        )

    def _resolve_max_model_len(self) -> Optional[int]:
        try:
            mml = self.rollout_config.get("max_model_length", None)
            if mml is not None:
                return int(mml)
        except (AttributeError, TypeError, ValueError):
            pass
        try:
            mpe = getattr(self.model_config, "max_position_embeddings", None)
            if mpe is not None:
                return int(mpe)
        except (AttributeError, TypeError, ValueError):
            pass
        return None

    async def _init_progress_store(self, progress_cfg) -> None:
        """Mode C: create and initialise the StoreActor. Called during FT assembly."""
        from verl.workers.rollout.fault_tolerance import RolloutProgressStoreActor

        self._progress_store = RolloutProgressStoreActor.remote()
        await self._progress_store.init.remote(progress_cfg)

    def get_client(
        self,
        fully_async: bool = False,
        retry: bool = False,
    ) -> LLMServerClient:
        """Get the LLMServerClient to request LLM server replicas.

        Args:
            fully_async (bool): Whether to return the FullyLLMServerClient.
            retry (bool): Whether to retry on server unavailability.
        """
        servers = dict(zip(self.server_addresses, self.server_handles, strict=True))
        common = dict(
            config=self.config,
            servers=servers,
            load_balancer_handle=self.global_load_balancer,
            run_id=self.run_id,
            progress_store=self._progress_store,
            max_model_len=self.max_model_len,
        )
        if fully_async:
            return FullyLLMServerClient(**common)
        if retry:
            return RetryLLMServerClient(**common)
        return LLMServerClient(config=self.config, servers=servers, load_balancer_handle=self.global_load_balancer)

    def get_addresses(self) -> list[str]:
        """Get the OpenAI chat completion API http addresses of the LLM server replicas."""
        return self.server_addresses

    def get_replicas(self) -> list[RolloutReplica]:
        """Get the LLM server replicas."""
        return self.rollout_replicas

    async def spawn_replacement(self, dead_id: str) -> "RolloutReplica":
        """Spawn a replacement RolloutReplica for `dead_id` (standalone mode only).

        Caller (on_spawn_success) wires the new replica into CKE / LB / Supervisor.
        """
        log = logging.getLogger(__name__)
        if self.worker_group is not None:
            raise RuntimeError("spawn_replacement is standalone-mode only")

        # Look up dead rank via server_addresses (not rollout_replicas — CKE may have already pruned).
        if dead_id not in self.server_addresses:
            raise ValueError(f"unknown dead_id={dead_id!r}; have {self.server_addresses}")
        dead_rank = self.server_addresses.index(dead_id)
        log.warning("[FT] spawn_replacement: dead_id=%s rank=%s — beginning respawn", dead_id, dead_rank)

        self.rollout_replicas = [r for r in self.rollout_replicas if r._server_address != dead_id]
        kept = {r._server_address for r in self.rollout_replicas}
        self.server_addresses = [a for a in self.server_addresses if a in kept]
        self.server_handles = [r._server_handle for r in self.rollout_replicas]

        await self._reclaim_ray_resources(dead_rank, log)

        # A killed named Ray actor can remain in Ray's name registry briefly.
        # Do not reuse the dead replica's fixed name prefix: on multi-node
        # replicas that window can otherwise surface as ActorAlreadyExistsError
        # while creating a worker on a non-zero local rank (for example
        # ``...CheckpointEngineWorker0:1``).
        recovery_suffix = f"recovery_{uuid4().hex}"
        new_replica = self.rollout_replica_class(
            replica_rank=dead_rank,
            config=self.rollout_config,
            model_config=self.model_config,
            gpus_per_node=self.rollout_config.n_gpus_per_node,
            name_suffix=recovery_suffix,
        )
        await new_replica.init_standalone()
        if not bool(await asyncio.wait_for(new_replica.health(), timeout=30.0)):
            raise RuntimeError(f"new replica for {dead_id} (rank={dead_rank}) failed health check")

        self.rollout_replicas.append(new_replica)
        self.server_addresses.append(new_replica._server_address)
        self.server_handles.append(new_replica._server_handle)
        log.warning(
            "[FT] spawn_replacement: new replica %s (rank=%s) up and healthy",
            new_replica._server_address,
            dead_rank,
        )
        return new_replica

    async def _reclaim_ray_resources(self, dead_rank: int, log: logging.Logger) -> None:
        """Kill stale actors, remove their placement groups, then await a full replica's resources."""
        from ray.util.placement_group import (
            get_placement_group,
            placement_group_table,
            remove_placement_group,
        )

        # Named-actor registry doesn't auto-clean on chaos kill; try known patterns.
        # (ray.util.state.list_actors needs the dashboard, which our `ray start` skips.)
        # RayWorkerGroup names workers as
        # ``{name_prefix}{class_name}{pg_index}:{local_rank}``.  The previous
        # cleanup only looked for ``...{i}:0`` and missed workers such as
        # ``rollout_standalone_2CheckpointEngineWorker0:1`` on the second
        # local rank/node.  Cover every placement group and local rank of the
        # failed standalone replica.
        nnodes = int(self.rollout_config.nnodes)
        local_world_size = int(self.rollout_config.n_gpus_per_node)
        candidate_names = (
            [
                f"rollout_standalone_{dead_rank}{cls}{pg_idx}:{local_rank}"
                for cls in ("CheckpointEngineWorker", "vLLMHttpServer")
                for pg_idx in range(nnodes)
                for local_rank in range(local_world_size)
            ]
            + [f"vllm_server_{dead_rank}_{i}" for i in range(nnodes)]
            + [f"rollout_standalone_{dead_rank}"]
        )
        for name in candidate_names:
            try:
                handle = ray.get_actor(name)
            except ValueError:
                continue
            ray.kill(handle, no_restart=True)
            log.warning("[FT] spawn_replacement: ray.kill stale actor %s", name)

        # PG names have a runtime suffix, so match by prefix.
        pg_prefix = f"rollout_pool_{dead_rank}"
        for pg_id, info in placement_group_table().items():
            name = info.get("name", "")
            if not name.startswith(pg_prefix):
                continue
            try:
                pg = get_placement_group(name)
            except ValueError:
                continue
            log.warning("[FT] spawn_replacement: removing leftover placement_group %s", name)
            remove_placement_group(pg)

        resource_name = get_resource_name()
        rollout_world_size = (
            self.rollout_config.tensor_model_parallel_size
            * self.rollout_config.data_parallel_size
            * self.rollout_config.pipeline_model_parallel_size
        ) // nnodes
        required_accelerators = rollout_world_size
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 180.0
        last_seen = -1.0
        while True:
            accelerator_avail = float(ray.available_resources().get(resource_name, 0.0))
            if accelerator_avail >= required_accelerators:
                log.warning(
                    "[FT] spawn_replacement: %.1f/%d %s resources free; proceeding to init",
                    accelerator_avail,
                    required_accelerators,
                    resource_name,
                )
                return
            if loop.time() > deadline:
                raise RuntimeError(
                    "dead replica resources not freed after 180s "
                    f"(available {resource_name}={accelerator_avail}, required={required_accelerators})"
                )
            if abs(accelerator_avail - last_seen) > 0.01:
                log.warning(
                    "[FT] spawn_replacement: waiting for %d %s resources, available=%.2f",
                    required_accelerators,
                    resource_name,
                    accelerator_avail,
                )
                last_seen = accelerator_avail
            await asyncio.sleep(2)

    @auto_await
    async def clear_kv_cache(self):
        """Clear all rollout kv cache, but don`t sleep."""
        await asyncio.gather(*[replica.clear_kv_cache() for replica in self.rollout_replicas])

    @auto_await
    async def start_profile(self, **kwargs):
        """Start profiling on all rollout replicas."""
        await asyncio.gather(*[replica.start_profile(**kwargs) for replica in self.rollout_replicas])

    @auto_await
    async def stop_profile(self):
        """Stop profiling on all rollout replicas."""
        await asyncio.gather(*[replica.stop_profile() for replica in self.rollout_replicas])
