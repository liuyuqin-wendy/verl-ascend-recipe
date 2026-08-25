"""Fault tolerance exceptions and the `is_transient_fault` equivalence class.

Per spec §硬规则: production code must catch via `is_transient_fault(exc)`,
not by specific Ray exception class — Ray wraps/subclasses unpredictably across
versions.
"""
from __future__ import annotations

import asyncio
from typing import Optional

import ray.exceptions


def _resolve_ray_exc(name: str):
    """Some Ray exception classes (e.g. ActorUnavailableError) are version-gated."""
    return getattr(ray.exceptions, name, None)


# Equivalence class: any exception in here means "transient fault, retry"
RAY_FAULT_EXCEPTIONS: tuple = tuple(
    e
    for e in [
        _resolve_ray_exc("RayActorError"),
        _resolve_ray_exc("ActorDiedError"),
        _resolve_ray_exc("ActorUnavailableError"),
        _resolve_ray_exc("WorkerCrashedError"),
        _resolve_ray_exc("OutOfMemoryError"),
        _resolve_ray_exc("GetTimeoutError"),
        ConnectionError,
        asyncio.TimeoutError,
    ]
    if e is not None
)


def is_transient_fault(exc: BaseException) -> bool:
    """Whether this exception should trigger transient fault retry.

    Catches direct instances of RAY_FAULT_EXCEPTIONS plus Ray's wrapper
    `RayTaskError` whose `.cause` / `.__cause__` is a transient fault.
    """
    if isinstance(exc, RAY_FAULT_EXCEPTIONS):
        return True
    ray_task_error = _resolve_ray_exc("RayTaskError")
    if ray_task_error is not None and isinstance(exc, ray_task_error):
        cause = getattr(exc, "cause", None) or getattr(exc, "__cause__", None)
        if cause is not None and isinstance(cause, RAY_FAULT_EXCEPTIONS):
            return True
    return False


class ServerUnavailable(Exception):
    """A single server is unavailable; client should switch and retry.

    Raised by L2 (`LLMServerClient.generate`); caught by L3 (`FullyLLMServerClient`).
    """

    def __init__(self, server_id: str, cause: Optional[BaseException] = None) -> None:
        self.server_id = server_id
        self.cause = cause
        super().__init__(f"server {server_id} unavailable: {cause!r}")

    def __reduce__(self):
        # Ray propagates exceptions via pickle. Default Exception.__reduce__ uses
        # `self.args` which we overwrote with a formatted string in __init__, so
        # without this override an unpickled instance loses .server_id / .cause.
        return (type(self), (self.server_id, self.cause))


class AllServersFailed(Exception):
    """L3 retry budget exhausted across all available servers.

    Raised by L3; caught by L4 (`AgentLoopWorker`) which decides batch policy.
    """


class BatchMostlyFailed(Exception):
    """L4 saw too many per-prompt failures (below min_ok_ratio).

    Raised by L4; propagates to L5+. Trainer treats as a skipped step.
    """


class BuildGroupPartialFailure(Exception):
    """CheckpointEngineManager.build_process_group dropped dead members.

    Carries the dropped worker handles so the caller can clean up.
    """

    def __init__(self, dead_workers: list, stage: str | None = None) -> None:
        self.dead_workers = list(dead_workers)
        self.stage = stage
        super().__init__(
            f"build_process_group dropped {len(self.dead_workers)} dead members"
        )

    def __reduce__(self):
        return (type(self), (self.dead_workers, self.stage))
