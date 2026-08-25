"""Fault tolerance for verl asynchronous rollout (P1/P2).

See `experiments/docs/elastic_rollout_spec.md` and
`experiments/docs/fully_async_recovery_design.md` for the design.
"""
from verl.workers.rollout.fault_tolerance.aio import lenient_gather
from verl.workers.rollout.fault_tolerance.batch import filter_partial_batch
from verl.workers.rollout.fault_tolerance.exceptions import (
    RAY_FAULT_EXCEPTIONS,
    AllServersFailed,
    BatchMostlyFailed,
    BuildGroupPartialFailure,
    ServerUnavailable,
    is_transient_fault,
)
from verl.workers.rollout.fault_tolerance.group_membership import split_refs_by_timeout
from verl.workers.rollout.fault_tolerance.progress import (
    CheckPointPayLoad,
    GCStats,
    LoadFailure,
    LoadResult,
    ModelVersionPolicy,
    ProgressConfig,
    ProgressContext,
    ResumeOutcome,
    ResumeResult,
    RolloutProgressStoreActor,
    VLLMProgressCheckPoint,
)
from verl.workers.rollout.fault_tolerance.supervisor import (
    HeartbeatTracker,
    Supervisor,
    make_on_dead,
)
from verl.workers.rollout.fault_tolerance.supervisor_thread import ThreadedSupervisor
from verl.workers.rollout.fault_tolerance.types import FaultToleranceConfig

__all__ = [
    "AllServersFailed",
    "BatchMostlyFailed",
    "BuildGroupPartialFailure",
    "CheckPointPayLoad",
    "FaultToleranceConfig",
    "GCStats",
    "HeartbeatTracker",
    "LoadFailure",
    "LoadResult",
    "ModelVersionPolicy",
    "ProgressConfig",
    "ProgressContext",
    "RAY_FAULT_EXCEPTIONS",
    "ResumeOutcome",
    "ResumeResult",
    "RolloutProgressStoreActor",
    "ServerUnavailable",
    "Supervisor",
    "ThreadedSupervisor",
    "VLLMProgressCheckPoint",
    "filter_partial_batch",
    "is_transient_fault",
    "lenient_gather",
    "make_on_dead",
    "split_refs_by_timeout",
]
