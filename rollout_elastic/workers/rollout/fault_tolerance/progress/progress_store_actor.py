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
from __future__ import annotations

import ray

from verl.workers.rollout.fault_tolerance.progress.progress_store import RolloutProgressStore
from verl.workers.rollout.fault_tolerance.progress.types import (
    CheckPointPayLoad,
    GCStats,
    LoadResult,
    ModelVersionPolicy,
    ProgressConfig,
)


@ray.remote
class RolloutProgressStoreActor:
    def __init__(self) -> None:
        self._core: RolloutProgressStore | None = None

    def _get_core(self) -> RolloutProgressStore:
        if self._core is None:
            raise RuntimeError("RolloutProgressStoreActor.init() must be called first")
        return self._core

    async def init(self, config: ProgressConfig) -> None:
        self._core = RolloutProgressStore(config)
        await self._core.init(config)

    def preflight_dir(self) -> None:
        self._get_core().preflight_dir()

    async def shutdown(self) -> None:
        await self._get_core().shutdown()

    async def save(self, payload: CheckPointPayLoad) -> None:
        await self._get_core().save(payload)

    async def load_latest(
        self,
        run_id: str,
        recovery_id: str,
        requested_model_version: str | None,
        policy: ModelVersionPolicy,
    ) -> LoadResult:
        return await self._get_core().load_latest(run_id, recovery_id, requested_model_version, policy)

    async def mark_superseded(self, run_id: str, recovery_id: str, attempt_id: int) -> None:
        await self._get_core().mark_superseded(run_id, recovery_id, attempt_id)

    async def periodic_collect(self) -> GCStats:
        return await self._get_core().periodic_collect()

    async def get_stats(self) -> dict:
        return await self._get_core().get_stats()
