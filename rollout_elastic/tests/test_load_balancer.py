"""Run in the recipe's verl/Ray environment with pytest.

These tests use real actor RPCs, including the patched manager creation path.
Server handles are opaque strings because the LB never invokes the servers.
"""

from types import SimpleNamespace

import pytest

ray = pytest.importorskip("ray")


@pytest.fixture
def ray_runtime():
    started = not ray.is_initialized()
    if started:
        ray.init(num_cpus=2, include_dashboard=False)
    yield
    if started:
        ray.shutdown()


@pytest.mark.parametrize("ft_enabled", [False, True])
def test_manager_lb_routing(ray_runtime, ft_enabled):
    import asyncio

    from rollout_elastic.patch.llm_server import LLMServerManager

    manager = object.__new__(LLMServerManager)
    manager.config = SimpleNamespace(
        async_training=SimpleNamespace(
            fault_tolerance=SimpleNamespace(enabled=ft_enabled)
        )
    )
    manager.server_addresses = ["a", "b"]
    manager.server_handles = ["handle-a", "handle-b"]

    async def initialize():
        await asyncio.wait_for(manager._init_global_load_balancer(), timeout=30)

    try:
        asyncio.run(initialize())
        lb = manager.global_load_balancer

        def call(name, *args):
            return ray.get(getattr(lb, name).remote(*args), timeout=15)

        assert call("acquire_server", "sticky") == "a"
        call("release_server", "a")
        call("mark_failed", "a")
        assert call("acquire_server", "sticky") == "b"
        assert call("acquire_server", "fresh") == "b"
        call("release_server", "b")
        call("release_server", "b")

        call("add_servers", {"c": "handle-c"})
        assert call("get_server_handle", "c") == "handle-c"
        call("remove_servers", ["b"])
        assert call("acquire_server", "sticky") == "c"
        call("release_server", "c")

        call("add_servers", {"a": "replacement-a"})
        call("remove_servers", ["c"])
        assert call("acquire_server", "after-replacement") == "a"
        assert call("get_server_handle", "a") == "replacement-a"
        call("release_server", "a")

        if ft_enabled:
            call("release_server", "a")
        else:
            with pytest.raises(ray.exceptions.RayTaskError, match="no inflight"):
                call("release_server", "a")
    finally:
        lb = getattr(manager, "global_load_balancer", None)
        if lb is not None:
            ray.kill(lb)
