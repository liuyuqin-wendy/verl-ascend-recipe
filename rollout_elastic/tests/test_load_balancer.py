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


@pytest.mark.parametrize("ft_enabled", [False, True, None])
def test_manager_lb_routing(ray_runtime, ft_enabled):
    import asyncio

    from rollout_elastic.patch.llm_server import FullyLLMServerClient, LLMServerClient, LLMServerManager

    manager = object.__new__(LLMServerManager)
    manager.config = SimpleNamespace(
        async_training=SimpleNamespace(
            fault_tolerance=SimpleNamespace(enabled=ft_enabled)
        )
    )
    if ft_enabled is None:
        manager.config = SimpleNamespace()
    manager.server_addresses = ["a", "b"]
    manager.server_handles = ["handle-a", "handle-b"]
    manager.run_id = "test-run"
    manager._progress_store = None
    manager.max_model_len = None

    async def initialize():
        await asyncio.wait_for(manager._init_global_load_balancer(), timeout=30)

    try:
        asyncio.run(initialize())
        lb = manager.global_load_balancer

        def call(name, *args):
            return ray.get(getattr(lb, name).remote(*args), timeout=15)

        assert call("acquire_server", "sticky") == "a"
        call("release_server", "a")
        if not ft_enabled:
            assert call("acquire_server", "sticky") == "a"
            call("release_server", "a")
            with pytest.raises(ray.exceptions.RayTaskError, match="no inflight"):
                call("release_server", "a")
            # Native LB exposes no fault-marking method and no elastic add path.
            assert not hasattr(lb, "mark_failed")
            with pytest.raises(ray.exceptions.RayTaskError, match="Not implemented"):
                call("add_servers", {"c": "handle-c"})
            for fully_async, expected in ((False, LLMServerClient), (True, FullyLLMServerClient)):
                client = manager.get_client(fully_async=fully_async, retry=True)
                assert type(client) is expected
                asyncio.run(client._mark_server_failed("a"))
                client._server_id_to_handle.clear()
                with pytest.raises(RuntimeError, match="Unknown server_id"):
                    asyncio.run(client._acquire_server("unknown-handle"))
            return

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

        call("release_server", "a")
    finally:
        lb = getattr(manager, "global_load_balancer", None)
        if lb is not None:
            ray.kill(lb)


@pytest.mark.parametrize("ft_enabled", [False, True, None])
def test_client_lb_interface(ft_enabled):
    import asyncio
    from unittest.mock import AsyncMock, Mock

    from rollout_elastic.patch.llm_server import LLMServerClient

    config = SimpleNamespace()
    if ft_enabled is not None:
        config.async_training = SimpleNamespace(
            fault_tolerance=SimpleNamespace(enabled=ft_enabled)
        )
    lb = SimpleNamespace(
        acquire_server=SimpleNamespace(remote=AsyncMock(return_value="a")),
        get_server_handle=SimpleNamespace(remote=AsyncMock(return_value="replacement-a")),
        mark_failed=SimpleNamespace(remote=AsyncMock()),
        release_server=SimpleNamespace(remote=Mock(side_effect=RuntimeError("submit failed"))),
    )
    client = LLMServerClient(config=config, servers={}, load_balancer_handle=lb)
    if ft_enabled:
        assert asyncio.run(client._acquire_server("request")) == ("a", "replacement-a")
        lb.get_server_handle.remote.assert_awaited_once_with(server_id="a")
        client._release_server("a")
    else:
        with pytest.raises(RuntimeError, match="Unknown server_id"):
            asyncio.run(client._acquire_server("request"))
        lb.get_server_handle.remote.assert_not_called()
        with pytest.raises(RuntimeError, match="submit failed"):
            client._release_server("a")

    asyncio.run(client._mark_server_failed("a"))
    if ft_enabled:
        lb.mark_failed.remote.assert_awaited_once_with(server_id="a")
    else:
        lb.mark_failed.remote.assert_not_called()
