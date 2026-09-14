"""Test production actor patching and rebinding with lightweight native classes.

Ray is real; training dependencies and native constructors are stand-ins. The
tests cover patch installation and RPC dispatch, not training or recovery.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import runpy
import subprocess
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch as mock_patch

PATCH_DIR = Path(__file__).parents[1] / "patch"


def _build_patched_actors(ray):
    spec = importlib.util.spec_from_file_location("actor_test_core", PATCH_DIR / "_core.py")
    assert spec is not None and spec.loader is not None
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)

    class ConfigAccess:
        @staticmethod
        def select(config, path, default=None):
            value = config
            for part in path.split("."):
                value = getattr(value, part, default)
            return value

    class Separate:
        def __init__(self, config):
            self.config = config

    class NativeRollouter(Separate):
        def __init__(self, config):
            super().__init__(config)
            self.llm_server_manager = SimpleNamespace(global_load_balancer="native-lb")

        async def _init_async_rollout_manager(self):
            return "native-manager"

        async def fit(self):
            return "native-fit"

        def ft_initialized(self):
            return hasattr(self, "_ft_supervisor"), hasattr(self, "_trainer_handle")

    class NativeTrainer(Separate):
        def _setup_checkpoint_manager(self, rollouter):
            return "native-checkpoint", rollouter

    class NativeTaskRunner:
        def _initialize_components(self, config):
            return "native-main"

    original = {
        "FullyAsyncRollouter": ray.remote(num_cpus=10, max_concurrency=100)(NativeRollouter),
        "FullyAsyncTrainer": ray.remote(num_cpus=10)(NativeTrainer),
        "FullyAsyncTaskRunner": ray.remote(num_cpus=1)(NativeTaskRunner),
    }
    main_module = SimpleNamespace(**original)
    rollouter_module = SimpleNamespace(FullyAsyncRollouter=original["FullyAsyncRollouter"])
    trainer_module = SimpleNamespace(FullyAsyncTrainer=original["FullyAsyncTrainer"])
    namespace = dict(
        original,
        __name__="fully_async_patch_test",
        ray=ray,
        OmegaConf=ConfigAccess,
        SeparateRayPPOTrainer=Separate,
        fully_async_main=main_module,
        fully_async_rollouter=rollouter_module,
        fully_async_trainer=trainer_module,
        add=core.add,
        patch=core.patch,
        wrap=core.wrap,
        unwrap_ray_remote=core.unwrap_ray_remote,
    )
    tree = ast.parse((PATCH_DIR / "experimental.py").read_text(encoding="utf-8"))
    nodes = []
    finalizing = False
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "FullyAsyncRollouter" for target in node.targets
        ):
            finalizing = True
        if finalizing:
            nodes.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.name == "_sep_trainer_init" or node.name.startswith(("_rollouter_", "_async_trainer_", "_async_main_"))
        ):
            nodes.append(node)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(tree, str(PATCH_DIR / "experimental.py"), "exec"), namespace)
    return namespace, original


class FullyAsyncRayPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("ray") is None:
            raise unittest.SkipTest("Ray is not installed")
        import ray

        cls.ray = ray
        cls.started_here = not ray.is_initialized()
        if cls.started_here:
            ray.init(num_cpus=1, include_dashboard=False, log_to_driver=False)
        try:
            cls.patched, cls.original = _build_patched_actors(ray)
        except Exception:
            if cls.started_here:
                ray.shutdown()
            raise

    @classmethod
    def tearDownClass(cls):
        if cls.started_here:
            cls.ray.shutdown()

    def test_creation_aliases_options_and_method_tables(self):
        options = {
            "FullyAsyncRollouter": {"num_cpus": 10, "max_concurrency": 100},
            "FullyAsyncTrainer": {"num_cpus": 10},
            "FullyAsyncTaskRunner": {"num_cpus": 1},
        }
        methods = {
            "FullyAsyncRollouter": (
                "get_load_balancer",
                "init_ft_supervisor",
                "report_sync_failure",
                "promote_synced_replica",
            ),
            "FullyAsyncTrainer": (
                "_on_replica_dead_from_supervisor",
                "_on_replica_added_from_supervisor",
            ),
            "FullyAsyncTaskRunner": ("_initialize_components",),
        }
        for name, actor_options in options.items():
            with self.subTest(actor=name):
                actor = self.patched[name]
                original = self.original[name]
                self.assertIsNot(actor, original)
                self.assertIs(getattr(self.patched["fully_async_main"], name), actor)
                self.assertEqual(actor._default_options, actor_options)
                self.assertIs(actor.__ray_actor_class__, original.__ray_actor_class__)
                self.assertIsNot(
                    actor.__ray_metadata__.method_meta,
                    original.__ray_metadata__.method_meta,
                )
                self.assertNotEqual(
                    actor.__ray_metadata__.actor_creation_function_descriptor.function_id,
                    original.__ray_metadata__.actor_creation_function_descriptor.function_id,
                )
                self.assertLessEqual(
                    set(methods[name]),
                    actor.__ray_metadata__.method_meta.methods.keys(),
                )
        self.assertIs(
            self.patched["fully_async_rollouter"].FullyAsyncRollouter,
            self.patched["FullyAsyncRollouter"],
        )
        self.assertIs(
            self.patched["fully_async_trainer"].FullyAsyncTrainer,
            self.patched["FullyAsyncTrainer"],
        )

    def test_new_actors_execute_constructor_and_added_rpc(self):
        for ft_enabled in (None, False, True):
            with self.subTest(ft_enabled=ft_enabled):
                config = SimpleNamespace()
                if ft_enabled is not None:
                    config.async_training = SimpleNamespace(fault_tolerance=SimpleNamespace(enabled=ft_enabled))
                actors = []
                try:
                    rollouter = self.patched["FullyAsyncRollouter"].options(num_cpus=0).remote(config)
                    actors.append(rollouter)
                    self.assertEqual(
                        self.ray.get(rollouter.ft_initialized.remote(), timeout=20),
                        (bool(ft_enabled), bool(ft_enabled)),
                    )
                    self.assertEqual(
                        self.ray.get(rollouter.get_load_balancer.remote(), timeout=20),
                        "native-lb",
                    )
                    self.assertIsNone(self.ray.get(rollouter.report_sync_failure.remote("replica"), timeout=20))
                    self.assertFalse(
                        self.ray.get(
                            rollouter.promote_synced_replica.remote("replica", {}, 1, 1),
                            timeout=20,
                        )
                    )
                    if not ft_enabled:
                        self.assertEqual(
                            self.ray.get(rollouter.fit.remote(), timeout=20),
                            "native-fit",
                        )
                        self.assertEqual(
                            self.ray.get(
                                rollouter._init_async_rollout_manager.remote(),
                                timeout=20,
                            ),
                            "native-manager",
                        )
                        trainer = self.patched["FullyAsyncTrainer"].options(num_cpus=0).remote(config)
                        runner = self.patched["FullyAsyncTaskRunner"].options(num_cpus=0).remote()
                        actors.extend([trainer, runner])
                        self.assertEqual(
                            self.ray.get(
                                trainer._setup_checkpoint_manager.remote("rollouter"),
                                timeout=20,
                            ),
                            ("native-checkpoint", "rollouter"),
                        )
                        self.assertEqual(
                            self.ray.get(
                                runner._initialize_components.remote(config),
                                timeout=20,
                            ),
                            "native-main",
                        )
                finally:
                    for actor in actors:
                        self.ray.kill(actor)

    def test_downstream_can_derive_from_rebound_plain_classes(self):
        for name in (
            "FullyAsyncRollouter",
            "FullyAsyncTrainer",
            "FullyAsyncTaskRunner",
        ):
            with self.subTest(actor=name):

                class Downstream(self.patched[name].__ray_actor_class__):
                    pass

                actor = self.ray.remote(Downstream)
                self.assertEqual(
                    actor.__ray_actor_class__.__bases__,
                    (self.patched[name].__ray_actor_class__,),
                )
                self.assertLessEqual(
                    self.patched[name].__ray_metadata__.method_meta.methods.keys(),
                    actor.__ray_metadata__.method_meta.methods.keys(),
                )


class FullyAsyncLauncherTests(unittest.TestCase):
    def _subprocess_env(self):
        if importlib.util.find_spec("verl") is None:
            self.skipTest("verl is not installed")
        env = os.environ.copy()
        env["VERL_USE_EXTERNAL_MODULES"] = "rollout_elastic.patch"
        return env

    def test_launcher_calls_canonical_hydra_main(self):
        module_name = "verl.experimental.fully_async_policy.fully_async_main"
        module = ModuleType(module_name)
        module.main = Mock()
        with mock_patch.dict(sys.modules, {module_name: module}):
            runpy.run_path(str(PATCH_DIR.parent / "fully_async_main.py"), run_name="__main__")
        module.main.assert_called_once_with()

    def test_real_launcher_prints_canonical_hydra_config(self):
        result = subprocess.run(
            [sys.executable, "-m", "rollout_elastic.fully_async_main", "--cfg", "job"],
            cwd=PATCH_DIR.parents[1],
            env=self._subprocess_env(),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, msg=output)
        self.assertIn("async_training", output)

    def test_repeated_install_preserves_canonical_actor_identities(self):
        code = """
from rollout_elastic.patch import install
from verl.experimental.fully_async_policy import fully_async_main

def actors():
    return (
        fully_async_main.FullyAsyncRollouter,
        fully_async_main.FullyAsyncTrainer,
        fully_async_main.FullyAsyncTaskRunner,
    )

before = actors()
install()
after_once = actors()
install()
after_twice = actors()
assert all(left is right for left, right in zip(before, after_once))
assert all(left is right for left, right in zip(after_once, after_twice))
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PATCH_DIR.parents[1],
            env=self._subprocess_env(),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
