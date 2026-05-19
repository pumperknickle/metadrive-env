"""
metadrive-env: MetaDrive driving simulator as an ecoframe EnvironmentProtocol.

Usage:
    from metadrive_env import MetaDriveEnvironment, MANIFEST, HARDWARE
    from ecoframe import TrainingEngine
    from ecoframe.protocol import HardwareSpec

    env    = MetaDriveEnvironment(n_worlds=8, n_agents=4)
    engine = TrainingEngine(brain, env)
    for step, metrics in engine.run(n_steps=2_000_000):
        log(metrics)

With ecoframe-runtime hardware isolation:
    from ecoframe_runtime import EnvironmentLauncher
    launcher = EnvironmentLauncher()
    env = launcher.launch(
        lambda: MetaDriveEnvironment(n_worlds=8, n_agents=4),
        env_id="roundabout_0",
        hardware=HARDWARE,
    )
"""
from metadrive_env.environment import MetaDriveEnvironment, MANIFEST, HARDWARE
from metadrive_env.worker_pool import EnvWorkerPool

__version__ = "0.1.0"
__all__ = ["MetaDriveEnvironment", "EnvWorkerPool", "MANIFEST", "HARDWARE"]
