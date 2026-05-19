"""
MetaDriveEnvironment: MetaDrive as a proper ecoframe EnvironmentProtocol.

Wraps EnvWorkerPool. Translates between:
  MetaDrive:     dict[agent_id, np.ndarray]  ↔  ActionBundle/SensorBundle
  ecoframe:      dict[global_id, SensorBundle]

Agent ID namespacing:
  MetaDrive uses "agent0", "agent1" within each world.
  Global IDs use "w{world}/agent{i}" to avoid collisions across worlds.

This class owns nothing GPU-related. No torch imports.
Hardware: CPU workers, declared via HardwareSpec.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from ecoframe.protocol import (
    ActionBundle, CapacityError, HardwareSpec,
    SensorBundle, SensorManifest, SensorSpec, Session,
)
from ecoframe.signal import EnvironmentSignal

if TYPE_CHECKING:
    from ecoframe.field import Field


# ── Manifest ──────────────────────────────────────────────────────────────────

MANIFEST = SensorManifest(
    env_id="metadrive_roundabout",
    sensors=(
        SensorSpec("visual",         (64, 64, 3), dtype="uint8",
                   action_affected=True,  world_external=True,  temporal_res=0.1),
        SensorSpec("proprioceptive", (5,),        dtype="float32",
                   action_affected=False, world_external=False, temporal_res=0.05),
        SensorSpec("reward",         (1,),        dtype="float32",
                   action_affected=True,  world_external=True,  temporal_res=0.05),
    ),
)

HARDWARE = HardwareSpec.cpu(n_workers=8, accelerator="panda3d")


# ── MetaDriveEnvironment ──────────────────────────────────────────────────────

class MetaDriveEnvironment:
    """
    M worlds × N agents per world as one ecoframe environment.

    Conforms to EnvironmentProtocol. Backed by EnvWorkerPool.
    """

    env_id        = "metadrive_roundabout"
    manifest      = MANIFEST
    hardware_spec = HARDWARE

    def __init__(
        self,
        n_worlds:        int   = 8,
        n_agents:        int   = 4,
        seed:            int   = 42,
        traffic_density: float = 0.0,
        field:           'Field | None' = None,
        verbose:         bool  = True,
    ):
        self._n_worlds = n_worlds
        self._n_agents = n_agents
        self._field    = field
        self._verbose  = verbose

        self.capacity  = n_worlds * n_agents

        # Sessions: brain_id → Session
        self._sessions: dict[str, Session] = {}
        self._step_count = 0

        # Eval tracking
        self._crashes     = 0
        self._completions = 0
        self._episodes    = 0
        self._ce_ema      = 5.5   # updated externally by training loop

        # Latest world observations (from last step_wait or init)
        self._world_obs: list[dict] = []
        self._last_rewards: list[dict] = []

        # Worker pool (created on start())
        self._pool = None

    # ── EnvironmentProtocol ────────────────────────────────────────────────────

    def start(self) -> None:
        from metadrive_env.worker_pool import EnvWorkerPool
        self._pool = EnvWorkerPool(
            n_worlds=self._n_worlds, n_agents=self._n_agents,
            verbose=self._verbose,
        )
        self._world_obs = self._pool.current_obs()
        self._publish_signal()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def enter(self, brain_id: str, ssm_state: dict | None = None) -> Session:
        if len(self._sessions) >= self.capacity:
            raise CapacityError(f"{self.env_id}: capacity {self.capacity} reached")
        session = Session(
            brain_id   = brain_id,
            env_id     = self.env_id,
            agent_id   = f"{brain_id}_slot{len(self._sessions)}",
            ssm_state  = ssm_state or {},
            entered_at = self._step_count,
        )
        self._sessions[brain_id] = session
        return session

    def exit(self, session: Session) -> dict:
        self._sessions.pop(session.brain_id, None)
        return session.ssm_state

    def reset(self, session: Session) -> dict[str, SensorBundle]:
        """Return current observations. Workers handle their own resets."""
        if self._pool is None:
            self.start()
        return self._world_obs_to_bundles(self._world_obs, {})

    def step_async(self, actions: dict[str, ActionBundle]) -> None:
        """
        Convert ActionBundles → MetaDrive actions, send to all workers.
        Non-blocking: workers begin stepping on CPU immediately.
        """
        actions_per_world = self._actions_to_metadrive(actions)
        self._pool.step_async(actions_per_world)

    def step_wait(self) -> dict[str, SensorBundle]:
        """
        Collect results from all workers.
        Blocks until env step complete — call after GPU backward for overlap.
        """
        world_obs, rewards, terminated, truncated, infos = self._pool.step_wait()
        self._world_obs     = world_obs
        self._last_rewards  = rewards
        self._step_count   += 1

        # Track eval metrics
        for w in range(self._n_worlds):
            for a in range(self._n_agents):
                aid = (world_obs[w]['agent_ids'][a]
                       if a < len(world_obs[w]['agent_ids']) else None)
                if aid and (terminated[w].get(aid, False) or
                            truncated[w].get(aid, False)):
                    if infos[w].get(aid, {}).get('arrive_dest', False):
                        self._completions += 1
                    else:
                        self._crashes += 1
                    self._episodes += 1

        if self._step_count % 100 == 0:
            self._publish_signal()

        return self._world_obs_to_bundles(world_obs, rewards)

    # ── Conversion helpers ─────────────────────────────────────────────────────

    def _world_obs_to_bundles(
        self,
        world_obs: list[dict],
        rewards:   list[dict],
    ) -> dict[str, SensorBundle]:
        """Convert packed world observations to global SensorBundles."""
        bundles: dict[str, SensorBundle] = {}
        for w, w_obs in enumerate(world_obs):
            for a in range(self._n_agents):
                global_id = f"w{w}/agent{a}"
                aid       = (w_obs['agent_ids'][a]
                             if a < len(w_obs['agent_ids']) else None)
                state   = w_obs['states'][a]
                camera  = w_obs['cameras'][a]
                reward  = float(rewards[w].get(aid, 0.)) if (rewards and aid) else 0.0

                # Proprioceptive: [speed, steer, angular_v, vy, vx]
                vx, vy, w_ = float(state[2]), float(state[3]), float(state[4])
                spd = math.sqrt(vx**2 + vy**2)
                st  = float(state[5]) if len(state) > 5 else 0.
                prop = np.array([spd, st, w_, vy, vx], dtype=np.float32)

                bundles[global_id] = SensorBundle(
                    visual         = camera,
                    proprioceptive = prop,
                    extra          = {'reward_scalar': reward},
                    reward         = reward,
                    done           = False,
                    env_id         = self.env_id,
                    agent_id       = global_id,
                    step           = self._step_count,
                )
        return bundles

    def _actions_to_metadrive(
        self,
        actions: dict[str, ActionBundle],
    ) -> list[dict[str, np.ndarray]]:
        """Convert global ActionBundles → per-world MetaDrive action dicts."""
        actions_per_world = [{} for _ in range(self._n_worlds)]
        for global_id, bundle in actions.items():
            try:
                w, rest = global_id.split('/', 1)
                w_idx   = int(w[1:])   # "w3" → 3
                local_id = rest        # "agent0"
            except (ValueError, IndexError):
                continue

            if 0 <= w_idx < self._n_worlds:
                if bundle.continuous is not None:
                    actions_per_world[w_idx][local_id] = bundle.continuous
                else:
                    actions_per_world[w_idx][local_id] = np.array([0.0, 0.5])

        # Fill missing agents with default action (move forward slowly)
        for w_idx, w_obs in enumerate(self._world_obs):
            for a in range(self._n_agents):
                aid = (w_obs['agent_ids'][a]
                       if a < len(w_obs['agent_ids']) else f'agent{a}')
                if aid not in actions_per_world[w_idx]:
                    actions_per_world[w_idx][aid] = np.array([0.0, 0.3])

        return actions_per_world

    # ── Discovery ─────────────────────────────────────────────────────────────

    def _publish_signal(self) -> None:
        if self._field is None:
            return
        self._field.register_agent(self.env_id, pos=(0.0, 0.0))
        sig = EnvironmentSignal(
            position      = (0.0, 0.0),
            timestamp     = self._step_count,
            publisher     = self.env_id,
            curiosity     = self._ce_ema,
            load_fraction = len(self._sessions) / max(1, self.capacity),
            env_type      = "metadrive_roundabout",
            manifest_hash = self.manifest.hash,
            device_type   = self.hardware_spec.device_type,
            memory_gb     = self.hardware_spec.memory_gb,
        )
        self._field.publish(self.env_id, sig)

    # ── Metrics ───────────────────────────────────────────────────────────────

    @property
    def crash_rate(self) -> float:
        return self._crashes / max(1, self._episodes)

    @property
    def completion_rate(self) -> float:
        return self._completions / max(1, self._episodes)
