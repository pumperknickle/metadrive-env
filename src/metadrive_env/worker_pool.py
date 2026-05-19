"""
EnvWorkerPool: M parallel MetaDrive worlds in separate subprocesses.

Pure subprocess management — no neural network code, no GPU code.
Actions in: dict[str, np.ndarray].
Observations out: list of packed obs dicts.

The async interface enables CPU-GPU overlap in the training loop:
    pool.step_async(actions)       # workers start stepping (non-blocking)
    # ... GPU forward+backward ... # overlaps with env step
    world_obs = pool.step_wait()   # collect when GPU work done
"""
from __future__ import annotations

import multiprocessing as mp
from typing import Any

import numpy as np

from metadrive_env.worker import worker_fn


class EnvWorkerPool:
    """
    M worlds running in parallel subprocesses.

    Each worker owns one MultiAgentRoundaboutEnv(n_agents).
    Main process communicates via mp.Queue pairs (action_q, obs_q).
    """

    def __init__(
        self,
        n_worlds:        int   = 8,
        n_agents:        int   = 4,
        seed:            int   = 42,
        traffic_density: float = 0.0,
        verbose:         bool  = True,
    ):
        self.n_worlds  = n_worlds
        self.n_agents  = n_agents
        self._verbose  = verbose

        ctx = mp.get_context('spawn')   # required for Panda3D
        self._action_qs: list[mp.Queue] = []
        self._obs_qs:    list[mp.Queue] = []
        self._procs:     list[mp.Process] = []

        if verbose:
            print(f'EnvWorkerPool: launching {n_worlds} worlds × {n_agents} agents...', flush=True)

        for i in range(n_worlds):
            aq = ctx.Queue()
            oq = ctx.Queue()
            p  = ctx.Process(
                target=worker_fn,
                args=(i, n_agents, seed + i * 1000, traffic_density, aq, oq),
                daemon=True,
            )
            p.start()
            self._action_qs.append(aq)
            self._obs_qs.append(oq)
            self._procs.append(p)

        # Collect initial observations from all workers
        self._current_obs: list[dict] = []
        for i in range(n_worlds):
            msg, obs = self._obs_qs[i].get(timeout=120)
            assert msg == 'ready', f"Worker {i} sent unexpected message: {msg}"
            self._current_obs.append(obs)

        if verbose:
            print(f'EnvWorkerPool: {n_worlds} worlds ready.', flush=True)

    # ── Async step interface ───────────────────────────────────────────────────

    def step_async(self, actions_per_world: list[dict[str, np.ndarray]]) -> None:
        """
        Send actions to all workers simultaneously (non-blocking).

        actions_per_world: one dict per world, keyed by local agent_id.
        Returns immediately — workers begin stepping on CPU.
        """
        for w, actions in enumerate(actions_per_world):
            self._action_qs[w].put(('step', actions))

    def step_wait(self) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
        """
        Collect results from all workers (blocks until all done).

        Returns: (world_obs, rewards, terminated, truncated, info)
        Each is a list of length n_worlds.
        """
        world_obs  = []
        rewards    = []
        terminated = []
        truncated  = []
        infos      = []
        for i in range(self.n_worlds):
            msg, obs, rew, term, trunc, info = self._obs_qs[i].get(timeout=15)
            assert msg == 'obs'
            world_obs.append(obs)
            rewards.append(rew)
            terminated.append(term)
            truncated.append(trunc)
            infos.append(info)
        self._current_obs = world_obs
        return world_obs, rewards, terminated, truncated, infos

    def current_obs(self) -> list[dict]:
        """Return buffered observations from last step_wait (or init)."""
        return list(self._current_obs)

    def close(self) -> None:
        for q in self._action_qs:
            try: q.put('close')
            except Exception: pass
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
