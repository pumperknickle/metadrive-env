"""
MetaDrive subprocess worker.

One process per world. Owns a MultiAgentRoundaboutEnv instance.
Receives actions (dict of np.ndarray), returns observations (packed dict).

No neural network code. No GPU code. CPU-only.
CUDA_VISIBLE_DEVICES='' is set at import time — inherited by all subprocesses.
"""
from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np

_CAM_W = 64
_CAM_H = 64


def worker_fn(
    worker_id:       int,
    n_agents:        int,
    seed:            int,
    traffic_density: float,
    action_q:        mp.Queue,
    obs_q:           mp.Queue,
) -> None:
    """
    Long-running subprocess: owns one MultiAgentRoundaboutEnv.

    Protocol:
      obs_q ← ('ready', packed_obs)              on startup
      action_q → ('step', {agent_id: np.array})  per step
      obs_q ← ('obs', packed_obs, rewards, terminated, truncated, info)
      action_q → 'close'                          to shutdown
    """
    os.environ.setdefault('DISPLAY', ':1')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''   # workers never touch GPU

    from metadrive.envs.marl_envs.marl_inout_roundabout import MultiAgentRoundaboutEnv
    from metadrive.component.sensors.rgb_camera import RGBCamera

    env = MultiAgentRoundaboutEnv({
        'num_agents':        n_agents,
        'start_seed':        seed,
        'image_observation': True,
        'sensors':           {'rgb_camera': (RGBCamera, _CAM_W, _CAM_H)},
        'vehicle_config':    {'image_source': 'rgb_camera'},
        'traffic_density':   traffic_density,
        'horizon':           500,
        'crash_done':        True,
        'out_of_road_done':  True,
        'allow_respawn':     True,
        'delay_done':        10,
        'truncate_as_terminate': True,
        'use_render':        False,
        'show_logo':         False,
        'show_fps':          False,
    })

    obs, _ = env.reset()
    obs_q.put(('ready', pack_obs(obs, n_agents)))

    while True:
        msg = action_q.get()
        if msg == 'close':
            env.close()
            return

        assert msg[0] == 'step'
        actions = msg[1]   # {agent_id: np.array([steer, speed])}

        obs_next, rewards, terminated, truncated, info = env.step(actions)

        # Auto-reset on full episode end
        if terminated.get('__all__', False) or truncated.get('__all__', False):
            obs_next, _ = env.reset()
            terminated  = {aid: False for aid in obs_next}
            terminated['__all__'] = False
            truncated   = {aid: False for aid in obs_next}
            truncated['__all__'] = False

        obs_q.put(('obs', pack_obs(obs_next, n_agents),
                   rewards, terminated, truncated, info))


def pack_obs(obs: dict, n_agents: int) -> dict:
    """
    Convert per-agent MetaDrive obs to fixed-size numpy arrays.

    Always returns exactly n_agents rows — pads with zeros for missing/crashed
    agents. This keeps batch dimensions constant regardless of episode state.

    Returns:
        agent_ids: list of active agent IDs (may be shorter than n_agents)
        states:    (n_agents, 19) float32
        cameras:   (n_agents, H, W, 3) uint8
    """
    agent_ids = sorted(obs.keys())[:n_agents]
    states  = []
    cameras = []

    for aid in agent_ids:
        o     = obs[aid]
        state = np.array(o.get('state', np.zeros(19)), dtype=np.float32)
        img   = o.get('image', o.get('rgb_camera', None))
        if img is None:
            img = np.zeros((_CAM_H, _CAM_W, 3), np.uint8)
        else:
            if img.ndim == 4:
                img = img[..., -1]
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        states.append(state)
        cameras.append(img)

    # Pad to exactly n_agents rows
    while len(states) < n_agents:
        states.append(np.zeros(19, dtype=np.float32))
        cameras.append(np.zeros((_CAM_H, _CAM_W, 3), np.uint8))

    return {
        'agent_ids': agent_ids,
        'states':    np.stack(states),    # (n_agents, 19)  — always
        'cameras':   np.stack(cameras),   # (n_agents, H, W, 3) — always
    }
