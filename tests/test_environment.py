"""
MetaDriveEnvironment tests using mock worker pool.

Tests validate:
  1. Conforms to EnvironmentProtocol interface
  2. Agent IDs namespaced "w{world}/agent{i}"
  3. SensorBundle shapes match manifest
  4. ActionBundle → MetaDrive action conversion
  5. Crash/completion tracking
  6. enter()/exit() session lifecycle
  7. CapacityError when full
  8. EnvironmentSignal published to Field
  9. HardwareSpec is CPU (no GPU required)
  10. reward in SensorBundle (action-affected prediction target)
"""
import math
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from ecoframe.protocol import ActionBundle, CapacityError, HardwareSpec, Session, SensorBundle
from ecoframe.signal import EnvironmentSignal
from metadrive_env.environment import MetaDriveEnvironment, MANIFEST, HARDWARE
from metadrive_env.worker import pack_obs


# ── Mock worker pool ───────────────────────────────────────────────────────────

def _make_mock_pool(n_worlds=2, n_agents=4):
    pool = MagicMock()
    pool.n_worlds = n_worlds
    pool.n_agents = n_agents

    def _make_world_obs():
        return {
            'agent_ids': [f'agent{i}' for i in range(n_agents)],
            'states':    np.zeros((n_agents, 19), dtype=np.float32),
            'cameras':   np.zeros((n_agents, 64, 64, 3), dtype=np.uint8),
        }

    obs = [_make_world_obs() for _ in range(n_worlds)]
    pool.current_obs.return_value = obs
    pool.step_wait.return_value = (
        obs,
        [{f'agent{i}': 0.1 for i in range(n_agents)} for _ in range(n_worlds)],
        [{f'agent{i}': False for i in range(n_agents)} | {'__all__': False} for _ in range(n_worlds)],
        [{f'agent{i}': False for i in range(n_agents)} | {'__all__': False} for _ in range(n_worlds)],
        [{f'agent{i}': {'arrive_dest': False} for i in range(n_agents)} for _ in range(n_worlds)],
    )
    return pool


@pytest.fixture
def env():
    e = MetaDriveEnvironment(n_worlds=2, n_agents=4, verbose=False)
    mock_pool = _make_mock_pool(n_worlds=2, n_agents=4)
    with patch('metadrive_env.environment.MetaDriveEnvironment.start'):
        e._pool      = mock_pool
        e._world_obs = mock_pool.current_obs()
    return e


# ── Protocol conformance ───────────────────────────────────────────────────────

def test_env_id(env):
    assert env.env_id == "metadrive_roundabout"


def test_manifest_is_ecoframe(env):
    assert env.manifest is MANIFEST
    targets = [s.name for s in env.manifest.prediction_targets]
    assert "visual"  in targets
    assert "reward"  in targets
    assert "proprioceptive" not in targets


def test_hardware_spec_is_cpu():
    assert HARDWARE.device_type == "cpu"
    assert HARDWARE.accelerator == "panda3d"


def test_hardware_spec_no_gpu_required():
    assert HARDWARE.is_compatible_with(HardwareSpec.cuda(device_id=0))
    assert HARDWARE.is_compatible_with(HardwareSpec.cuda(device_id=1))


# ── Session lifecycle ──────────────────────────────────────────────────────────

def test_enter_returns_session(env):
    session = env.enter("brain0")
    assert isinstance(session, Session)
    assert session.brain_id == "brain0"
    assert session.env_id   == "metadrive_roundabout"


def test_exit_frees_slot(env):
    session = env.enter("brain0")
    assert len(env._sessions) == 1
    env.exit(session)
    assert len(env._sessions) == 0


def test_capacity_error_when_full(env):
    env.capacity = 2
    env.enter("brain0")
    env.enter("brain1")
    with pytest.raises(CapacityError):
        env.enter("brain2")


# ── Observations ──────────────────────────────────────────────────────────────

def test_step_wait_returns_global_agent_ids(env):
    bundles = env.step_wait()
    # Should have n_worlds * n_agents = 8 entries
    assert len(bundles) == 8
    for key in bundles:
        assert key.startswith('w') and '/' in key, f"Bad agent ID: {key}"


def test_agent_id_format(env):
    bundles = env.step_wait()
    ids = sorted(bundles.keys())
    assert "w0/agent0" in ids
    assert "w1/agent3" in ids


def test_sensor_bundle_visual_shape(env):
    bundles = env.step_wait()
    for b in bundles.values():
        assert b.visual is not None
        assert b.visual.shape == (64, 64, 3)


def test_sensor_bundle_proprioceptive_shape(env):
    bundles = env.step_wait()
    for b in bundles.values():
        assert b.proprioceptive.shape == (5,)


def test_sensor_bundle_has_reward(env):
    bundles = env.step_wait()
    for b in bundles.values():
        assert isinstance(b.reward, float)


def test_sensor_bundle_env_id(env):
    bundles = env.step_wait()
    for b in bundles.values():
        assert b.env_id == "metadrive_roundabout"


# ── Actions ────────────────────────────────────────────────────────────────────

def test_step_async_sends_to_pool(env):
    actions = {
        "w0/agent0": ActionBundle(continuous=np.array([0.1, 0.5]), agent_id="w0/agent0"),
        "w1/agent2": ActionBundle(continuous=np.array([-0.2, 0.8]), agent_id="w1/agent2"),
    }
    env.step_async(actions)
    env._pool.step_async.assert_called_once()


def test_actions_routed_to_correct_world(env):
    captured = []
    env._pool.step_async.side_effect = lambda x: captured.append(x)
    env.step_async({
        "w0/agent0": ActionBundle(continuous=np.array([0.3, 0.6])),
        "w1/agent1": ActionBundle(continuous=np.array([-0.1, 0.4])),
    })
    assert len(captured) == 1
    world_actions = captured[0]
    assert 'agent0' in world_actions[0]   # world 0
    assert 'agent1' in world_actions[1]   # world 1


def test_missing_agents_get_default_action(env):
    """Agents not in actions dict get a default forward action."""
    captured = []
    env._pool.step_async.side_effect = lambda x: captured.append(x)
    env.step_async({})   # no explicit actions
    world_actions = captured[0]
    # All agents should have a default action
    for w_actions in world_actions:
        for aid in [f'agent{i}' for i in range(4)]:
            assert aid in w_actions


# ── Crash/completion tracking ─────────────────────────────────────────────────

def test_crash_tracked_from_terminated(env):
    env._pool.step_wait.return_value = (
        env._world_obs,
        [{'agent0': -1.0, 'agent1': 0.1, 'agent2': 0.1, 'agent3': 0.1},
         {'agent0': 0.1, 'agent1': 0.1, 'agent2': 0.1, 'agent3': 0.1}],
        [{'agent0': True, '__all__': False, 'agent1': False, 'agent2': False, 'agent3': False},
         {'agent0': False, '__all__': False, 'agent1': False, 'agent2': False, 'agent3': False}],
        [{'agent0': False, '__all__': False, 'agent1': False, 'agent2': False, 'agent3': False}] * 2,
        [{'agent0': {'arrive_dest': False}, 'agent1': {}, 'agent2': {}, 'agent3': {}},
         {'agent0': {}, 'agent1': {}, 'agent2': {}, 'agent3': {}}],
    )
    env.step_wait()
    assert env._crashes == 1
    assert env._episodes == 1


def test_completion_tracked_from_arrive_dest(env):
    env._pool.step_wait.return_value = (
        env._world_obs,
        [{'agent0': 1.0, 'agent1': 0.1, 'agent2': 0.1, 'agent3': 0.1}] * 2,
        [{'agent0': True, '__all__': False, 'agent1': False, 'agent2': False, 'agent3': False}] * 2,
        [{'agent0': False, '__all__': False, 'agent1': False, 'agent2': False, 'agent3': False}] * 2,
        [{'agent0': {'arrive_dest': True}, 'agent1': {}, 'agent2': {}, 'agent3': {}}] * 2,
    )
    env.step_wait()
    assert env._completions == 2   # one per world


# ── Field discovery ───────────────────────────────────────────────────────────

def test_publishes_environment_signal(env):
    from ecoframe.field import Field
    field = Field(backend='local')
    env._field = field
    env._step_count = 99
    env.step_wait()   # triggers _publish_signal at step 100
    signals = field.query(pos=(0., 0.), radius=1.0)
    env_sigs = [s for s in signals if isinstance(s, EnvironmentSignal)]
    assert len(env_sigs) == 1
    assert env_sigs[0].env_type == "metadrive_roundabout"
    assert env_sigs[0].device_type == "cpu"


# ── pack_obs helper ───────────────────────────────────────────────────────────

def test_pack_obs_always_n_agents():
    obs = {f'agent{i}': {
        'state': np.zeros(19),
        'image': np.zeros((64, 64, 3, 1)),
    } for i in range(2)}   # only 2 agents, expect 4
    packed = pack_obs(obs, n_agents=4)
    assert packed['states'].shape  == (4, 19)
    assert packed['cameras'].shape == (4, 64, 64, 3)
    assert len(packed['agent_ids']) == 2   # actual agents, not padded


def test_pack_obs_full_agents():
    obs = {f'agent{i}': {
        'state': np.ones(19) * i,
        'image': np.zeros((64, 64, 3)),
    } for i in range(4)}
    packed = pack_obs(obs, n_agents=4)
    assert packed['states'].shape == (4, 19)
    assert packed['states'][2, 0] == pytest.approx(2.0)
