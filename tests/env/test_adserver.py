import unittest
import pytest

import gym
from gym import envs

import gym_adserver
from gym_adserver.envs.ad import Ad
from gym_adserver.envs.adserver import AdServerEnv


def test_environment_reset():
    # Arrange 
    env = envs.make('AdServer-v0', num_ads=2, time_series_frequency=10)

    # Act
    (observation, info) = env.reset(options={"scenario_name": 'Test'})

    # Assert
    assert info["clicks"] == 0
    assert info["impressions"] == 0
    assert info["ads"] == [Ad(0), Ad(1)]


def test_environment_step_no_reward():
    # Arrange
    env = envs.make('AdServer-v0', num_ads=2, time_series_frequency=10,
                    ads_info=[{"cpi": 0.0, "rpc": 0.0}, {"cpi": 0.2, "rpc": 0.5}], click_simulation=lambda x: 0)
    env.reset(options={"scenario_name": 'Test'})

    # Act
    (observation, reward, terminated, truncated, info) = env.step(0)

    # Assert
    assert info["clicks"] == 0
    assert info["impressions"] == 1
    # assert info == {}
    assert reward == 0
    assert not terminated
    assert not truncated
    assert info["ads"] == [Ad(0, impressions=1, cpi=0.0, rpc=0.0), Ad(1, cpi=0.2, rpc=0.5)]


def test_environment_step_with_reward():
    # Arrange
    env = envs.make('AdServer-v0', num_ads=2, time_series_frequency=10,
                    ads_info=[{"cpi": 0.0, "rpc": 0.0}, {"cpi": 0.2, "rpc": 0.5}], click_simulation=lambda x: 1)
    env.reset(options={"scenario_name": 'Test'})

    # Act
    (observation, reward, terminated, truncated, info) = env.step(1)

    # Assert
    assert info["clicks"] == 1
    assert info["impressions"] == 1
    # assert info == {}
    assert reward == 0.3
    assert not terminated
    assert not truncated
    assert info["ads"] == [Ad(0, cpi=0.0, rpc=0.0), Ad(1, impressions=1, clicks=1, cpi=0.2, rpc=0.5)]
