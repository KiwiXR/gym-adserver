from gym.envs.registration import register

register(
    id='AdServer-v0',
    entry_point='gym_adserver.envs:AdServerEnv',
    max_episode_steps=500
)
