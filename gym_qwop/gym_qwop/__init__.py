from gymnasium.envs.registration import register

register(
    id='frame-qwop-v0',
    entry_point='gym_qwop.envs:FrameQWOPEnv',
)
