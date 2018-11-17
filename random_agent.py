#!/usr/bin/env python3

import time
import atari_wrappers
import numpy as np

params = {
        'env_name': "SeaquestNoFrameskip-v4",
    }

def make_env(params):
    env = atari_wrappers.make_atari(params['env_name'])
    env = atari_wrappers.wrap_deepmind(env, frame_skip=4, frame_stack=True)
    return env

env = make_env(params)

env.reset()
start = time.time()
frames = 10000
for i in range(frames):
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    if done:
        env.reset()

done = time.time()
elapsed = done - start
print(elapsed)
print(frames/elapsed)