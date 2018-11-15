import gym
import time
from datetime import datetime

print(datetime.now())
env = gym.make("Seaquest-v0")
observation = env.reset()
for _ in range(60):
    # env.render()
    action = 5  # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)

# time.sleep(5)
for _ in range(1000):
    # env.render()
    action = env.action_space.sample()  # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
print(datetime.now())
