import gym
import time
import scipy.ndimage.interpolation as interpolation


def decrease_dimensionality(image):
    # downsample size = 80 x 80
    downsampled = interpolation.zoom(image[20:-30, ], [0.5, 0.5, 1])
    r, g, b = downsampled[:, :, 0], downsampled[:, :, 1], downsampled[:, :, 2]
    # convert to grayscale
    return 0.299 * r + 0.587 * g + 0.114 * b


env = gym.make("Seaquest-v0")
observation = env.reset()
for _ in range(60):
    env.render()
    action = 5  # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    decrease_dimensionality(observation)

# time.sleep(5)
for _ in range(1000):
    env.render()
    action = env.action_space.sample()  # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    decrease_dimensionality(observation)
