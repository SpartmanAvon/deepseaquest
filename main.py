import gym
import time
import numpy as np


class Transition:
    """
    state-action-reward-nextState tuples
    """

    def __init__(self, _state, _action, _reward, next_state):
        self.state = _state
        self.action = _action
        self.reward = _reward
        self.next_state = next_state


class ReplayMemory:
    """
    Fixed size queue
    """

    def __init__(self, size):
        self.size = size
        self.memory = []

    def add(self, transition):
        pass

    def sample(self, mini_batch_size):
        pass


class RecentFrames:
    """
    Container last N frames
    """

    def __init__(self, size):
        self.size = size
        self.frames = []

    def add(self, new_frame):
        pass

    def get_state_from_frames(self):
        pass


class DQN:
    """
    Deep Q Network
    """

    def __init__(self):
        pass

    def train(self, transitions):
        pass


class DeepQLearning:
    def __init__(self, num_episodes, episode_length, recent_frames_size, replay_memory_size, mini_batch_size):
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        self.recent_frames = RecentFrames(recent_frames_size)
        self.dqn = DQN()
        self.replay_memory = ReplayMemory(replay_memory_size)

    def learn(self):
        pass


env = gym.make('Seaquest-v0')
print(env.action_space)
print(env.observation_space)
for i_episode in range(20):
    state = env.reset()
    print(state.shape)
    for t in range(1000):
        env.render()
        action = np.random.choice([7, 9])
        time.sleep(0.05)
        state, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
