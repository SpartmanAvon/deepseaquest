import time

import gym
import numpy as np
import scipy.ndimage.interpolation as interpolation


class StateGenerator:
    @classmethod
    def decrease_dimensionality(cls, image):
        # downsample size = 80 x 80
        downsampled = interpolation.zoom(image[20:-30, ], [0.5, 0.5, 1])
        r, g, b = downsampled[:, :, 0], downsampled[:, :, 1], downsampled[:, :, 2]
        # convert to grayscale
        return 0.299 * r + 0.587 * g + 0.114 * b

    def __init__(self, capacity):
        self.capacity = capacity
        self.frames = []
        self.frameShape = (80, 80)

    def getState(self, frame):
        if len(self.frames) == self.capacity:
            self.frames.pop(0)
        self.frames.append(frame)
        return self.generateState()

    def generateState(self):
        paddingSize = self.capacity - len(self.frames)
        zeroFrames = [np.zeros(self.frameShape) for _ in range(paddingSize)]
        frames = [StateGenerator.decrease_dimensionality(f) for f in self.frames]
        frames = zeroFrames + frames
        return np.stack(frames, axis=0)


class Transition:
    def __init__(self, state, action, reward, nextState):
        self.state = state
        self.action = action
        self.reward = reward
        self.nextState = nextState


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.transitions = []

    def addTransition(self, transition):
        if len(self.transitions) == self.capacity:
            self.transitions.pop(0)
        self.transitions.append(transition)

    def sample(self, batchSize):
        # todo remove zero frames here
        indices = list(range(len(self.transitions)))
        np.random.shuffle(indices)
        sampleIndices = indices[:batchSize]
        sample = [self.transitions[i] for i in sampleIndices]
        return sample


class DQN:
    def __init__(self):
        pass

    def bestActionFor(self, state):
        return 0

    def train(self, transitions):
        pass


class DeepReinforcingAgent:
    def __init__(self, env, stateGeneratorCapacity, replayMemoryCapacity, explorationRate):
        self.env = env
        self.stateGenerator = StateGenerator(stateGeneratorCapacity)
        self.replayMemory = ReplayMemory(replayMemoryCapacity)
        assert (0 <= explorationRate <= 1)
        self.explorationRate = explorationRate
        self.Q = DQN()

    def chooseActionGreedily(self, state):
        if np.random.rand() <= self.explorationRate:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q.bestActionFor(state))

    def learn(self, numEpisodes, episodeMaxLength):
        for ithEpisode in range(numEpisodes):
            frame = self.env.reset()
            state = self.stateGenerator.getState(frame)
            for ithTimestep in range(episodeMaxLength):
                self.env.render()
                time.sleep(0.01)
                action = self.chooseActionGreedily(state)
                nextFrame, reward, done, info = self.env.step(action)
                nextState = self.stateGenerator.getState(nextFrame)
                self.replayMemory.addTransition(Transition(state, action, reward, nextState))
                transitions = self.replayMemory.sample(32)
                self.Q.train(transitions)
                if done:
                    print("Episode finished after %d timesteps" % (ithTimestep + 1,))
                    break


STATE_GENERATOR_CAPACITY = 4
REPLAY_MEMORY_CAPACITY = 500
EXPLORATION_RATE = 0

NUMBER_OF_EPISODES = 100
EPISODE_MAX_LENGTH = 1000

deepReinforcingAgent = DeepReinforcingAgent(gym.make('Seaquest-v0'),
                                            STATE_GENERATOR_CAPACITY,
                                            REPLAY_MEMORY_CAPACITY,
                                            EXPLORATION_RATE)

deepReinforcingAgent.learn(NUMBER_OF_EPISODES, EPISODE_MAX_LENGTH)
