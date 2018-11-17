import gym
import numpy as np
import scipy.ndimage.interpolation as interpolation
from scipy.misc import imsave

from dqn import DQN
import atari_wrappers

import time

params = {
        'env_name': "SeaquestNoFrameskip-v4",
    }

def make_env(params):
    env = atari_wrappers.make_atari(params['env_name'])
    env = atari_wrappers.wrap_deepmind(env, frame_skip=4, frame_stack=True)
    return env

class StateGenerator:
    @classmethod
    def decrease_dimensionality(cls, image):
        assert (image.shape == (210, 160, 3))
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
        # add frame
        if len(self.frames) == self.capacity:
            self.frames.pop(0)
        self.frames.append(StateGenerator.decrease_dimensionality(frame))
        # generate state
        paddingSize = self.capacity - len(self.frames)
        zeroFrames = [np.zeros(self.frameShape) for _ in range(paddingSize)]
        frames = zeroFrames + self.frames
        return np.stack(frames, axis=2)

    def resetFrames(self):
        self.frames = []


class Transition:
    def __init__(self, state, action, reward, nextState, isTerminal):
        self.state = state
        self.action = action
        self.reward = reward
        self.nextState = nextState
        self.isTerminal = isTerminal
    def extractStates(self):
        return Transition(self.state.__array__(), self.action, self.reward, self.nextState.__array__(), self.isTerminal)

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.transitions = []

    def addTransition(self, transition):
        # todo donot add zero layer states here
        if len(self.transitions) == self.capacity:
            self.transitions.pop(0)
        self.transitions.append(transition)

    def sample(self, batchSize):
        sampleIndices = np.random.randint(low=0, high=len(self.transitions), size=batchSize)
        sample = [self.transitions[i].extractStates() for i in sampleIndices]
        return sample


class DeepReinforcingAgent:
    def __init__(self, env, stateGeneratorCapacity, replayMemoryCapacity, explorationRate, discount,
                 numEvaluatingStates):
        assert (0 <= explorationRate <= 1)
        assert (0 <= discount <= 1)
        self.env = env
        self.stateGenerator = StateGenerator(stateGeneratorCapacity)
        self.replayMemory = ReplayMemory(replayMemoryCapacity)
        self.explorationRate = explorationRate
        self.discount = discount
        self.Q = DQN(env.action_space.n)
        self.evaluatingStates = self.sampleEvaluatingStates(numEvaluatingStates)

    def sampleEvaluatingStates(self, numEvaluatingStates):
        NUM_SAMPLING_TIMESTEPS = 1000
        sampledStates = []

        while len(sampledStates) < NUM_SAMPLING_TIMESTEPS:
            self.env.reset()
            # The following loop is to set the agent in the middle of arena for better evaluation states
            for _ in range(60):
                frame, reward, isTerminal, info = self.env.step(5)

            while len(sampledStates) < NUM_SAMPLING_TIMESTEPS:
                # self.sampleenv.render()
                action = self.env.action_space.sample()
                frame, reward, isTerminal, info = self.env.step(action)
                sampledStates.append(frame)
                if isTerminal:
                    break

        sampleIndices = np.random.randint(low=0, high=len(sampledStates), size=numEvaluatingStates)
        evaluatingStates = [sampledStates[i].__array__() for i in sampleIndices]

        for i, state in enumerate(evaluatingStates):
            imsave('evaluating_state_%d.png' % i, state[:, :, 0])
        print('Evaluation states are sampled')

        return evaluatingStates

    def chooseActionGreedily(self, state, ithTimestep, episodeMaxLength):
        self.explorationRate = max(1 - (0.9 * ithTimestep / episodeMaxLength), 0.1)
        if np.random.rand() <= self.explorationRate:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q.bestActionFor(state))

    def learn(self, numEpisodes, episodeMaxLength):
        for ithEpisode in range(numEpisodes):
            print('Episode: %d' % ithEpisode)
            self.env.reset()
            state, reward, isTerminal, info = self.env.step(0)
            start = time.time()
            for ithTimestep in range(episodeMaxLength):
                # self.env.render()
                # time.sleep(0.01)
                action = self.chooseActionGreedily(state, ithTimestep, episodeMaxLength)
                nextState, reward, isTerminal, info = self.env.step(action)
                # print(np.shape(nextState.__array__()))
                self.replayMemory.addTransition(Transition(state, action, np.sign(reward), nextState, isTerminal))
                transitions = self.replayMemory.sample(32)
                self.Q.train(transitions, self.discount)
                state = nextState
                if isTerminal:
                    done = time.time()
                    elapsed = done - start
                    print("Episode finished after %d frames, in %d seconds, with an average rate of %d frames per second" % (ithTimestep + 1,elapsed,(ithTimestep+1)/elapsed))
                    break
            print('Evaluation metric =', self.Q.getEvaluationScore(self.evaluatingStates))
            self.Q.saveModel()


STATE_GENERATOR_CAPACITY = 4
REPLAY_MEMORY_CAPACITY = 5000
EXPLORATION_RATE = 1
DISCOUNT = 0.9

NUM_OF_EPISODES = 100
EPISODE_MAX_LENGTH = 10000
NUM_EVALUATING_STATES = 30

np.random.seed(0)

deepReinforcingAgent = DeepReinforcingAgent(make_env(params),
                                            STATE_GENERATOR_CAPACITY,
                                            REPLAY_MEMORY_CAPACITY,
                                            EXPLORATION_RATE,
                                            DISCOUNT,
                                            NUM_EVALUATING_STATES)

deepReinforcingAgent.learn(NUM_OF_EPISODES, EPISODE_MAX_LENGTH)
