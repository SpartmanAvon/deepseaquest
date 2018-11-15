from keras.models import Model
from keras.layers import *


class DQN:
    def __init__(self, num_actions):
        inputLayer = Input(name='inputLayer',
                           shape=(80, 80, 4))
        firstHiddenLayer = Conv2D(name='firstHiddenLayer',
                                  filters=16,
                                  kernel_size=(8, 8),
                                  strides=(4, 4),
                                  activation='relu')(inputLayer)
        secondHiddenLayer = Conv2D(name='secondHiddenLayer',
                                   filters=32,
                                   kernel_size=(4, 4),
                                   strides=(2, 2),
                                   activation='relu')(firstHiddenLayer)
        flattenLayer = Flatten(name='flattenLayer')(secondHiddenLayer)
        thirdHiddenLayer = Dense(name='thirdHiddenLayer',
                                 units=256,
                                 activation='relu')(flattenLayer)
        outputLayer = Dense(name='outputLayer',
                            units=num_actions,
                            activation='linear')(thirdHiddenLayer)

        model = Model(inputLayer, outputLayer)
        model.compile(optimizer='rmsprop', loss='mse')
        # print(model.summary())
        self.model = model

    def predictFor(self, state):
        return self.model.predict(np.array([state]))

    def getTrueLabels(self, transition, discount):
        trueLabels = self.predictFor(transition.state).flatten()
        if transition.isTerminal:
            trueLabels[transition.action] = transition.reward
        else:
            trueLabels[transition.action] = transition.reward + discount * np.max(
                self.predictFor(transition.nextState), axis=1)
        return trueLabels

    def bestActionFor(self, state):
        return np.argmax(self.predictFor(state), axis=1)

    def train(self, transitions, discount):
        states = np.array([transition.state for transition in transitions])
        trueLabels = np.array([self.getTrueLabels(transition, discount) for transition in transitions])
        self.model.fit(states, trueLabels, verbose=0)
        return

    def getEvaluationScore(self, evaluationStates):
        # todo improve
        totalStates = len(evaluationStates)
        totalQ = 0
        for state in evaluationStates:
            totalQ += np.max(self.predictFor(state), 1)
        return totalQ / totalStates
