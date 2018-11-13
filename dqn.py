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
        thirdHiddenLayer = Dense(name='thirdHiddenLayer',
                                 units=256,
                                 activation='relu')(secondHiddenLayer)
        outputLayer = Dense(name='outputLayer',
                            units=num_actions,
                            activation='linear')(thirdHiddenLayer)

        model = Model(inputLayer, outputLayer)
        model.compile(optimizer='rmsprop', loss='mse')
        print(model.summary())
        self.model = model

    def bestActionFor(self, state):
        return 0

    def train(self, transitions):
        pass
