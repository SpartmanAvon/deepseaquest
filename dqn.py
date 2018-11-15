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
		flattenLayer = Flatten(name='flattenLayer') (secondHiddenLayer)
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

	def getQVals(self,state):
		return self.model.predict(np.array([state]))

	def getLabels(self,transition, discount):
		QVals = self.getQVals(transition.state).flatten()
		if transition.isTerminal:
			QVals[transition.action] = transition.reward
		else:
			QVals[transition.action] = transition.reward + discount*np.amax(self.getQVals(transition.nextState))
		return QVals


	def bestActionFor(self, state):
		return np.argmax(self.getQVals(state),axis=1)

	def train(self, transitions, discount):
		inputStates = np.array([transition.state for transition in transitions])
		inputLabels = np.array([self.getLabels(transition,discount) for transition in transitions])
		self.model.fit(inputStates,inputLabels,verbose=0)
		return

	def getEvaluationScore(self, evaluationStates):
		totalStates = len(evaluationStates)
		totalQ = 0
		for state in evaluationStates:
			totalQ += np.amax(self.getQVals(state))
		return totalQ/totalStates


