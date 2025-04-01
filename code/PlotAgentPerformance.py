
import random
import numpy.random

import WizardEnvironment
from HumanAgent import HumanAgent
from RandomAgent import RandomAgent
from RolloutUtils import rolloutGame
from PlotUtils import plot_curves

numPlayers = 6
numHumans = 0

numGames = 100

seed = 0
verbosity = 0
dealer = 0

random.seed(seed)
numpy.random.seed(seed)

env = WizardEnvironment.env(renderMode="ansi", numPlayers=numPlayers, verbosity=verbosity)
agents = [HumanAgent(env) for _ in range(numHumans)] + [RandomAgent(env, verbosity=verbosity) for _ in range(numPlayers - numHumans)]

scores = [[] for _ in range(numPlayers)]
for _ in range(numGames):
	thisRoundScores = rolloutGame(env, agents, dealer=dealer, verbosity=verbosity)
	for player in range(numPlayers):
		scores[player].append(thisRoundScores[player])

roundNumbers = range(1, 60 // numPlayers + 1)
expectedScores = [20 + 10 * (round / (0.0 + numPlayers)) for round in roundNumbers]

plotColors = ['r','g','b','k','y','c']
plot_curves(scores, ['Agent %d'%i for i in range(len(agents))], plotColors[:numPlayers], x_values=roundNumbers, xlabel='Round', ylabel='Score', upper_bound=expectedScores, upper_bound_label='Nominal')
