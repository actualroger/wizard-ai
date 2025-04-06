
import random
import numpy

import WizardEnvironment
from HumanAgent import HumanAgent
from RandomAgent import RandomAgent
from RolloutUtils import rolloutGame
from PlotUtils import plotAgentScores

numPlayers = 5
numHumans = 0

numGames = 10

seed = 0
verbosity = 0
dealer = 0

random.seed(seed)
numpy.random.seed(seed)

env = WizardEnvironment.env(renderMode="ansi", numPlayers=numPlayers, verbosity=verbosity)
agents = [HumanAgent(env) for _ in range(numHumans)] + [RandomAgent(env, verbosity=verbosity) for _ in range(numPlayers - numHumans)]

scores = []
for _ in range(numGames):
	thisRoundScores, _ = rolloutGame(env, agents, dealer=dealer, verbosity=verbosity)
	scores.append(thisRoundScores)

scores = numpy.array(scores) # [game][player][hand]
scores = numpy.transpose(scores, (1,0,2)) # [player][game][hand]

plotAgentScores(scores)
