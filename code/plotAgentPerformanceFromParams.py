
import random
import numpy
import sys

import WizardEnvironment
from RolloutUtils import rolloutGame
from PlotUtils import plotAgentScores
from AgentUtils import loadAgent

agentFolder = sys.argv[1]
numPlayers = 4

numGames = 50

seed = 0
verbosity = 0
dealer = 0

random.seed(seed)
numpy.random.seed(seed)

env = WizardEnvironment.env(renderMode="ansi", numPlayers=numPlayers, verbosity=verbosity)
agents, baseAgent = loadAgent(env, agentFolder)

scores = []
for _ in range(numGames):
	thisRoundScores, _ = rolloutGame(env, agents, dealer=dealer, verbosity=verbosity)
	scores.append(thisRoundScores)

scores = numpy.array(scores) # [game][player][hand]
scores = numpy.transpose(scores, (1,0,2)) # [player][game][hand]

plotAgentScores(scores)
