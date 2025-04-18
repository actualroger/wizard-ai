
import random
import numpy
import sys
import json

import WizardEnvironment
from RolloutUtils import rolloutGame
from PlotUtils import plotAgentScores
from AgentUtils import loadAgent
from HumanAgent import HumanAgent

agentFolder = sys.argv[1]
saveFilename = sys.argv[2]

numPlayers = 4
numHumans = 1
numGames = 5
minRound = 1
maxRound = 6

seed = 1
envVerbosity = 2
gameVerbosity = 1
dealer = 0

random.seed(seed)
numpy.random.seed(seed)

env = WizardEnvironment.env(renderMode="ansi", numPlayers=numPlayers, verbosity=envVerbosity)
agents, baseAgent = loadAgent(env, agentFolder)
for n in range(numHumans):
	agents[n] = HumanAgent(env)

scores = []
for _ in range(numGames):
	thisGameScores, _ = rolloutGame(env, agents, dealer=dealer, verbosity=gameVerbosity, minRound=minRound, maxRound=maxRound)
	scores.append(thisGameScores)

with open(saveFilename, 'w') as f:
	json.dump({'scores' : scores}, f)

scores = numpy.array(scores) # [game][player][hand]
scores = numpy.transpose(scores, (1,0,2)) # [player][game][hand]

plotAgentScores(scores)
