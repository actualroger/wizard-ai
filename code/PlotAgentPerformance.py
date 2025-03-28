
import WizardEnvironment
from HumanAgent import HumanAgent
from RandomAgent import RandomAgent
from Rollout import rolloutGame
from Plot import plot_curves

numPlayers = 6
numHumans = 0

numGames = 100

seed = 0
verbosity = 0
dealer = 0

env = WizardEnvironment.env(renderMode="ansi", numPlayers=numPlayers, verbosity=verbosity)
agents = [HumanAgent(env) for _ in range(numHumans)] + [RandomAgent(env, verbosity=verbosity) for _ in range(numPlayers - numHumans)]

scores = []
for _ in range(numGames):
	scores.append(rolloutGame(env, agents, dealer=dealer, verbosity=verbosity))

plot_curves(scores, ['Agent %d'%i for i in range(len(agents))], ['r','g','b','k'], xlabel='Round', ylabel='Score')
