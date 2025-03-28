
import random
import numpy as np

import WizardEnvironment
from HumanAgent import HumanAgent
from RandomAgent import RandomAgent

numPlayers = 4
numHumans = 1
startRound = 1
endRound = 15

seed = 0
verbosity = 2
dealer = 0

scores = [0] * numPlayers

random.seed(seed=seed)
np.random.seed(seed)

env = WizardEnvironment.env(renderMode="ansi", numPlayers=numPlayers, verbosity=verbosity)
agents = [HumanAgent(env) for _ in range(numHumans)] + [RandomAgent(env, verbosity=verbosity) for _ in range(numPlayers - numHumans)]

for round in range(startRound, endRound + 1):
    env.reset(round, dealer=dealer)
    print("\nStarted round %d with dealer %d\n" % (round, dealer))
    dealer = (dealer + 1) % numPlayers
    
    while not env.terminated:
        agent = env.currentAgent
        print("\nAgent %d's turn" % agent)
        observation = env.observe(agent)
        # print("Agent %d observes\n\t" % agent, '\t\n'.join([' : '.join([str(k),str(v)]) for k,v in observation.items()]), sep='')
        
        action = agents[agent].getAction(observation)
        _, reward, _, _, _ = env.step(agent, action)
        if reward != 0:
            print("Agent %d received reward %d"%(agent, reward))
            scores[agent] += reward
    
    print("End of round scores:", scores, "\n")

print("\nEnd of game")
env.close() # end of game
