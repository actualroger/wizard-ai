
import WizardEnvironment
from RandomAgent import RandomAgent
from RolloutUtils import *

def playRandomGame(numPlayers, verbosity: int = 0):
    env = WizardEnvironment.env(numPlayers, verbosity=verbosity)
    agents = [RandomAgent(env, verbosity=verbosity) for _ in range(numPlayers)]
    scores, length = rolloutRound(env, agents, 1, verbosity)
    return scores, length

if __name__ == '__main__':
    s,l = playRandomGame(4, 2)
    print('scores', s, '\nlength', l)

'''
import WizardEnvironment
env = WizardEnvironment.env(4)
env.verbosity = 2
from DQNAgent import DQNAgent
import json
with open('../runs/gpu_test/settings.json','r') as f:
    params = json.load(f)

agents2 = [DQNAgent(env, params) for _ in range(4)]
from RolloutUtils import *
s,l = rolloutRound(env, agents2, 1)
'''
