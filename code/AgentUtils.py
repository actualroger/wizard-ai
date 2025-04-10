
import sys
import os.path
import json

from DQNAgent import *
from Agent import *

# construct an agent from raw parameters
def constructAgentsFromParams(env, params):
    numAgents = params['num_agents']
    agentClass = getattr(sys.modules[__name__], params['agent_type'])
    if params['pooled']:
        baseAgent = agentClass(env, params)
        agents = [PassThroughAgent(baseAgent) for _ in range(numAgents)]
    else:
        baseAgent = None
        agents = [agentClass(env, params) for _ in range(numAgents)]

    return agents, baseAgent

# construct an agent and fill its network with trained parameters
def loadAgent(env, folderPath):
    # load run settings and agent architecture
    settingsFile = os.path.join(os.path.abspath(folderPath), 'settings.json')
    with open(settingsFile, 'r') as f:
        params = json.load(f)

    # construct agent
    agents, baseAgent = constructAgentsFromParams(env, params)

    # load agent weights and deactivate learning & epsilon
    weightFile = os.path.join(os.path.abspath(folderPath), 'agent_dict.pt')
    weights = torch.load(weightFile, weights_only=False)

    # fill agents
    if baseAgent is not None:
        baseAgent.behaviorPolicyNet.load_state_dict(weights)
        baseAgent.setLearningActive(False)
        baseAgent.setEpsilonActive(False)
    else:
        for agent in agents:
            agent.behaviorPolicyNet.load_state_dict(weights)
            agent.setLearningActive(False)
            agent.setEpsilonActive(False)

    return agents, baseAgent
