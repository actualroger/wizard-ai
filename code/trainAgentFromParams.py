
import sys
import json
import numpy as np
import random
import torch
from tqdm import tqdm
import os.path
import traceback

import WizardEnvironment
from PlotUtils import *
from RolloutUtils import rolloutGame

from DQNAgent import *
from Agent import *

# train a number of individual agents
def trainAgents(env, params, saveAgent: bool = False):
    # create the agents
    numAgents = params['num_agents']
    agentClass = getattr(sys.modules[__name__], params['agent_type'])
    if params['pooled']:
        baseAgent = agentClass(env, params)
        agents = [PassThroughAgent(baseAgent) for _ in range(numAgents)]
    else:
        agents = [agentClass(env, params) for _ in range(numAgents)]

    # TODO fill buffer with random agent data
    # if 'buffer_prefill_size' in params: # check pooling?
    #     prefillBuffer(agents, params['buffer_prefill_size'])

    # training variables
    train_returns = []
    train_lengths = []
    total_scores = [0] * numAgents # total score for each agent

    # start training
    total_length = 0
    pbar = tqdm(total=params['total_training_time_step'])
    while total_length < params['total_training_time_step']:

        scores, length = rolloutGame(env, agents)
        total_length += length
        train_lengths.append(length)
        pbar.update(length)

        train_returns.append(scores)
        if total_length > params['total_training_time_step'] * 0.9: # more than 90% through
            for a in range(numAgents):
                total_scores[a] += sum(scores[a])

    if params['pooled']:
        train_losses = [baseAgent.trainingLosses]
    else:
        train_losses = [agents[p].trainingLosses for p in range(numAgents)]

    pbar.close()

    # print best agent to json
    if saveAgent:
        if params['pooled']:
            bestAgent = baseAgent
        else:
            bestAgent = agents[np.argmax(total_scores)]
        try:
            torch.save(bestAgent.behaviorPolicyNet.state_dict(), f"./runs/{params['model_name']}")
        except Exception:
            print('Failed to write agent')
            print(traceback.format_exc)

    # save the results
    return train_returns, train_losses, train_lengths

# trains agents for multiple runs
def trainAgentsMultipleTimes(params):
    # set the random seed
    seed = params['seed']
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # create environment
    my_env = WizardEnvironment.env(numPlayers=params['num_agents'])

    # create experiment
    train_returns = []
    train_losses = []
    train_lengths = []
    for i in range(params['num_runs']):
        tr, tl, tll = trainAgents(my_env, params, saveAgent=(i==0))
        train_returns.append(tr)
        train_losses.append(tl)
        train_lengths.append(tll)

    return train_returns, train_losses, train_lengths

def trainAgentFromParams(settingsFile):
    print('Training from', settingsFile)
    absFile = os.path.abspath(settingsFile)
    absFolder = os.path.dirname(absFile)
    
    # load run parameters from json
    params = {}
    with open(absFile, 'r') as f:
        try:
            params = json.load(f)
        except Exception:
            print(traceback.format_exc())
    if len(params) == 0:
        print('Params not found')
        return
    # print(params)

    # train agents
    try:
        train_returns, train_losses, train_lengths = trainAgentsMultipleTimes(params)
    except Exception:
        print("\nFailure during training")
        print(traceback.format_exc())
        return

    # write result vectors
    try:
        with open(os.path.join(absFolder, 'training_results.json'), 'w') as f:
            json.dump({'train_returns' : train_returns, 'train_losses' : train_losses, 'train_lengths' : train_lengths}, f)
    except Exception:
        print("\nFailure during writing")
        print(traceback.format_exc())

    # generate plots
    try:
        plotTrainingReturns(train_returns, show=False, filename=os.path.join(absFolder, 'training_returns.png'))
        plotTrainingLosses(train_losses, show=False, filename=os.path.join(absFolder, 'training_losses.png'))
    except Exception:
        print("\nFailure during plotting")
        print(traceback.format_exc())

if __name__ == '__main__':
    for s in sys.argv[1:]:
        trainAgentFromParams(s)

# TODO
# to load agent from run:
# load params, check for agent type
# construct agent of type
# turn off learning, epsilon(?)
# load state dict from json
# with open('agent.json', 'r') as f:
#     self.behaviorPolicyNet.load_state_dict(json.load(f))

# TODO plot 500k and 500k_pooled
