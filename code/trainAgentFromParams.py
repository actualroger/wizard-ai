
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
from AgentUtils import constructAgentsFromParams

# train a number of individual agents
def trainAgents(env, params):
    # create the agents
    numAgents = params['num_agents']
    agents, baseAgent = constructAgentsFromParams(env, params)

    # TODO fill buffer with random agent data
    # if 'buffer_prefill_size' in params: # check pooling?
    #     prefillBuffer(agents, params['buffer_prefill_size'])

    # get game start and end rounds
    startRound = params['start_round'] if 'start_round' in params else 1
    endRound = params['end_round'] if 'end_round' in params else 60 // numAgents

    # training variables
    train_returns = []
    train_lengths = []
    total_scores = [0] * numAgents # total score for each agent

    # start training
    total_length = 0
    pbar = tqdm(total=params['total_training_time_step'])
    while total_length < params['total_training_time_step']:

        scores, length = rolloutGame(env, agents, minRound=startRound, maxRound=endRound)
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

    # save best agent to file
    if params['pooled']:
        bestAgent = baseAgent
        bestScore = max(total_scores)
    else:
        agentI = np.argmax(total_scores)
        bestAgent = agents[agentI]
        bestScore = total_scores[agentI]

    # save the results
    return train_returns, train_losses, train_lengths, bestScore, bestAgent

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
    best_score = -float('inf')
    best_agent = None
    for i in range(params['num_runs']):
        tr, tl, tll, bs, ba = trainAgents(my_env, params)
        train_returns.append(tr)
        train_losses.append(tl)
        train_lengths.append(tll)
        if bs > best_score:
            best_score = bs
            best_agent = ba

    return train_returns, train_losses, train_lengths, best_agent

def trainAgentFromParams(settingsFile):
    if os.path.isdir(settingsFile):
        settingsFile = os.path.join(settingsFile, 'settings.json')
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

    # train agents
    try:
        train_returns, train_losses, train_lengths, bestAgent = trainAgentsMultipleTimes(params)
    except Exception:
        print("\nFailure during training")
        print(traceback.format_exc())
        return

    # save agent
    try:
        torch.save(bestAgent.behaviorPolicyNet.state_dict(), os.path.join(absFolder, 'agent_dict.pt'))
    except Exception:
        print('Failed to write agent')
        print(traceback.format_exc)

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
