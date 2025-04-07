
import numpy as np
import random
import torch
from tqdm import tqdm

from PlotUtils import plotAgentScores, plotCurvesAutolabel, trimDims
import WizardEnvironment
from DQNAgent import DQNAgent
from RolloutUtils import rolloutGame

def trainMultipleDQNAgents(seed, numRuns, numAgents):
    # set the random seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # create environment
    my_env = WizardEnvironment.env(numPlayers=numAgents)

    # create training parameters
    train_parameters = {
        'num_agents': numAgents,
        'device': 'cpu',

        'observation_dim': my_env.OBSERVATION_DIM,
        'action_dim': my_env.ACTION_DIM,
        'hidden_layer_num': 2,
        'hidden_layer_dim': 64,
        'gamma': 1,

        'total_training_time_step': 5_00_000,

        'schedule_type': 'linear',
        'epsilon_start_value': 1.0,
        'epsilon_end_value': 0.01,
        'epsilon_duration': 250_000,

        'buffer_type': 'replay',
        'replay_buffer_size': 50000,
        'start_training_step': 2000,
        'freq_update_behavior_policy': 4,
        'freq_update_target_policy': 2000,

        'optimizer': 'Adam',
        'loss': 'MSELoss',
        'batch_size': 32,
        'learning_rate': 1e-3,

        'model_name': "wizard_%dp_DQN.pt"%numAgents
    }

    # TODO fill buffer with random agent data?

    # create experiment
    train_returns = []
    train_losses = []
    train_lengths = []
    for _ in range(numRuns):
        tr, tl, tll = trainDQNAgent(my_env, train_parameters)
        train_returns.append(tr)
        train_losses.append(tl)
        train_lengths.append(tll)

    return train_returns, train_losses, train_lengths

def trainDQNAgent(env, params):
    # create the DQN agents
    numAgents = params['num_agents']
    agents = [DQNAgent(env, params) for _ in range(numAgents)]

    # training variables
    train_returns = []
    train_lengths = []

    # start training
    total_length = 0
    pbar = tqdm(total=params['total_training_time_step'])
    while total_length < params['total_training_time_step']:

        scores, length = rolloutGame(env, agents)
        total_length += length
        train_lengths.append(length)
        pbar.update(length)

        train_returns.append(scores)

    train_losses = [agents[p].trainingLosses for p in range(numAgents)]

    pbar.close()
    # save the results
    return train_returns, train_losses, train_lengths

if __name__ == '__main__':
    seed = 0
    numRuns = 1
    numAgents = 4

    train_returns, train_losses, train_lengths = trainMultipleDQNAgents(seed, numRuns, numAgents)

    train_returns_trimmed = trimDims(train_returns) # trim to consistent dimensions
    train_returns_trimmed = np.array(train_returns_trimmed) # cast to numpy array
    train_returns_trimmed = np.transpose(train_returns_trimmed, (0,2,1,3)) # [run][game][player][hand] -> [run][player][game][hand]
    train_returns_trimmed = train_returns_trimmed.reshape((-1,) + train_returns_trimmed.shape[2:]) # flatten run and agent to one dimension
    plotAgentScores(train_returns_trimmed)
    # minLen = min([len(n) for n in train_returns])
    # plot_curves([np.array([n[:minLen] for n in train_returns])], ['DQN'], ['r'], 'discounted return', 'discounted returns wrt episode')

    train_losses_trimmed = trimDims(train_losses) # trim to consistent dimensions
    train_losses_trimmed = np.array(train_losses_trimmed) # cast to numpy array
    train_losses_trimmed = np.transpose(train_losses_trimmed, (1,0,2)) # [run][player][hand] -> [player][run][hand]
    # train_losses_trimmed = train_losses_trimmed.reshape((-1,) + train_losses_trimmed.shape[2:]) # flatten run and agent to one dimension
    plotCurvesAutolabel(train_losses_trimmed, xlabel='Training Step', ylabel='MSE Loss')
    # minLen = min([len(n) for n in train_loss])
    # plot_curves([np.array([n[:minLen] for n in train_loss])], ['DQN'], ['r'], 'training loss', 'loss wrt training steps')
