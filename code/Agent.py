
import torch
import torch.optim as opt
import torch.nn as nn
import numpy as np
import copy

import Buffer
import Schedule
from DeepQNet import customized_weights_init
from WizardDeck import Suit

# defines an Agent class which all other agents should descend from
class Agent(object):
    # constructor
    def __init__(self, env, params):
        self.env = env # environment
        self.params = params # params dictionary

        # environment parameters
        self.action_dim = env.ACTION_DIM
        self.obs_dim = env.OBSERVATION_DIM

    # action evaluation
    def getAction(self, obs) -> int:
        raise NotImplementedError("Agent subclass must implement getAction(obs)->int ")

    def clearPreviousState(self):
        pass

    def storeReward(self, reward, done):
        pass

    def storeExperience(self, newState, reward, done):
        pass


# defines a Neural Net Agent class which all other agents that contain NNs should descend from
class NNAgent(Agent):
    # constructor
    def __init__(self, env, params,
                 behaviorPolicyNet: nn.Module = None,
                 buffer: Buffer.Buffer = None,
                 schedule: Schedule.Schedule = None,
                 optimizer: opt.Optimizer = None,
                 loss: nn.modules.Module = None):
        super().__init__(env, params)

        # initialize networks
        self.behaviorPolicyNet = behaviorPolicyNet # network which is used to directly evaluate actions
        self.targetPolicyNet = copy.deepcopy(self.behaviorPolicyNet) # network which is used to train behavior
        self.behaviorPolicyNet.apply(customized_weights_init)
        self.targetPolicyNet.load_state_dict(self.behaviorPolicyNet.state_dict())

        # buffer used to store (s,a,r,s',d) experiences
        if buffer is None:
            match params['buffer_type']:
                case 'replay':
                    self.buffer = Buffer.ReplayBuffer(params['replay_buffer_size'])
                case 'action_priority':
                    self.buffer = Buffer.ActionPriorityBuffer(params['replay_buffer_size'], params)
                case 'q_priority':
                    self.buffer = Buffer.QPriorityBuffer(params['replay_buffer_size'], params)
                case _:
                    raise RuntimeError("No buffer provided")
        else:
            self.buffer = buffer

        # epsilon schedule
        if schedule is None:
            epsParams = {'schedule_type' : params['epsilon_schedule_type'],
                         'start_value' : params['epsilon_start_value'],
                         'end_value' : params['epsilon_end_value'],
                         'duration' : params['epsilon_duration']}
            self.schedule = Schedule.createSchedule(epsParams)
            if self.schedule is None:
                raise RuntimeError("No epsilon schedule provided")
        else:
            self.schedule = schedule

        # optimizer
        if optimizer is None:
            match params['optimizer']:
                case 'Adam':
                    self.optimizer = torch.optim.Adam(self.behaviorPolicyNet.parameters(), lr=params['learning_rate'])
                case _:
                    raise RuntimeError("No optimizer provided")
        else:
            self.optimizer = optimizer

        # loss function
        if loss is None:
            match params['loss']:
                case 'MSELoss':
                    self.lossFunction = nn.MSELoss()
                case _:
                    raise RuntimeError("No loss function provided")
        else:
            self.lossFunction = loss

        # send the agent to a specific device: cpu or gpu
        self.device = torch.device(params['device'])
        self.behaviorPolicyNet.to(self.device)
        self.targetPolicyNet.to(self.device)

        # set training parameters
        self.freqUpdateBehaviorPolicy = params['freq_update_behavior_policy']
        self.freqUpdateTargetPolicy = params['freq_update_target_policy']
        self.batchSize = params['batch_size']
        self.startTrainingStep = params['start_training_step']

        # keep track of training
        self.timestep = 0
        self.learningActive = 1 # are we learning
        self.epsilonActive = True # is epsilon allowed to be nonzero

        self.trainingLosses = [] # loss from each time we train behavior net
        self.clearPreviousState()

    # convert observation into nn-readable vector
    def processObservation(self, obs):
        observation = obs["observation"]
        state = [
                observation["selfNeeded"], # personal bet remaining
                observation["totalNeeded"], # table bet remaining
                observation["numPlayers"], # number of players
                int(observation["trump"] != Suit.NONE), # whether there is a trump suit
                int(observation["led"] == observation["trump"]), # whether trump == led
            ] + [
                2 if i in observation["pile"] else # card in pile
                1 if i in observation["hand"] else # card in hand
                -1 if i in observation["previous"] else # card in previous pile
                0 # card unseen
                for i in self.env.deck.reorderCards(observation["trump"], observation["led"])
            ]
        return state, obs["actionMask"]

    # get action
    def getAction(self, obs):
        state, actionMask = self.processObservation(obs) # process info
        state = self._arr_to_tensor(state).view(1, -1)

        self.storeExperience(state) # store experience tuple if the rest of the tuple is present

        # with probability eps, the agent selects a random action
        if self.epsilonActive and np.random.random() < self.schedule.getValue(self.timestep):
            availableActions = [i for i in range(len(actionMask)) if actionMask[i]]
            action = np.random.choice(availableActions, 1)[0]
        else:  # with probability 1 - eps, the agent selects a greedy policy
            with torch.no_grad():
                actionMaskTensor = torch.tensor(actionMask, dtype=bool)
                q_values = self.behaviorPolicyNet.forward(state).masked_fill(~actionMaskTensor, float('-inf'))
                action = q_values.max(dim=-1)[1].item()

        if self.learningActive: # learn if we want to
            self.previousState = state
            self.previousAction = action

        if self.epsilonActive: # update timestep
            self.timestep += 1

        return action

    # update behavior policy and return loss
    def updateBehaviorPolicy(self, batch_data) -> float:
        raise NotImplementedError("NNAgent subclass must implement updateBehaviorPolicy(batch_data) -> float")

    # evaluate behavior network
    def evaluateBehaviorNetwork(self, batch_data_tensor):
        raise NotImplementedError("NNAgent subclass must implement evaluateBehaviorNetwork(batch_data_tensor) -> tensor")

    # evaluate target network
    def evaluateTargetNetwork(self, batch_data_tensor):
        raise NotImplementedError("NNAgent subclass must implement evaluateTargetNetwork(batch_data_tensor) -> tensor")

    # evaluate loss function
    def evaluateLoss(self, q_behavior, q_target):
        baseLoss = self.lossFunction(q_behavior, q_target)

        # apply weighting if necessary
        match self.params['buffer_type']:
            case 'action_priority':
                loss = (baseLoss / torch.tensor(self.buffer.prev_weights)).mean()
            case 'q_priority':
                errors = torch.abs(q_behavior - q_target)
                self.buffer.setErrors(self.buffer.prev_indices, errors)
                loss = (torch.FloatTensor(self.buffer.prev_weights) * baseLoss).mean()
            case _:
                loss = baseLoss

        return loss

    # update target policy
    def updateTargetPolicy(self):
        # hard update
        self.targetPolicyNet = copy.copy(self.behaviorPolicyNet)

    # set learning active
    def setLearningActive(self, value):
        self.learningActive = value

    # set epsilon active
    def setEpsilonActive(self, value): # also pauses timestep
        self.epsilonActive = int(value)

    # remove stored states
    def clearPreviousState(self):
        self.previousState = None
        self.previousAction = None
        self.previousReward = None
        self.previousDone = None

    # store previous reward
    def storeReward(self, reward, done):
        self.previousReward = reward
        self.previousDone = done
        if done: # this is a terminal action and we need to store it because newState will not be provided
            self.storeExperience(self.previousState) # provided value is arbitrary besides dimension

    # store new state, which should complete most recent experience tuple
    def storeExperience(self, newState):
        if self.previousState is not None and self.previousReward is not None:
            self.addToBuffer(
                self.previousState,
                self.previousAction,
                self.previousReward,
                newState,
                self.previousDone )

    # add experience to buffer
    def addToBuffer(self, *args):
        self.buffer.add(*args)
        if self.params['buffer_type'] == 'q_priority': # update buffer weights
            with torch.no_grad():
                next_idx = self.buffer.__len__() - 1 if self.buffer._next_idx == 0 else self.buffer._next_idx - 1
                encodedSample = self.buffer._encode_sample([next_idx])
                batch_data_tensor = self._batch_to_tensor(encodedSample)
                q_behavior = self.evaluateBehaviorNetwork(batch_data_tensor)
                q_target = self.evaluateTargetNetwork(batch_data_tensor)
                error = abs(q_behavior - q_target)
                self.buffer.addError(error.item())
        self.checkLearning()

    # consider updating the neural nets
    def checkLearning(self):
        if self.learningActive and self.timestep >= self.startTrainingStep:
            if not np.mod(self.timestep, self.freqUpdateBehaviorPolicy): # update behavior with batch and save loss
                self.trainingLosses.append(self.updateBehaviorPolicy(self.buffer.sampleBatch(self.batchSize)))
            if not np.mod(self.timestep, self.freqUpdateTargetPolicy): # hard update target policy
                self.updateTargetPolicy()

    # auxiliary functions
    def _arr_to_tensor(self, arr):
        arr = np.array(arr)
        arr_tensor = torch.from_numpy(arr).float().to(self.device)
        return arr_tensor

    def _batch_to_tensor(self, batch_data):
        # store the tensor
        batch_data_tensor = {'obs': [], 'action': [], 'reward': [], 'next_obs': [], 'done': []}
        # get the numpy arrays
        obs_arr = batch_data[0]
        action_arr = batch_data[1]
        reward_arr = batch_data[2]
        next_obs_arr = batch_data[3]
        done_arr = batch_data[4]
        # convert to tensors
        batch_data_tensor['obs'] = torch.tensor(obs_arr, dtype=torch.float32).to(self.device)
        batch_data_tensor['action'] = torch.tensor(action_arr).long().view(-1, 1).to(self.device)
        batch_data_tensor['reward'] = torch.tensor(reward_arr, dtype=torch.float32).view(-1, 1).to(self.device)
        batch_data_tensor['next_obs'] = torch.tensor(next_obs_arr, dtype=torch.float32).to(self.device)
        batch_data_tensor['done'] = torch.tensor(done_arr, dtype=torch.float32).view(-1, 1).to(self.device)

        return batch_data_tensor


# empty Agent which functions as a shell for another agent
class PassThroughAgent(NNAgent):
    # constructor
    def __init__(self, baseAgent: NNAgent):
        self.baseAgent = baseAgent
        self.clearPreviousState()

    # ask agent for action
    def getAction(self, obs):
        state, actionMask = self.baseAgent.processObservation(obs) # process info
        state = self.baseAgent._arr_to_tensor(state).view(1, -1)

        self.storeExperience(state) # store experience tuple if the rest of the tuple is present

        action = self.baseAgent.getAction(obs) # ask agent for action

        if self.baseAgent.learningActive: # learn if we want to
            self.previousState = state # store our own experience
            self.previousAction = action

        return action

    # store previous reward
    def storeReward(self, reward, done):
        self.previousReward = reward
        self.previousDone = done
        if done: # this is a terminal action and we need to store it because newState will not be provided
            self.storeExperience(self.previousState) # provided value is arbitrary besides dimension

    # store new state, which should complete most recent experience tuple
    def storeExperience(self, newState):
        if self.previousState is not None and self.previousReward is not None: # send to base agent
            self.baseAgent.addToBuffer(
                self.previousState,
                self.previousAction,
                self.previousReward,
                newState,
                self.previousDone)
