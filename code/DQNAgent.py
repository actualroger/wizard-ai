
import torch
import sys

from DeepQNet import *
from Agent import NNAgent

# DQN Agent
class DQNAgent(NNAgent):
    # initialize the agent
    def __init__(self,
                 env,
                 params,
                 ):

        # create value network
        qnetClass = getattr(sys.modules[__name__], params['qnet_type']) # find class with name
        behaviorPolicyNet = qnetClass(params)

        super().__init__(env, params, behaviorPolicyNet=behaviorPolicyNet)

    # update behavior policy
    def updateBehaviorPolicy(self, batch_data):
        # convert batch data to tensor and put them on device
        batch_data_tensor = self._batch_to_tensor(batch_data)

        # compute the q value estimation using the behavior network
        q_behavior = self.evaluateBehaviorNetwork(batch_data_tensor)

        # compute the TD target using the target network
        q_target = self.evaluateTargetNetwork(batch_data_tensor)

        # compute the loss
        td_loss = self.evaluateLoss(q_behavior, q_target)

        # minimize the loss
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

        return td_loss.item()

    # evaluate behavior network
    def evaluateBehaviorNetwork(self, batch_data_tensor):
        # get the transition data
        obs_tensor = batch_data_tensor['obs']
        actions_tensor = batch_data_tensor['action']

        # compute the q value estimation using the behavior network
        q_behavior = self.behaviorPolicyNet.forward(obs_tensor).gather(-1, actions_tensor)
        return q_behavior # shape should be [batch_size, 1]

    # evaluate target network
    def evaluateTargetNetwork(self, batch_data_tensor):
        # get the transition data
        next_obs_tensor = batch_data_tensor['next_obs']
        rewards_tensor = batch_data_tensor['reward']
        dones_tensor = batch_data_tensor['done']

        # compute the TD target using the target network
        with torch.no_grad():
            q_target = rewards_tensor + self.params['gamma'] * (1 - dones_tensor) * self.targetPolicyNet.forward(next_obs_tensor).max(-1).values.unsqueeze(-1)
        return q_target # shape should be [batch_size, 1]

# double DQN Agent
class DDQNAgent(DQNAgent):
    # constructor
    def __init__(self, env, params):
        super().__init__(env, params)

    # evaluate target network
    def evaluateTargetNetwork(self, batch_data_tensor):
        # get the transition data
        next_obs_tensor = batch_data_tensor['next_obs']
        rewards_tensor = batch_data_tensor['reward']
        dones_tensor = batch_data_tensor['done']

        # compute the TD target using the target network
        with torch.no_grad():
            # consult behavior network for action recommendations
            next_actions_tensor = self.behaviorPolicyNet.forward(next_obs_tensor).argmax(dim=-1)
            q_target = rewards_tensor + self.params['gamma'] * (1 - dones_tensor) * torch.gather(self.targetPolicyNet.forward(next_obs_tensor), -1, next_actions_tensor.unsqueeze(-1))
        return q_target # shape should be [batch_size, 1]

# PPO Agent
class PPOAgent(DQNAgent):
    # constructor
    def __init__(self, env, params):
        super().__init__(env, params)

    # get action
    def getAction(self, obs): # TODO include storing action prob
        return super().getAction(obs)

    # update behavior policy
    def updateBehaviorPolicy(self, batch_data):
        return super().updateBehaviorPolicy(batch_data)

    # clear previous states
    def clearPreviousState(self):
        self.previousActionProb = None
        return super().clearPreviousState()

    # store new state
    def storeExperience(self, newState):
        if self.previousState is not None and self.previousReward is not None:
            self.addToBuffer(
                self.previousState,
                self.previousAction,
                self.previousActionProb,
                self.previousReward,
                newState,
                self.previousDone )

    # process batch data (which now has action probabilities/logits)
    def _batch_to_tensor(self, batch_data): # TODO
        return super()._batch_to_tensor(batch_data)

# TODO LOSS TYPES?
