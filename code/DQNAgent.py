
import torch

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
        behaviorPolicyNet = DeepQNet(params)

        super().__init__(env, params, behaviorPolicyNet=behaviorPolicyNet)

    # update behavior policy
    def updateBehaviorPolicy(self, batch_data):
        # convert batch data to tensor and put them on device
        batch_data_tensor = self._batch_to_tensor(batch_data)

        # get the transition data
        obs_tensor = batch_data_tensor['obs']
        actions_tensor = batch_data_tensor['action']
        action_masks_tensor = batch_data_tensor['action_mask']
        next_obs_tensor = batch_data_tensor['next_obs']
        rewards_tensor = batch_data_tensor['reward']
        dones_tensor = batch_data_tensor['done']

        # compute the q value estimation using the behavior network
        q_behavior = self.behaviorPolicyNet.forward(obs_tensor).gather(-1, actions_tensor.unsqueeze(-1)).squeeze(-1)

        # compute the TD target using the target network
        with torch.no_grad():
            q_target = rewards_tensor + self.params['gamma'] * (1 - dones_tensor) * self.targetPolicyNet.forward(next_obs_tensor).max(-1).values

        # compute the loss
        td_loss = self.evaluateLoss(q_behavior, q_target)

        # minimize the loss
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

        return td_loss.item()

# convolutional DQN Agent
class ConvDQNAgent(NNAgent):
    # initialize the agent
    def __init__(self,
                 env,
                 params,
                 ):

        # create value network
        behaviorPolicyNet = ConvDeepQNet(params)

        super().__init__(env, params, behaviorPolicyNet=behaviorPolicyNet)

    # update behavior policy
    def updateBehaviorPolicy(self, batch_data):
        # convert batch data to tensor and put them on device
        batch_data_tensor = self._batch_to_tensor(batch_data)

        # get the transition data
        obs_tensor = batch_data_tensor['obs']
        actions_tensor = batch_data_tensor['action']
        action_masks_tensor = batch_data_tensor['action_mask']
        next_obs_tensor = batch_data_tensor['next_obs']
        rewards_tensor = batch_data_tensor['reward']
        dones_tensor = batch_data_tensor['done']

        # compute the q value estimation using the behavior network
        q_behavior = self.behaviorPolicyNet.forward(obs_tensor).gather(-1, actions_tensor)

        # compute the TD target using the target network
        with torch.no_grad():
            q_target = rewards_tensor + self.params['gamma'] * (1 - dones_tensor) * self.targetPolicyNet.forward(next_obs_tensor).max(-1).values.unsqueeze(-1)

        # compute the loss
        td_loss = self.evaluateLoss(q_behavior, q_target)

        # minimize the loss
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

        return td_loss.item()
