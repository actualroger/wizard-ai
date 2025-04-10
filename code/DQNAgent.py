
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
        match self.params['qnet_type']:
            case 'DeepQNet':
                q_behavior = self.behaviorPolicyNet.forward(obs_tensor).gather(-1, actions_tensor.unsqueeze(-1)).squeeze(-1)
            case 'ConvDeepQNet': # it is annoying that the output dimensions are different on this one
                q_behavior = self.behaviorPolicyNet.forward(obs_tensor).gather(-1, actions_tensor)
            case _: # should throw error, but we shouldn't get here because constructor should crash
                pass
        return q_behavior

    # evaluate target network
    def evaluateTargetNetwork(self, batch_data_tensor):
        # get the transition data
        next_obs_tensor = batch_data_tensor['next_obs']
        rewards_tensor = batch_data_tensor['reward']
        dones_tensor = batch_data_tensor['done']

        # compute the TD target using the target network
        with torch.no_grad():
            match self.params['qnet_type']:
                case 'DeepQNet':
                    q_target = rewards_tensor + self.params['gamma'] * (1 - dones_tensor) * self.targetPolicyNet.forward(next_obs_tensor).max(-1).values
                case 'ConvDeepQNet': # it is annoying that the output dimensions are different on this one
                    q_target = rewards_tensor + self.params['gamma'] * (1 - dones_tensor) * self.targetPolicyNet.forward(next_obs_tensor).max(-1).values.unsqueeze(-1)
                case _: # should throw error, but we shouldn't get here because constructor should crash
                    pass
        return q_target
