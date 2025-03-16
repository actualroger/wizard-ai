
import numpy as np
from typing import Optional
import gymnasium as gym
from WizardDeck import WizardDeck

class WizardEnvironment(gym.Env):
    def __init__(self, numPlayers: int = 3):
        # number of players
        self.numPlayers = numPlayers
        

        # deck
        self.deck = WizardDeck()

    def reset(self, seed: Optional[int] = None, ):
        # reset the terminal flag
        terminated = False
        return self.state, terminated

    def step(self, act, forcedChance=-1):
        self.act = act # store action
        self.t += 1 # increment time

        if self.state == self.goal_state: # we are at goal
            reward = 0
            terminated = True
            return self.state, reward, terminated
        
        new_state = self.state + self.action_space[act] # evaluate new state

        windStrength = self.wind[self.state[1]] # evaluate wind
        if self.stochastic and windStrength > 0:
            windChoices = [-1,0,1]
            if forcedChance in range(len(windChoices)): # if we have the flag forcing which random occurence (for DP evaluation)
                windStrength += windChoices[forcedChance]
            else:
                windStrength += np.random.choice([-1,0,1]) # actual randomness
        new_state += np.array([-windStrength, 0])

        if new_state[0] not in range(6) or new_state[1] not in range(10): # see if we are on board
            new_state = [min(max(new_state[0],0),6),min(max(new_state[1],0),9)]
        self.state = list(new_state)

        reward = -1
        terminated = False

        return self.state, reward, terminated

    def render(self):
        # plot the agent and the goal
        # agent = 1
        # goal = 2
        plot_arr = self.grid.copy()
        plot_arr[self.state[0], self.state[1]] = 1.0
        plot_arr[self.goal_state[0], self.goal_state[1]] = 2.0
        plt.clf()
        fig, arr = plt.subplots(1, 1)
        arr.set_title(f"state={self.state}, act={self.act}")
        arr.imshow(plot_arr)
        plt.show(block=False)
        plt.pause(1)
        plt.close(fig)
