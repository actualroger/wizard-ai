
import numpy as np
from typing import Optional
import gymnasium as gym
from WizardDeck import WizardDeck

class WizardEnvironment(gym.Env):
    metadata = {"render_modes":["ansi"]}

    def __init__(self, numPlayers: int = 3, renderMode = None):
        # number of players
        self.MIN_PLAYERS = 3
        self.MAX_PLAYERS = 6
        assert numPlayers in range(self.MIN_PLAYERS, self.MAX_PLAYERS + 1)
        self.numPlayers = numPlayers

        # render mode
        assert renderMode is None or renderMode in self.metadata["render_modes"]
        self.renderMode = renderMode

        # deck
        self.deck = WizardDeck()

        # round info
        self.bets = [0] * self.numPlayers
        self.hands = [] * self.numPlayers        

    def reset(self, hand, seed: Optional[int] = None): # todo add flags to force certain situations
        # reset rng
        super().reset(seed=seed)

        # deal hands
        assert hand > 0 and hand <= self.deck.size / hand        
        self.deck.shuffle()
        for i in range(self.numPlayers):
            self.hands[i] = self.deck.deal()

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
        if self.render_mode == "ansi":
            return(self.toString())
    
    def toString(self):

