
import random

from Agent import Agent

class RandomAgent(Agent):
    def __init__(self, env, verbosity: int = 0):
        super().__init__(env, {})
        self.verbosity = verbosity
    
    def getAction(self, obs) -> int:
        if self.verbosity > 1:
            print("Received game state:\n\tHand: %s\n\tPile: %s\n\tPrevious: %s\n\tTrump: %s\n\tNeeded: %d\n\tTotal Needed: %d" % (
                ' '.join(self.env.deck.toString(obs["observation"]["hand"])),
                ' '.join(self.env.deck.toString(obs["observation"]["pile"])),
                ' '.join(self.env.deck.toString(obs["observation"]["previous"])),
                ' '.join(self.env.deck.toSuitString(obs["observation"]["trump"])),
                obs["observation"]["selfNeeded"],
                obs["observation"]["totalNeeded"] )
                )
        
        availableActions = [i for i in range(len(obs["actionMask"])) if obs["actionMask"][i]]
        action = random.choice(availableActions)
        if self.verbosity > 0:
            print('Taking random action from', availableActions, '=', action)

        return action
