
from WizardEnvironment import Phase
from WizardDeck import Suit
from Agent import Agent

class HumanAgent(Agent):
    def __init__(self, env):
        super().__init__(env, {})
    
    def getAction(self, obs) -> int:
        # print observation
        print("Received game state:\n\tHand: %s\n\tPile: %s\n\tPrevious: %s\n\tPrevious Sorted: %s\n\tTrump: %s\n\tLed: %s\n\tNeeded: %d\n\tTotal Needed: %d\n\tPlayers: %d" % (
              ' '.join(self.env.deck.toString(self.env.deck.reorderGroup(obs["observation"]["hand"], obs["observation"]["trump"], obs["observation"]["led"]))),
              ' '.join(self.env.deck.toString(obs["observation"]["pile"])),
              ' '.join(self.env.deck.toString(obs["observation"]["previous"])),
              ' '.join(self.env.deck.toString(self.env.deck.reorderGroup(obs["observation"]["previous"], obs["observation"]["trump"], obs["observation"]["led"]))),
              ' '.join(self.env.deck.toSuitString(obs["observation"]["trump"])),
              ' '.join(self.env.deck.toSuitString(obs["observation"]["led"])),
              obs["observation"]["selfNeeded"],
              obs["observation"]["totalNeeded"],
              obs["observation"]["numPlayers"] )
              )
        # print action mask
        print("Action mask ", [i for i in range(len(obs["actionMask"])) if obs["actionMask"][i]] )

        if self.env.playPhase == Phase.TRUMP: # choose trump suit
            # print suits for user
            suitNames = [self.env.deck.toSuitString(s) for s in [Suit.DIAMONDS, Suit.SPADES, Suit.HEARTS, Suit.CLUBS]]
            # until we have good response
            while True:
                userInput = input("Input suit from %s : " % ' '.join(suitNames))
                # test user input
                if len(userInput) == 1 and userInput in suitNames: # only return if valid
                    return 81 + [i for i in range(len(suitNames)) if suitNames[i] == userInput][0]
                else:
                    print('Invalid input')
            
        elif self.env.playPhase == Phase.BET: # choose bet
            # print bets for user
            maxBet = sum(obs["actionMask"]) - 1
            # until we have good response
            while True:
                userInput = input("Input bet from [ 0 - %d ] : " % maxBet)
                # test user input
                try:
                    intValue = int(userInput)
                except Exception:
                    print('Invalid input')
                    continue
                if userInput and intValue in range(maxBet + 1): # only return if valid
                    return 60 + intValue
                else:
                    print('Invalid input')
        
        elif self.env.playPhase == Phase.PLAY: # choose card
            # print hand for user
            allowedHandSorted = [c for c in
                                 self.env.deck.reorderGroup(obs["observation"]["hand"], obs["observation"]["trump"], obs["observation"]["led"])
                                 if obs["actionMask"][c]]
            allowedHandSortedString = self.env.deck.toString(allowedHandSorted)
            # until we have good response
            while True:
                userInput = input("Input card from %s : " % ' '.join(allowedHandSortedString))
                # test user input
                if userInput.upper() in allowedHandSortedString: # only return if valid
                    return [allowedHandSorted[i] for i in range(len(allowedHandSorted)) if userInput.upper() == allowedHandSortedString[i]][0]
                elif len(allowedHandSortedString) == 1: # one move
                    return allowedHandSorted[0]
                else:
                    print('Invalid input')
        
        else: # phase = terminal
            print("Taking null action")
            return 85
