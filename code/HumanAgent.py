
from WizardEnvironment import Phase
from WizardDeck import WizardDeck, Suit

class HumanAgent:
    def __init__(self, env):
        self.env = env
    
    def getAction(self, obs):
        # print observation
        print("Received game state:\n\tHand: %s\n\tPile: %s\n\tPrevious: %s\n\tTrump: %s\n\tNeeded: %d\n\tTotal Needed: %d" % (
              ' '.join(self.env.deck.toString(obs["observation"]["hand"])),
              ' '.join(self.env.deck.toString(obs["observation"]["pile"])),
              ' '.join(self.env.deck.toString(obs["observation"]["previous"])),
              ' '.join(self.env.deck.toSuitString(obs["observation"]["trump"])),
              obs["observation"]["selfNeeded"],
              obs["observation"]["totalNeeded"] )
              )
        # print action mask
        print("Action mask ", [i for i in range(len(obs["actionMask"])) if obs["actionMask"][i]] )

        if self.env.playPhase == Phase.TRUMP: # choose trump suit
            # print suits for user
            suitNames = [self.env.deck.toSuitString([Suit.DIAMONDS, Suit.SPADES, Suit.HEARTS, Suit.CLUBS])]
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
            allowedHand = [c for c in obs["observation"]["hand"] if obs["actionMask"][c] ]
            allowedHandString = self.env.deck.toString(allowedHand)
            # until we have good response
            while True:
                userInput = input("Input card from %s : " % ' '.join(allowedHandString))
                # test user input
                if userInput.upper() in allowedHandString: # only return if valid
                    return [allowedHand[i] for i in range(len(allowedHand)) if userInput.upper() == allowedHandString[i]][0]
                elif len(allowedHandString) == 1: # one move
                    return allowedHand[0]
                else:
                    print('Invalid input')
        
        else: # phase = terminal
            print("Taking null action")
            return 85
