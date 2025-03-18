
class HumanAgent:
    def __init__(self, env):
        self.env = env
    
    def getAction(self, obs):
        while True:
            userInput = input("Input direction [0-3]: ")
            if len(userInput) == 1 and userInput in '0123':
                return int(userInput) # only return if valid
