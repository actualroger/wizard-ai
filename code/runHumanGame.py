
import WizardEnvironment
from HumanAgent import HumanAgent

numPlayers = 3
startRound = 1
endRound = 3
dealer = 0
scores = [0] * numPlayers

seed = 0

env = WizardEnvironment.env(renderMode="ansi", numPlayers=numPlayers, verbose=True)
agents = [HumanAgent(env) for i in range(numPlayers)]

for round in range(startRound, endRound + 1):
    if round == 1:
        env.reset(round, dealer=dealer, seed=seed)
    else:
        env.reset(round, dealer=dealer)
    print("\nStarted round %d with dealer %d\n" % (round, dealer))
    dealer = (dealer + 1) % numPlayers
    
    while not env.terminated:
        agent = env.currentAgent
        print("\nAgent %d's turn" % agent)
        observation = env.observe(agent)
        # print("Agent %d observes\n\t" % agent, '\t\n'.join([' : '.join([str(k),str(v)]) for k,v in observation.items()]), sep='')
        
        action = agents[agent].getAction(observation)
        _, reward, _, _, _ = env.step(agent, action)
        if reward != 0:
            print("Agent %d received reward %d"%(agent, reward))
            scores[agent] += reward
    
    # scores = [scores[i] + env.scores[i] for i in range(numPlayers)] # record scores
    print("End of round scores:", scores, "\n")

print("\nEnd of game")
env.close() # end of game
