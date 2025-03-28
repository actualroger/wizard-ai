
def rolloutRound(env, agents, round, dealer: int = 0, verbosity: int = 0):
	env.reset(round, dealer=dealer)
	if verbosity > 0:
		print("\nStarted round %d with dealer %d\n" % (round, dealer))
	scores = [0] * len(agents)
	
	while not env.terminated:
		agent = env.currentAgent
		if verbosity > 0:
			print("\nAgent %d's turn" % agent)

		observation = env.observe(agent)
		if verbosity > 1:
			print("Agent %d observes\n\t" % agent, '\t\n'.join([' : '.join([str(k),str(v)]) for k,v in observation.items()]), sep='')
		
		action = agents[agent].getAction(observation)
		_, reward, _, _, _ = env.step(agent, action)
		if reward != 0:
			if verbosity > 0:
				print("Agent %d received reward %d"%(agent, reward))
			scores[agent] += reward
	
	if verbosity > 0:
		print("End of round scores:", scores, "\n")
	return scores

def rolloutGame(env, agents, minRound: int = 1, maxRound: int = -1, dealer: int = 0, verbosity: int = 0):
	numAgents = len(agents)
	scores = [[] for _ in range(numAgents)]
	if maxRound == -1:
		maxRound = env.deck.size // numAgents

	for round in range(minRound, maxRound + 1):
		roundScores = rolloutRound(env, agents, round, dealer=dealer, verbosity=verbosity)
		for agent in range(numAgents):
			scores[agent].append(roundScores[agent])
		dealer = (dealer + 1) % len(agents)

	if verbosity > 0:
		print("\nEnd of game")
	return scores
