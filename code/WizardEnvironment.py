
from typing import Optional
from enum import Enum

from WizardDeck import WizardDeck, Suit

class Phase(Enum):
	TERMINAL = 0
	PLAY = 1
	BET = 2
	TRUMP = 3

# class raw_env():
class env():
	metadata = {"renderModes":["ansi"]}

	def __init__(self, numPlayers: int = 3, renderMode: str = None, verbosity: int = 0):
		# number of players
		self.MIN_PLAYERS = 3
		self.MAX_PLAYERS = 6
		assert numPlayers in range(self.MIN_PLAYERS, self.MAX_PLAYERS + 1)
		self.numPlayers = numPlayers

		# render mode
		assert renderMode is None or renderMode in self.metadata["renderModes"]
		self.renderMode = renderMode

		# printing flag
		self.verbosity = verbosity

		# deck
		self.deck = WizardDeck()

		# round info
		self.bets = [] * self.numPlayers # bet tricks
		self.wins = [] * self.numPlayers # won tricks
		self.hands = [] * self.numPlayers # cards in hand
		self.pile = [] # cards in this trick pile
		self.previousPiles = [] # played cards
		self.trumpSuit = Suit.NONE # which suit is trump
		self.ledSuit = Suit.NONE # which suit was led
		self.playPhase = -1 # phase of play (-1 = done, 0 = normal play, 1 = betting, 2 = choose trump)
		self.leader = -1 # who dealt this hand

		# action masks
		self.ACTION_MASKS = {
			Phase.PLAY : [True] * 60 + [False] * (21 + 4 + 1), # play card 0-59
			Phase.BET : [False] * 60 + [True] * 21 + [False] * (4 + 1), # bet 0-20
			Phase.TRUMP : [False] * (60 + 21) + [True] * 4 + [False] * 1, # choose trump suit 1-4
			Phase.TERMINAL : [False] * (60 + 21 + 4) + [True] * 1, # null action to receive reward
		}
		self.ACTION_DIM = 86

		# observation
		self.OBSERVATION_DIM = 65

		# agent management
		self.dealer = -1 # who dealt
		self.currentAgent = -1 # current agent index
		self.agentTerminated = [] * self.numPlayers # whether this agent is terminated
		self.agentObservation = [] * self.numPlayers # agent last observation
		self.terminated = True # whole env is done

	def reset(self, round, dealer: Optional[int] = 0): # TODO add flags to force certain situations
		if self.verbosity > 0:
			print("Resetting with round = %d, dealer = %d" % (round, dealer))

		# reset the terminal flag
		self.terminated = False

		# deal hands
		assert round > 0 and round <= self.deck.size / self.numPlayers        
		self.deck.shuffle()
		self.hands = []
		for i in range(self.numPlayers):
			self.hands.append(self.deck.deal(round))

		# agent management
		self.dealer = dealer
		self.currentAgent = dealer # current agent index
		self.agentTerminated = [False] * self.numPlayers # whether this agent is terminated
		self.agentObservation = [{}] * self.numPlayers

		# reset other things
		self.pile = [] # empty pile
		self.playPhase = Phase.BET
		self.setTrump() # sets trump or enters that phase of play

		self.bets = [None for _ in range(self.numPlayers)] # bet tricks
		self.wins = [0] * self.numPlayers # won tricks
		self.previousPiles = [] # played cards
		self.ledSuit = Suit.NONE # which suit was led

	def setTrump(self): # sets the trump suit
		if self.deck.numCards == 0:
			if self.verbosity > 0:
				print("Set no trump (empty deck)")
			self.trumpSuit = Suit.NONE # no trump
			self.incrementAgent() # player after dealer bets first
			return
		
		self.previousPiles = self.deck.deal(1)
		rawSuit = self.deck.suits[self.previousPiles[0]]
		if rawSuit == Suit.JESTER: # jester
			self.trumpSuit = Suit.NONE
			if self.verbosity > 0:
				print("Setting no trump (Jester)")
			self.incrementAgent() # player after dealer bets first
		elif rawSuit == Suit.WIZARD: # wizard
			self.playPhase = Phase.TRUMP
			if self.verbosity > 0:
				print("Entering trump phase")
		else:
			self.trumpSuit = rawSuit # normal card
			if self.verbosity > 0:
				print("Set trump to", self.deck.toSuitString(self.trumpSuit))
			self.incrementAgent() # player after dealer bets first		

	def observe(self, agent): # construct dict of agent's observation
		obs = {"hand": self.hands[agent],
			   "pile": self.pile,
			   "previous": self.previousPiles,
			   "trump": self.trumpSuit,
			   "led": self.ledSuit,
			   "selfNeeded": (0 if self.bets[agent] is None else self.bets[agent]) - self.wins[agent],
			   "totalNeeded": sum([s for s in self.bets if s is not None]) - sum(self.wins),
			   "numPlayers": self.numPlayers}
		actionMask = self.actionMask(agent)
		observation = {"observation": obs, "actionMask": actionMask}
		self.agentObservation[agent] = observation
		return observation
	# TODO can agent distinguish between wizarded no-led and leading no-led?
	# TODO can agent tell how many agents have bet?

	def actionMask(self, agent): # gets allowable actions for agent
		if agent != self.currentAgent: # wrong turn
			return [0] * (60 + 21 + 4 + 1)
		
		actionMask = []
		if self.playPhase == Phase.TRUMP or self.playPhase == Phase.TERMINAL:
			actionMask = self.ACTION_MASKS[self.playPhase]
		elif self.playPhase == Phase.BET:
			roundNum = len(self.hands[0])
			actionMask = [False] * 60 + [True] * (1 + roundNum) + [False] * (20 - roundNum + 4 + 1) # bet 0-N
		elif self.playPhase == Phase.PLAY:
			if self.verbosity > 1:
				print('Finding cards from', self.deck.toString(self.hands[agent]), 'for pile', self.deck.toString(self.pile), 'led =', self.ledSuit)
			actionMask = [card in self.hands[agent] and ( # card must be in hand
							len(self.pile) == 0 or # we are the leading card
							self.ledSuit == Suit.JESTER or # led jester
							self.ledSuit == Suit.WIZARD or # wizard on pile
							self.deck.suits[card] == self.ledSuit) # we match the leading card
						   for card in range(self.deck.size)]
			if self.verbosity > 1:
				print('Found normal cards:', self.deck.toString([i for i in range(len(actionMask)) if actionMask[i]]))
			if not any(actionMask): # no valid cards in hand
				actionMask = [card in self.hands[agent] for card in range(self.deck.size)] # play any card in hand
				if self.verbosity > 1:
					print('No valid cards: allowing whole hand')
			else: # check for non-led allowed cards
				for i in range(len(self.hands[agent])):
					if (self.deck.suits[self.hands[agent][i]] == Suit.WIZARD or # wizards always allowed
						self.deck.suits[self.hands[agent][i]] == Suit.JESTER): # jesters always allowed
						actionMask[self.hands[agent][i]] = True
			if self.verbosity > 0:
				print('Agent can play:', self.deck.toString([i for i in range(len(actionMask)) if actionMask[i]]))
			actionMask = actionMask + [False] * (21 + 4 + 1)
		
		return actionMask

	def calculateReward(self, agent): # get agent score
		score = 10 * ((2 + self.bets[agent]) if self.wins[agent] == self.bets[agent] else -abs(self.wins[agent] - self.bets[agent]))
		if self.verbosity > 0:
			print('Agent %d bet %d and won %d for %d points' % (agent, self.bets[agent], self.wins[agent], score) )
		return score
	
	def incrementAgent(self): # gets next agent
		if self.verbosity > 0:
			print("Looking for agent after %d" % self.currentAgent)

		if self.terminated or all(self.agentTerminated): # all done
			self.currentAgent = -1
			self.terminated = True
			if self.verbosity > 0:
				print("Found no active agents")
			return self.currentAgent
		
		nextAgent = [i for i in range(self.currentAgent+1, self.numPlayers) if not self.agentTerminated[i]] # search rest of list
		if nextAgent: # found one
			self.currentAgent = nextAgent[0]
		else:
			nextAgent = [i for i in range(self.currentAgent) if not self.agentTerminated[i]] # search before list
			if nextAgent: # found one
				self.currentAgent = nextAgent[0]
		
		if self.verbosity > 0:
			print("Found next agent %d" % self.currentAgent)

		return self.currentAgent

	def step(self, agent, act):
		# default returns
		reward = 0
		truncation = False
		info = {}
		
		# debug print
		if self.verbosity > 1:
			print('Agent %d takes action %d'%(agent, act))

		# check if valid
		if "actionMask" not in self.agentObservation[agent] or not self.agentObservation[agent]["actionMask"][act]:
			print('Agent %d tried to take invalid action %d\n'%(agent, act))
			reward = -1000
			self.terminated = True # kill session
			return self.agentObservation[agent], reward, True, truncation, info

		# execute turn
		if self.playPhase == Phase.TERMINAL: # distribute points and mark player done
			reward = self.calculateReward(self.currentAgent)
			self.agentTerminated[agent] = True
			self.incrementAgent() # next player
		elif self.playPhase == Phase.PLAY:
			card = act # action 0-59 = card id 0-59

			# change led suit if necessary
			playedSuit = self.deck.suits[card]
			if playedSuit == Suit.WIZARD: # wizard destroys led suit
				self.ledSuit = Suit.WIZARD
			elif self.ledSuit == Suit.JESTER or self.ledSuit == Suit.NONE: # no valid led suit
				self.ledSuit = playedSuit
			
			# add card to pile and remove from hand
			self.pile.append(card)
			self.hands[agent].remove(card)
			
			# check for end of round
			if len(self.pile) == self.numPlayers:
				winner = self.leader # winning player (with led suit)
				winnerCard = self.pile[0]
				for i in range(1, self.numPlayers):
					otherPlayer = (self.leader + i) % self.numPlayers
					otherCard = self.pile[i]
					if self.verbosity > 1:
						print("Comparing agent %d's card %s with agent %d's card %s" % (winner, self.deck.toString(winnerCard)[0], otherPlayer, self.deck.toString(otherCard)[0]))
						print(self.deck.suits[otherCard] == self.deck.suits[winnerCard], # same suit
							self.deck.suits[otherCard] == Suit.WIZARD, # new wizard
							self.deck.suits[otherCard] == Suit.JESTER, # new jester
							self.deck.suits[winnerCard] == Suit.WIZARD, # winning wizard
							self.deck.suits[winnerCard] == Suit.JESTER, # winning jester
							self.deck.values[otherCard] > self.deck.values[winnerCard], # greater value
							self.deck.suits[otherCard] == self.trumpSuit, # new trump
							self.deck.suits[winnerCard] != self.trumpSuit, # old not trump
							self.deck.suits[winnerCard] != Suit.WIZARD, # also old not wizard
							self.trumpSuit
						)
					#       TRICK WINNER LOGIC:
					#               CURRENT
					#           Wiz Tru Led Oth Jes
					#       Wiz >   >   >   N/A >
					#       Tru >   >   Y   N/A >
					# NEW   Led >   X   >   N/A >
					#       Oth >   X   X   N/A >
					#       Jes >   >   >   N/A >
					if (
						   (self.deck.suits[otherCard] == self.deck.suits[winnerCard] or # same suit
							self.deck.suits[otherCard] == Suit.WIZARD or # new wizard
							self.deck.suits[otherCard] == Suit.JESTER or # new jester
							self.deck.suits[winnerCard] == Suit.WIZARD or # winning wizard
							self.deck.suits[winnerCard] == Suit.JESTER) and # winning jester
							self.deck.values[otherCard] > self.deck.values[winnerCard] # greater value
						) or (
							self.deck.suits[otherCard] == self.trumpSuit and # new trump
	   						self.deck.suits[winnerCard] != self.trumpSuit and # old not trump
							self.deck.suits[winnerCard] != Suit.WIZARD # also old not wizard
						):
						winner = otherPlayer # new winner
						winnerCard = otherCard
						if self.verbosity > 1:
							print("New winner")
					elif self.verbosity > 1:
						print("Same winner")
				self.wins[winner] += 1 # winner takes trick
				self.currentAgent = winner # winner plays next
				self.leader = self.currentAgent
				if self.verbosity > 0:
					print("Player %d won trick" % winner)
				self.previousPiles.extend(self.pile)
				self.pile = []
				self.ledSuit = Suit.NONE
			else:
				self.incrementAgent() # next player
		elif self.playPhase == Phase.BET:
			self.bets[agent] = act - 60 # 60-80 -> bet 0-20
			self.incrementAgent() # next player
		elif self.playPhase == Phase.TRUMP:
			self.trumpSuit = Suit(act - 80) # 81-84 -> suits 1-4
			if self.verbosity > 0:
				print("Dealer set trump to", self.deck.toSuitString(self.trumpSuit))
			self.incrementAgent() # next player

		# check for phase change
		if self.playPhase == Phase.PLAY and not any(self.hands): # end of round
			self.playPhase = Phase.TERMINAL
			if self.verbosity > 0:
				print("Entering Terminal")
		elif self.playPhase == Phase.BET and all([a is not None for a in self.bets]): # everyone has bet
			self.leader = self.currentAgent
			self.playPhase = Phase.PLAY
			if self.verbosity > 0:
				print("Entering Play with leader %d" % self.currentAgent)
		elif self.playPhase == Phase.TRUMP and self.trumpSuit != Suit.NONE: # trump has been set
			self.playPhase = Phase.BET
			if self.verbosity > 0:
				print("Entering Bet starting with agent %d" % self.currentAgent)
		
		# return
		return self.agentObservation[agent], reward, self.agentTerminated[agent], truncation, info

	def render(self):
		if self.render_mode == "ansi":
			return(self.toString())
	
	def toString(self):
		return ['Trump: %s '%(self.deck.toSuitString(self.trumpSuit), )]
