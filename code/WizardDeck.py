
import numpy.random

class WizardDeck:
	def __init__(self):
		# lists of card suits and values
		self.suits = []
		self.values = []

		# jesters
		for j in range(4):
			self.suits.append(0)
			self.values.append(1)

		# normal deck
		for s in range(1,4+1):
			for v in range(2,14+1):
				self.suits.append(s)
				self.values.append(v)

		# wizards
		for w in range(4):
			self.suits.append(5)
			self.values.append(15)

		self.size = len(self.suits) # total deck size
		self.numCards = len(self.suits) # number of undealt cards
		self.state = [v for v in range(len(self.suits))] # current deck state (indexes)

		self.suitNames = ['D','S','H','C'] # diamonds, spades, hearts, clubs
		self.cardNames = [str(d) for d in range(2,9+1)] + ['X','J','Q','K','A'] # 2 - Ace

	def reset(self):
		# reset without shuffling
		self.numCards = self.size

	def shuffle(self):
		# reshuffle deck
		self.numCards = self.size
		numpy.random.shuffle(self.state)

	def deal(self, n):
		# deal n cards
		curNum = self.numCards
		n = min(n, self.numCards)
		self.numCards -= n
		return self.state[self.numCards:curNum]

	def toString(self, cards):
		if not isinstance(cards, list):
			cards = [cards]
		# converts indices to human-readable card names
		return [('E' + str(c+1)) if self.suits[c] == 0 else # jester
			('W' + str(c-55)) if self.suits[c] == 5 else # wizard
			(self.cardNames[self.values[c] - 2] + self.suitNames[self.suits[c] - 1]) # suit, value
			for c in cards]
