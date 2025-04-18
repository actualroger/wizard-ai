
import numpy.random
from enum import Enum

class Suit(Enum):
	JESTER = 0
	DIAMONDS = 1
	SPADES = 2
	HEARTS = 3
	CLUBS = 4
	WIZARD = 5
	NONE = 6

class WizardDeck:
	def __init__(self):
		# lists of card suits and values
		self.suits = []
		self.values = []

		# jesters
		for j in range(4):
			self.suits.append(Suit.JESTER)
			self.values.append(1)

		# normal deck
		for v in range(2,14+1):
			self.suits.append(Suit.DIAMONDS)
			self.values.append(v)
		for v in range(2,14+1):
			self.suits.append(Suit.SPADES)
			self.values.append(v)
		for v in range(2,14+1):
			self.suits.append(Suit.HEARTS)
			self.values.append(v)
		for v in range(2,14+1):
			self.suits.append(Suit.CLUBS)
			self.values.append(v)
		
		# wizards
		for w in range(4):
			self.suits.append(Suit.WIZARD)
			self.values.append(15)

		self.size = len(self.suits) # total deck size
		self.numCards = len(self.suits) # number of undealt cards
		self.state = [v for v in range(len(self.suits))] # current deck state (indexes)

		self.suitNames = {
			Suit.JESTER : 'E',
			Suit.DIAMONDS : 'D',
			Suit.SPADES : 'S',
			Suit.HEARTS : 'H',
			Suit.CLUBS : 'C',
			Suit.WIZARD : 'W'
			}
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

	def reorderCards(self, trumpSuit: Suit, ledSuit: Suit):
		# reorders deck indices to [wizard, trump, led, other, other, jester]
		#						or [wizard, trump/led, other, other, other, jester]
		#						or [wizard, led, other, other, other, jester]
		#						or [wizard, other, other, other, other, jester]
		# with suits staying in default order if either trump or led not in four normal suits
		normalSuits = [Suit.DIAMONDS, Suit.SPADES, Suit.HEARTS, Suit.CLUBS]
		sortedSuits = [trumpSuit, ledSuit] + normalSuits
		sortedSuits = [sortedSuits[i] for i in range(len(sortedSuits)) if sortedSuits[i] in normalSuits and sortedSuits[i] not in sortedSuits[:i]]
		return [56,57,58,59] + [i for s in sortedSuits for i in range((s.value-1)*13+4, s.value*13+4)] + [0,1,2,3]
		# TODO this puts low trumps next to wizards

	def reorderGroup(self, cards, trumpSuit: Suit, ledSuit: Suit):
		# reorders group of cards via reorderCards
		return [c for c in self.reorderCards(trumpSuit, ledSuit) if c in cards]

	def toString(self, cards):
		if not isinstance(cards, list):
			cards = [cards]
		# converts indices to human-readable card names
		return [('E' + str(c+1)) if self.suits[c] == Suit.JESTER else # jester
			('W' + str(c-55)) if self.suits[c] == Suit.WIZARD else # wizard
			(self.cardNames[self.values[c] - 2] + self.toSuitString(self.suits[c])) # suit, value
			for c in cards]

	def toSuitString(self, suit):
		return self.suitNames[suit] if suit in self.suitNames.keys() else ''
