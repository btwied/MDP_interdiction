from numpy import array
from types import StringTypes

from CachedAttr import CachedAttr
from useful_functions import powerset

class State:
	"""Parent class for Outcome, Prereq, and Basis."""
	def __init__(self, pos, neg):
		"""
		pos: variables that are true/positive
		neg: variables that are false/negative

		The state space may have more variables than pos + neg; In this case,
		their values are unspecified, and the state may be viewed as a class
		of states consistent with what what is specified.
		"""
		self.pos = frozenset(pos)
		self.neg = frozenset(neg)

	def __eq__(self, other):
		if isinstance(other, State):
			return self.pos == other.pos and self.neg == other.neg
		return False

	def __ne__(self, other):
		if isinstance(other, State):
			return self.pos != other.pos or self.neg != other.neg
		return True

	def __lt__(self, other):
		return self.pos < other.pos and self.neg < other.neg

	def __le__(self, other):
		"""Useful for testing whether (other holds) ==> (state holds)"""
		return self.pos <= other.pos and self.neg <= other.neg

	def __gt__(self, other):
		return self.pos > other.pos and self.neg > other.neg

	def __ge__(self, other):
		return self.pos >= other.pos and self.neg >= other.neg

	def __sub__(self, var):
		return State(self.pos - {var}, self.neg - {var})

	def __contains__(self, item):
		return item in self.pos or item in self.neg
	
	def __iter__(self):
		for v in self.pos:
			yield v
		for v in self.neg:
			yield v

	def __len__(self):
		return self._len

	@CachedAttr
	def _len(self):
		return len(self.pos) + len(self.neg)

	def __hash__(self):
		return self._hash

	@CachedAttr
	def _hash(self):
		return hash((self.pos, self.neg))

	def __repr__(self):
		return self._repr

	@CachedAttr
	def _repr(self):
		return "+" + repr(sorted(self.pos)) + "-" + repr(sorted(self.neg))


class Outcome(State):
	"""
	Adds literals to or deletes literals from the state.
	
	Performing an action will result in some associated outcome's transition
	being applied to the state.
	"""
	def transition(self, state):
		"""
		Add the positive literals to the state and remove the negative ones.
		"""
		return State(self.pos | state.pos - state.neg, \
					self.neg | state.neg - state.pos)

	def changes(self, state):
		return self.transition(state) != state


class Basis(State):
	"""
	A specification of a basis function for linear value approximation.

	When a state is consistent with the basis, the basis function has value
	1. For example, if pos=(v2,v4) and neg=(v5,), then the function is as
	follows:
			{ 1 if s=..1.10.*
	f(s) =	{ 0 if s=..0....*
			{	or s=....0..*
			{	or s=.....1.*
	"""
	def backprojection(self, action):
		"""
		Construct g_i^a function.

		action:	A factored_MDP.Action object.

		return:	A dict mapping states, z, in the function's domain
				to g_i^a(z) values.
		"""
		must_start_pos = action.prereq.pos | (self.pos - action.can_make_pos)
		must_start_neg = action.prereq.neg | (self.neg - action.can_make_neg)
		must_end_pos = self.pos | action.must_make_pos
		must_end_neg = self.neg | action.must_make_neg

		g = {}
		if not (must_start_pos.isdisjoint(must_start_neg) and \
				must_end_pos.isdisjoint(must_end_neg)):
			return g

		fixed_at_start = must_start_pos | must_start_neg
		fixed_at_end = must_end_pos | must_end_neg
		free_at_start = fixed_at_end - fixed_at_start
		can_start_neg = must_start_neg | free_at_start

		for z in all_states(must_start_pos, must_start_neg, free_at_start):
			for outcome in action.outcomes:
				if self <= outcome.transition(z):
					g[z] = g.get(z,0) + action.outcome_probs[outcome]
		return g, domain


def all_states(pos=set(), neg=set(), free=set(), cls=State):
	"""
	Generate all states consistent with pos, neg, and free variable sets.
	"""
	for s in powerset(free):
		yield cls(pos | s, neg | (free - s))
