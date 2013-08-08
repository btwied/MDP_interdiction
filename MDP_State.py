from collections import defaultdict

from numpy import array

from CachedAttr import CachedAttr
from LazyCollection import LazyCollection
from useful_functions import powerset



class State(LazyCollection):
	"""Parent class for Outcome and Prereq."""
	def __init__(self, pos, neg, sort=False):
		"""
		pos: variables that are true/positive
		neg: variables that are false/negative

		The state space may have more variables than pos + neg; In this case,
		their values are unspecified, and the state may be viewed as a class
		of states consistent with what what is specified.
		"""
		self.pos = LazyCollection(pos, sort)
		self.neg = LazyCollection(neg, sort)
		self.data = (self.pos, self.neg)

	@CachedAttr
	def as_str(self):
		return "+" + repr(self.pos) + "-" + repr(self.neg)

	def __cmp__(self, other):
		if isinstance(other, State):
			return cmp(self.pos, other.pos) or cmp(self.neg, other.neg)
		return LazyCollection.__cmp__(self, other)


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
		pos_vars = set(state.pos)
		pos_vars.update(self.pos)
		pos_vars.difference_update(self.neg)
		neg_vars = set(state.neg)
		neg_vars.update(self.neg)
		neg_vars.difference_update(self.pos)
		return State(pos_vars, neg_vars)

	def changes(self, state):
		return self.transition(state) != state


class Prereq(State):
	"""
	Specifies prerequisites that must hold for an action to be available.
	"""
	def consistent(self, state):
		"""
		Tests whether a state is consistend with the prerequisite.

		A state is consistent with a prereq if all the positives are true
		and all the negatives are false.
		"""
		return all(v in state.pos for v in self.pos) and \
				all(v in state.neg for v in self.neg)


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
	def triggered_by(self, state):
		return self.pos.issubset(state.pos) and self.neg.issubset(state.neg)

	def backprojection(self, action):
		"""
		Construct g_i^a function.

		action:	A factored_MDP.Action object.

		return:	A dict mapping states, z, in the function's domain
				to g_i^a(z) values.
		"""
		must_start_pos = set(action.prereq.pos)
		must_start_neg = set(action.prereq.neg)
		must_end_pos = set(self.pos)
		must_end_neg = set(self.neg)

		must_start_pos.update(must_end_pos - action.can_make_pos)
		must_start_neg.update(must_end_neg - action.can_make_neg)
		must_end_pos.update(must_start_pos - action.can_make_neg)
		must_end_neg.update(must_start_neg - action.can_make_pos)
		must_end_pos.update(action.must_make_pos)
		must_end_neg.update(action.must_make_neg)

		g = defaultdict(lambda: 0)
		if not (must_start_pos.isdisjoint(must_start_neg) and \
				must_end_pos.isdisjoint(must_end_neg)):
			return g

		fixed_at_start = must_start_pos.union(must_start_neg)
		fixed_at_end = must_end_pos.union(must_end_neg)
		free_at_start = fixed_at_end - fixed_at_start
		can_start_neg = must_start_neg.union(free_at_start)

		for true_vars in powerset(free_at_start):
			z = State(must_start_pos.union(true_vars), \
							can_start_neg.difference(true_vars), True)
			for outcome in action.outcomes:
				if self.triggered_by(outcome.transition(z)):
					g[z] += action.outcome_probs[outcome]
		return g

