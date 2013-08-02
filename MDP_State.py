from collections import defaultdict

from numpy import array

from CachedAttr import CachedAttr
from LazyCollection import LazyCollection
from useful_functions import powerset



class FullState:
	def __init__(self):
		raise NotImplementedError("TODO")


class PartialState(LazyCollection):
	"""Parent class for Outcome and Prereq."""
	def __init__(self, pos, neg, sort=False):
		self.pos = LazyCollection(pos, sort)
		self.neg = LazyCollection(neg, sort)
		self.data = (self.pos, self.neg)

	@CachedAttr
	def as_str(self):
		return "+" + repr(self.pos) + "-" + repr(self.neg)

	def __cmp__(self, other):
		if isinstance(other, PartialState):
			return cmp(self.pos, other.pos) or cmp(self.neg, other.neg)
		return LazyCollection.__cmp__(self, other)


class Outcome(PartialState):
	"""
	Adds literals to or deletes literals from the state.
	
	Performing an action will result in some associated outcome's transition
	being applied to the state.
	"""
	def transition(self, state, variables):
		"""
		Add the positive literals to the state and remove the negative ones.
		"""
		next_state = array(state)
		next_state[[variables.index(v) for v in self.pos]] = True
		next_state[[variables.index(v) for v in self.neg]] = False
		return tuple(next_state)

	def changes(self, state, variables):
		return self.transition(state, variables) != state

	def partial_transition(self, partial_state):
		"""
		Same as transition(), but operates on PartialState objects.
		"""
		pos = partial_state.pos.union(self.pos).difference(self.neg)
		neg = partial_state.neg.union(self.neg).difference(self.pos)
		return PartialState(pos, neg, True)


class Prereq(PartialState):
	"""
	Specifies prerequisites that must hold for an action to be available.
	"""
	def consistent(self, state, variables):
		"""
		Tests whether a state is consistend with the prerequisite.

		A state is consistent with a prereq if all the positives are true
		and all the negatives are false.
		"""
		for v in self.pos:
			if not state[variables.index(v)]:
				return False
		for v in self.neg:
			if state[variables.index(v)]:
				return False
		return True


class Basis(PartialState):
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
	def triggered_by(self, partial_state):
		return self.pos.issubset(partial_state.pos) and \
				self.neg.issubset(partial_state.neg)

	def backprojection(self, action):
		"""
		Construct g_i^a function.

		action:	A factored_MDP.Action object.

		return:	A dict mapping partial states, z, in the function's domain
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
			z = PartialState(must_start_pos.union(true_vars), \
							can_start_neg.difference(true_vars), True)
			for outcome in action.outcomes:
				if self.triggered_by(outcome.partial_transition(z)):
					g[z] += action.outcome_probs[outcome]
		return g

