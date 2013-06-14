from collections import namedtuple
from random import randrange, sample

from numpy.random import uniform
from numpy import zeros, array

from warnings import warn
try:
	import gurobipy
except ImportError as e:
	warn(e.message)

class MDP:
	"""
	Stores a restricted factored MDP and allows conversion to gurobi LP.

	Restrictions:
		- boolean variables
		- actions have prerequisites
		- actions induce distributions over fixed outcomes
		- an outcome can add some variables and/or remove others

	TODO: finish commenting this!
	"""
	def __init__(self, variables, initial, actions, rewards):
		"""
		variables: a collection of variable names
		initial: the subset of variables that are initially True
		actions: 
		rewards: 
		basis: 

		may also need to specify a basis
		"""
		self.variables = sorted(set(variables))
		self.variable_index = {v:i for i,v in enumerate(self.variables)}
		self.initial = zeros(len(self.variables), dtype=bool)
		self.initial[[self.variable_index[v] for v in initial]] = True
		#TODO: finish implementing this
		self.actions = actions
		self.rewards = rewards

	
	def to_LP(self, basis):
		"""
		Construct an approximate LP to solve the MDP using gurobipy.
		"""
		assert gurobipy
		#TODO: implement this


class PartialState:
	def __init__(self, pos, neg):
		self.pos = pos
		self.neg = neg
		self.tup = (tuple(self.pos), tuple(self.neg))

	def __hash__(self):
		return hash(self.tup)
	
	def __cmp__(self, other):
		try:
			return cmp(self.tup, other.tup)
		except AttributeError:
			return cmp(self.tup, other)


class Outcome(PartialState):
	"""
	Adds literals to or deletes literals from the state.
	
	Performing an action will result in some associated outcome's transition 
	being applied to the state.
	"""
	def transition(self, state, variable_index):
		"""
		Add the positive literals to the state and remove the negative ones.
		"""
		next_state = array(state)
		next_state[[variable_index[v] for v in self.adds]] = True
		next_state[[variable_index[v] for v in self.dels]] = False
		return next_state


class Prereq(PartialState):
	"""
	Specifies prerequisites that must hold for an action to be available.
	"""
	def consistent(self, state, variable_index):
		"""
		Tests whether a state is consistend with the prerequisite.

		A state is consistent with a prereq if all the positives are true and
		all the negatives are false.
		"""
		for v in self.pos:
			if not state[variable_index[v]]:
				return False
		for v in self.neg:
			if state[variable_index[v]]:
				return False
		return True


class Action:
	"""
	TODO: comment this!
	"""
	def __init__(self, name, cost, prereqs, outcome_dist):
		"""
		prereqs: a mapping of variables to values, indicating what must be true
				or false if the action is to be performed.
		outcome_dist: a mapping of outcomes to probabilities such that
				p > 0 and sum(p) < 1
		"""
		self.name = name
		self.cost = cost
		self.prereqs = prereqs
		self.outcomes = outcomes

	#TODO: implement this!


# prereqs should be a dict mapping vars to vals
# outcomes should be a dict mapping tuples of vars to probs
# 0 < probs; sum(probs) < 1
Action = namedtuple("Action", "name prereqs outcomes cost")


def random_MDP(min_vars=10, max_vars=10, min_acts=10, max_acts=10, \
				max_prereqs=2, min_outs=10, max_outs=10, max_outs_per_act=3, \
				max_vars_per_out=3, min_cost=0, max_cost=10, min_discount=.8, \
				max_discount=.999, nonzero_rwd_prob=0.3, min_rwd=-10, \
				max_rwd=10):
	"""Creates an MDP for testing."""
	MDP_vars = ['v'+str(i) for i in range(randrange(min_vars, max_vars+1))]
	MDP_outs = [tuple(sample(MDP_vars, randrange(1, max_vars_per_out+1))) for \
				i in range(randrange(min_outs, max_outs+1))]

	MDP_acts = []
	for i in range(randrange(min_acts, max_acts+1)):
		act_prereqs = {v:True for v in sample(MDP_vars, \
						randrange(max_prereqs+1))}
		act_outs =  sample(MDP_outs, randrange(1, max_outs_per_act+1))
		act_probs = uniform(0,1,10)
		act_probs /= (act_probs.sum() / uniform(min_discount, max_discount))
		MDP_acts.append(Action("a"+str(i), act_prereqs, dict(zip(act_outs, \
						act_probs)), uniform(min_cost, max_cost)))

	MDP_rwds = {v : uniform(min_rwd, max_rwd) for v in filter(lambda x: \
				uniform(0,1) < nonzero_rwd_prob, MDP_vars)}
	return MDP(MDP_vars, [], MDP_acts, MDP_rwds)


if __name__ == "__main__":
	m = random_MDP()
	m.to_LP([()])
