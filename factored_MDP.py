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

	TODO: comment this!
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
		self.actions = actions
		self.rewards = rewards

	
	def to_LP(self, basis):
		"""
		Construct an approximate LP to solve the MDP using gurobipy.
		"""
		assert gurobipy
		#TODO: implement this


class Outcome:
	"""
	TODO: comment this!
	"""
	def __init__(self, adds, dels):
		self.adds = adds
		self.dels = dels

	def transition(self, state, variable_index):
		next_state = array(state)
		next_state[[variable_index[v] for v in self.adds]] = True
		next_state[[variable_index[v] for v in self.dels]] = False
		return next_state

	def set_vectors(self, variable_index);
		self.add_vect = np.zeros(len(variable_index))
		self.del_vect = np.zeros(len(variable_index))
		self.add_vect[[variable_index[v] for v in self.adds]] = True
		self.del_vect[[variable_index[v] for v in self.dels]] = True
	
	def fast_trans(self, state):
		"""
		Performs state transition in place, without generating a new array.
		
		set_vectors() must have been performed first.
		"""
		state[self.add_vect] = True
		state[self.del_vect] = False


class Action:
	"""
	TODO: comment this!
	"""
	def __init__(self, name, prereqs, outcomes, cost):
		#TODO: implement this!
		self.name = name
		self.prereqs = prereqs
		self.outcomes = outcomes
		self.cost = cost


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
