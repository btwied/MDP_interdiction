#!/usr/bin/python

from random import randrange, sample
from itertools import product

from numpy.random import uniform
from numpy import zeros, array

from warnings import warn
try:
	import gurobipy as G
except ImportError as e:
	warn(e.message)

class MDP:
	"""
	Stores a restricted factored MDP and allows conversion to gurobi LP.

	Restrictions:
		- boolean variables
		- actions have prerequisites
		- actions induce distributions over fixed outcomes
		- only positive variables can yield nonzero terminal reward

	TODO: finish commenting this!
	"""
	def __init__(self, variables, initial, actions, true_rwds, false_rwds):
		"""
		variables: a collection of variable names
		initial: the subset of variables that are initially True
		actions: 
		rewards: mapping of variables to rewards
		
		TODO: finish commenting this!
		"""
		self.variables = sorted(set(variables))
		self.variable_index = {v:i for i,v in enumerate(self.variables)}
		self.initial = zeros(len(self.variables), dtype=bool)
		self.initial[[self.variable_index[v] for v in initial]] = True
		
		#TODO: finish implementing this
		self.actions = actions
		self.true_rewards = array([true_rwds[v] if v in true_rwds else 0 \
								for v in self.variables])
		self.false_rewards = array([false_rwds[v] if v in false_rwds \
								else 0 for v in self.variables])
		

	def __repr__(self):
		s = "MDP: "
		s += str(len(self.variables)) + " variables, "
		s += str(len(self.actions)) + " actions."
		return s

	def exact_LP(self):
		"""
		Construct an exponentially large LP to solve the MDP with gurobi.
		"""
		m = new_gurobi_model()
		states =  map(array, product([0,1], repeat=len(self.variables)))
		values = {}
		for s in states:
			s_name = self.state_name(s)
			values[s_name] = m.addVar(name=s_name, lb=-float('inf'))
		m.update()
		m.setObjective(G.quicksum(m.getVars()))
		m.update()
		for state,action in product(states, self.actions):
			if action.prereq.consistent(state, self.variable_index):
				const = action.stop_prob * self.terminal_reward(state)
				const -= action.cost
				expr = G.LinExpr(float(const))
				for outcome,prob in action.outcome_probs.items():
					expr += prob * values[self.state_name( \
							outcome.transition(state, self.variable_index))]
				m.addConstr(values[self.state_name(state)] >= expr)
		m.update()
		return m

	def approx_LP(self, basis):
		"""
		Construct a factored LP to approximately solve the MDP with gurobi.
		"""
		m = new_gurobi_model()
		raise NotImplementedError("TODO")
		#TODO: implement this!

	def terminal_reward(self, state_vect):
		"""
		TODO: comment this
		"""
		return float(self.true_rewards.dot(state_vect) + \
						self.false_rewards.dot(1-state_vect))

	def state_name(self, state_vect):
		"""
		TODO: comment this
		"""
		return "V_" + "".join([var if val else "" for var, val in \
					zip(self.variables, state_vect)])


def new_gurobi_model():
		try:
			return G.Model()
		except NameError:
			raise ImportError("gurobipy required")


class PartialState:
	"""Parent class for Outcome and Prereq."""
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
		next_state[[variable_index[v] for v in self.pos]] = True
		next_state[[variable_index[v] for v in self.neg]] = False
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
	def __init__(self, name, cost, prereq, outcome_dist):
		"""
		outcome_dist: a mapping of outcomes to probabilities such that
				p > 0 and sum(p) < 1

		TODO: finish commenting this!
		"""
		self.name = name
		self.cost = cost
		self.prereq = prereq
		self.outcome_probs = outcome_dist
		self.stop_prob = 1. - sum(outcome_dist.values())

	#TODO: implement this!



def random_MDP(min_vars=10, max_vars=10, min_acts=10, max_acts=10, \
				max_pos_prereqs=2, max_neg_prereqs=0, min_outs=20, \
				max_outs=20, min_outs_per_act=1, max_outs_per_act=3, \
				min_pos_vars_per_out=1, max_pos_vars_per_out=3, \
				min_neg_vars_per_out=0, max_neg_vars_per_out=0, \
				min_cost=0, max_cost=2, min_cont_prob=.8, max_cont_prob= \
				.999, true_rwds=3, false_rwds=1, min_true_rwd=0, \
				max_true_rwd=10, min_false_rwd=-10, max_false_rwd=0):
	"""Creates an MDP for testing."""
	MDP_vars = ['l'+str(i) for i in range(randrange(min_vars, max_vars+1))]
	vars_set = set(MDP_vars)
	
	MDP_outs = []
	for i in range(randrange(min_outs, max_outs+1)):
		pos_outs = sample(MDP_vars, randrange(min_pos_vars_per_out, 
							max_pos_vars_per_out + 1))
		neg_outs = sample(vars_set - set(pos_outs), randrange( \
							max_neg_vars_per_out + 1))
		MDP_outs.append(Outcome(pos_outs, neg_outs))

	MDP_acts = []
	for i in range(randrange(min_acts, max_acts + 1)):
		pos_prereqs = sample(MDP_vars, randrange(max_pos_prereqs + 1))
		neg_prereqs = sample(vars_set - set(pos_prereqs), \
							randrange(max_neg_prereqs + 1))
		act_prereq = Prereq(pos_prereqs, neg_prereqs)

		act_outs =  sample(MDP_outs, randrange(min_outs_per_act, \
							max_outs_per_act + 1))
		act_probs = uniform(0,1,len(act_outs))
		act_probs /= (act_probs.sum() / uniform(min_cont_prob, max_cont_prob))
		MDP_acts.append(Action("a"+str(i), uniform(min_cost, max_cost), \
						act_prereq, dict(zip(act_outs, act_probs))))

	MDP_acts.append(Action("stop", 1, Prereq([],[]), {}))

	true_rwds = {v : uniform(min_true_rwd, max_true_rwd) for v in \
				sample(MDP_vars, true_rwds)}
	false_rwds = {v : uniform(min_false_rwd, max_false_rwd) for v in \
				sample(vars_set - set(true_rwds), false_rwds)}
	return MDP(MDP_vars, [], MDP_acts, true_rwds, false_rwds)


if __name__ == "__main__":
	mdp = random_MDP()
	lp = mdp.exact_LP()
	lp.optimize()
