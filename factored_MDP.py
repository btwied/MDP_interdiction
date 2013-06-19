#!/usr/bin/python

from random import randrange, sample
from itertools import product

from numpy.random import uniform
from numpy import zeros, array

from warnings import warn
try:
	import gurobipy as G
except ImportError as e:
	warn("Gurobi is required to solve MDPs by linear programming.")

class MDP:
	"""
	Stores a restricted factored MDP and allows conversion to gurobi LP.

	A MDP models probabilistic agent actions whose outcomes depend only on 
	current state. The model also captures agent reward.

	A factored MDP has the following properties:
		- The state space is the cross product of the domains of a finite
			set of finite-domain variables.
		- The rewards and transition probabilities depend only on small-
			domain functions of state variables.

	This class imposes the following additional restrictions:
		- boolean variables
		- Actions have prerequisites, and are feasible iff those are met.
		- Actions induce distributions over fixed outcomes.
		- Outcomes specify some variables made true and some made false.
		- The only rewards other than fixed action costs are received upon 
			termination, and are additive across single-variable functions.
		- By default, the discount rate is 1.0, which means there is no 
			explicit discounting. This assumes that the sum of outcome 
			probabilities will be strictly less than 1 for all actions.
	"""
	def __init__(self, variables, initial, actions, true_rwds, false_rwds, \
				discount=1.):
		"""
		variables: a collection of variable names
		initial: the subset of variables that are initially True
		actions: a collection of Action objects.
		true_rwds: mapping {v:r}, where reward r is received if v=True
				upon termination.
		true_rwds: same, but v=False
		discount: 0 < discount <= 1
		"""
		self.variables = sorted(set(variables))
		self.variable_index = {v:i for i,v in enumerate(self.variables)}
		self.initial = zeros(len(self.variables), dtype=bool)
		self.initial[[self.variable_index[v] for v in initial]] = True
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

		This LP follows the standard construction given, for example, on
		p.25 of 'Competitive Markov Decision Processes' by Filar & Vrieze.

		The solution to this LP is the average value over all states. The
		the value an individual state can be extracted from the var.x of
		the state's lp variable. A list of these lp variables can be
		retreived using m.getVars().
		"""
		m = G.Model()
		states =  map(array, product([0,1], repeat=len(self.variables)))
		lp_vars = {}
		for s in states:
			s_name = self.lp_var_name(s)
			lp_vars[s_name] = m.addVar(name=s_name, lb=-float('inf'))
		m.update()
		m.setObjective(G.quicksum(m.getVars()) / len(states))
		m.update()
		for state,action in product(states, self.actions):
			if action.prereq.consistent(state, self.variable_index):
				const = action.stop_prob * self.terminal_reward(state)
				const -= action.cost
				expr = G.LinExpr(float(const))
				for outcome,prob in action.outcome_probs.items():
					expr += discount * prob * lp_vars[self.lp_var_name( \
							outcome.transition(state, self.variable_index))]
				m.addConstr(lp_vars[self.lp_var_name(state)] >= expr)
		m.update()
		self.exact_model = m
		return m

	def factored_LP(self, basis):
		"""
		Construct a factored LP to approximately solve the MDP with gurobi.

		This LP follows the construction given by Guestrin, et al. in
		'Efficient Solution Algorithms for Factored MDPs', JAIR 2003.

		basis: a collection of 'basis functions' each specifying some subset
				of the variables. As an example, if (v2,v4) is in the
				basis collection, then the following is a basis function:
						{ 1 if x=..1.1.*
				f(x) =	{ 0 if x=..0.1.*
						{	or x=..1.0.*
						{	or x=..0.0.*
				The constant function f(x)=1 will be added to the basis
				automatically to ensure LP feasibility. A good baseline
				basis to try is an indicator for each variable.
		"""
		basis = set(map(tuple, basis))
		basis = sorted(basis.union({()})) # add constant basis function
		m = G.Model()
		#TODO: implement this!
		raise NotImplementedError("TODO")
		self.factored_model = m
		return m

	def terminal_reward(self, state_vect):
		"""
		Calculates the reward if the MDP terminates in the given state.
		"""
		return float(self.true_rewards.dot(state_vect) + \
						self.false_rewards.dot(1-state_vect))

	def lp_var_name(self, state_vect):
		"""
		Gives the name of the LP variable used to represent the state's
		value in the exact_LP.
		"""
		return "V_" + "".join([var if val else "" for var, val in \
					zip(self.variables, state_vect)])


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

		A state is consistent with a prereq if all the positives are true
		and all the negatives are false.
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
	MDP action.

	Prerequisites specify the states in which the action is available.
	Actions encode the MDP's transition function through their distributions
	over outcomes. They also partially specify the reward function through
	their costs.
	"""
	def __init__(self, name, cost, prereq, outcome_dist):
		"""
		cost > 0
		outcome_dist: a mapping of Outcome objects to probabilities such
				that p > 0 and sum(p) <= 1
		"""
		self.name = name
		self.cost = cost
		self.prereq = prereq
		self.outcome_probs = outcome_dist
		self.stop_prob = 1. - sum(outcome_dist.values())


def random_MDP(min_vars=10, max_vars=10, min_acts=10, max_acts=10, \
				max_pos_prereqs=2, max_neg_prereqs=0, min_outs=20, \
				max_outs=20, min_outs_per_act=1, max_outs_per_act=3, \
				min_pos_vars_per_out=1, max_pos_vars_per_out=3, \
				min_neg_vars_per_out=0, max_neg_vars_per_out=0, \
				min_cost=0, max_cost=2, min_stop_prob=.001, max_stop_prob= \
				.2, true_rwds=3, false_rwds=1, min_true_rwd=-10, \
				max_true_rwd=10, min_false_rwd=-10, max_false_rwd=10, \
				stop_action=True, stop_cost=2):
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
		act_probs /= (act_probs.sum() / (1 - uniform(min_stop_prob, \
													max_stop_prob)))
		MDP_acts.append(Action("a"+str(i), uniform(min_cost, max_cost), \
						act_prereq, dict(zip(act_outs, act_probs))))

	if stop_action:
		MDP_acts.append(Action("stop", stop_cost, Prereq([],[]), {}))

	true_rwds = {v : uniform(min_true_rwd, max_true_rwd) for v in \
				sample(MDP_vars, true_rwds)}
	false_rwds = {v : uniform(min_false_rwd, max_false_rwd) for v in \
				sample(vars_set - set(true_rwds), false_rwds)}
	return MDP(MDP_vars, [], MDP_acts, true_rwds, false_rwds)


if __name__ == "__main__":
	mdp = random_MDP()
	lp = mdp.exact_LP()
	lp.optimize()
