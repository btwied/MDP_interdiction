from random import randrange, sample
from itertools import product
from collections import defaultdict

from numpy.random import uniform

from CachedAttr import CachedAttr
from MDP_State import Basis, Outcome, State, all_states
from MDP_Action import Action


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
		- Discount rate = 1.0, which means there is no explicit discounting.
			This assumes that a terminal state will be reached with
			probability 1. One way this can happen is if the sum of outcome
			probabilities is strictly less than 1 for all actions.
		- At any time the agent can stop and receive reward for the
			current state.
	"""
	def __init__(self, variables, initial, actions, true_rwds, false_rwds):
		"""
		variables: a collection of variable names
		initial: the subset of variables that are initially True
		actions: a collection of Action objects; none may be named STOP.
		true_rwds: mapping {v:r}, where reward r is received if v=True
				upon termination.
		false_rwds: same, but v=False
		"""
		self.variables = frozenset(variables)
		self.initial = State(initial, self.variables.difference(initial))
		self.actions = actions
		self.true_rewards = true_rwds
		self.false_rewards = false_rwds
		self.state_rewards = {} #stores terminal rewards once computed

	def __repr__(self):
		s = "MDP: "
		s += str(len(self.variables)) + " variables, "
		s += str(len(self.actions)) + " actions."
		return s
	
	def terminal_reward(self, state):
		"""
		The reward if the MDP terminates in the given state.
		"""
		if state in self.state_rewards:
			return self.state_rewards[state]
		self.state_rewards[state] = sum(self.true_rewards.get(v, 0.) \
				for v in state.pos) + sum(self.false_rewards.get(v, 0.) \
				for v in state.neg)
		return self.state_rewards[state]

	@CachedAttr
	def full_states(self):
		"""The full (exponentially large) set of states."""
		return all_states(free=self.variables)

	@CachedAttr
	def reachable_states(self):
		"""The set of states reachable from initial (probably exponential)."""
		unvisited = {self.initial}
		visited = set()
		reachable = {self.initial:set()}
		while unvisited:
			state = unvisited.pop()
			visited.add(state)
			for action in self.actions:
				if action.prereq <= state:
					for outcome in action.outcomes:
						next_state = outcome.transition(state)
						parents = reachable.get(next_state,set())
						parents.add((state,action))
						reachable[next_state] = parents
						if next_state not in visited:
							unvisited.add(next_state)
		return reachable
		

def random_MDP(min_vars=10, max_vars=10, min_acts=10, max_acts=10, \
				max_pos_prereqs=2, max_neg_prereqs=0, min_outs=20, \
				max_outs=20, min_outs_per_act=1, max_outs_per_act=3, \
				min_pos_vars_per_out=1, max_pos_vars_per_out=3, \
				min_neg_vars_per_out=0, max_neg_vars_per_out=0, \
				min_cost=0, max_cost=2, min_stop_prob=.001, max_stop_prob= \
				.2, true_rwds=3, false_rwds=1, min_true_rwd=-10, \
				max_true_rwd=10, min_false_rwd=-10, max_false_rwd=10):
	"""Creates an MDP for testing."""
	MDP_vars = ["l"+str(i) for i in range(randrange(min_vars, max_vars+1))]
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
		act_prereq = State(pos_prereqs, neg_prereqs)

		act_outs =  sample(MDP_outs, randrange(min_outs_per_act, \
							max_outs_per_act + 1))
		act_probs = uniform(0,1,len(act_outs))
		act_probs /= (act_probs.sum() / (1 - uniform(min_stop_prob, \
													max_stop_prob)))
		MDP_acts.append(Action("a"+str(i), uniform(min_cost, max_cost), \
						act_prereq, dict(zip(act_outs, act_probs))))

	true_rwds = {v : uniform(min_true_rwd, max_true_rwd) for v in \
				sample(MDP_vars, true_rwds)}
	false_rwds = {v : uniform(min_false_rwd, max_false_rwd) for v in \
				sample(vars_set - set(true_rwds), false_rwds)}
	return MDP(MDP_vars, [], MDP_acts, true_rwds, false_rwds)

