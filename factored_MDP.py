#!/usr/bin/python

from random import randrange, sample
from itertools import product

from numpy.random import uniform
from numpy import zeros, array


# The following simply imports Gurobi and looks for Gurobi parameters
from os import listdir
import warnings
warnings.formatwarning = lambda msg, *args: "warning: " + str(msg) + "\n"
try:
	import gurobipy as G
	param_file = filter(lambda fn: fn.endswith('.prm'), listdir('.'))[0]
	G.readParams(param_file)
except ImportError:
	warnings.warn("Gurobi is required to solve MDPs by linear programming.")
except:
	warnings.warn("Failed to read parameter file.")



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
			explicit discounting. This assumes that a terminal state will
			be reached with probability 1. One way this can happen is if
			the sum of outcome probabilities is strictly less than 1 for
			all actions.
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
		init_state = zeros(len(self.variables), dtype=bool)
		init_state[[self.variable_index[v] for v in initial]] = True
		self.initial = tuple(init_state)
		self.actions = actions
		self.true_rewards = array([true_rwds[v] if v in true_rwds else 0 \
								for v in self.variables])
		self.false_rewards = array([false_rwds[v] if v in false_rwds \
								else 0 for v in self.variables])
		self.discount = discount

	def __repr__(self):
		s = "MDP: "
		s += str(len(self.variables)) + " variables, "
		s += str(len(self.actions)) + " actions."
		return s
	
	def terminal_reward(self, state):
		"""
		Calculates the reward if the MDP terminates in the given state.
		"""
		state_vect = array(state)
		return float(self.true_rewards.dot(state_vect) + \
						self.false_rewards.dot(1-state_vect))

	def full_states(self):
		"""
		Initializes and returns the (exponentially large) set of states.
		"""
		try:
			return self.states
		except AttributeError:
			self.states = map(tuple, product([0,1], repeat= \
										len(self.variables)))
			return self.states

	def reachable_states(self):
		"""
		Searches for all states reachable from the initial state.

		This set is likely to be exponentially large.
		"""
		try:
			return self.reachable
		except AttributeError:
			pass
		unvisited = {self.initial}
		visited = set()
		while unvisited:
			state = unvisited.pop()
			visited.add(state)
			for action in self.actions:
				if action.prereq.consistent(state, self.variable_index):
					for outcome in action.outcomes:
						next_state = outcome.transition(state, \
										self.variable_index)
						if next_state not in visited:
							unvisited.add(next_state)
		self.reachable = sorted(visited)
		return self.reachable

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
		m = G.Model() # Throws a NameError if gurobipy isn't installed
		states = self.reachable_states()
		self.lp_state_vars = {}
		for s in states:
			s_name = "V_" + self.lp_var_name(s)
			self.lp_state_vars[s] = m.addVar(name=s_name, lb=-float("inf"))
		m.update()
		m.setObjective(G.quicksum(m.getVars()) / len(states))
		m.update()
		useless_actions = 0 #TODO: remove
		for state,action in product(states, self.actions):
			if action.prereq.consistent(state, self.variable_index):
				const = action.stop_prob * self.terminal_reward(state)
				const -= action.cost
				expr = G.LinExpr(float(const))

				#TODO: remove
				if all(o.transition(state, self.variable_index) == state for o in action.outcomes):
					useless_actions += 1

				for outcome,prob in action.outcome_probs.items():
					lp_var = self.lp_state_vars[outcome.transition(state, \
										self.variable_index)]
					expr += self.discount * prob * lp_var
				m.addConstr(self.lp_state_vars[state] >= expr)
		m.update()
		print useless_actions, "unnecessary constraints" #TODO: remove
		return m

	def factored_LP(self, basis):
		"""
		Construct a factored LP to approximately solve the MDP with gurobi.

		This LP follows the construction given by Guestrin, et al. in
		'Efficient Solution Algorithms for Factored MDPs', JAIR 2003.

		basis: a collection of 'basis functions' each specifying two tuples
				of variables. The first tuple specifies positive literals
				and the second specifies negative literals; when all these
				literals have their specified values f=1. As an example, if
				((v2,v4),(v5)) is in the 'basis' collection, then the
				following is a basis function:
						{ 1 if x=..1.10.*
				f(x) =	{ 0 if x=..0....*
						{	or x=....0..*
						{	or x=.....1.*
				The constant function f(x)=1 will be added to the basis
				automatically to ensure LP feasibility. A good baseline
				basis to try is an indicator for each variable.
		"""
		m = G.Model() # Throws a NameError if gurobipy isn't installed
		basis = sorted(set(basis).union({((),())}))

		self.lp_basis_vars = {}
		for b in basis:
			b_name = "W_+" + self.lp_var_name(b[0])
			b_name += "_-" + self.lp_var_name(b[1])
			self.lp_basis_vars[b] = m.addVar(name=b_name, lb=-float("inf"))
		
		#TODO: finish implementing this!
		raise NotImplementedError("TODO")

		return m

	def default_basis(self):
		"""
		Creates a basis function for each literal, prereq, and outcome.
		"""
		basis = [((v),()) for v in self.variables]
		for a in self.actions:
			basis.append(a.prereq.tup)
			for o in a.outcomes:
				basis.append(o.tup)
		return basis

	def lp_var_name(self, state_vect):
		"""
		Gives the name of the LP variable used to represent the state's
		value in the exact_LP.
		"""
		return "".join([var if val else "" for var, val in \
					zip(self.variables, state_vect)])

	def action_value(self, state, action, values):
		"""
		Expected next-state value of perfrming action in state according
		to values.
		"""
		value = -action.cost
		for outcome,prob in action.outcome_probs.items():
			next_state = outcome.transition(state, self.variable_index)
			value += self.discount * prob * values[next_state]
		value += action.stop_prob * self.terminal_reward(state)
		return value

	def state_values(self, policy, values, iters=1000, cnvrg_thresh=1e-6):
		"""
		Expected value estimate for each state when following policy.

		An accurate estimate requires convergence, which may require a
		a large number of iterations. For modified policy iteration, iters
		can be set relatively low to return before convergence.
		"""
		states = self.reachable_states()
		for _i in range(iters):
			new_values = {}
			for state in states:
				action = policy[state]
				new_values[state] = self.action_value(state, action, values)
			if converged(values, new_values, cnvrg_thresh):
				break
			values = new_values
		return new_values

	def greedy_policy(self, values):
		"""
		State-action map that is one-step optimal according to values.
		"""
		states = self.reachable_states()
		new_policy = {}
		for state in states:
			best_action = None
			best_value = -float("inf")
			for action in self.actions:
				if action.prereq.consistent(state, self.variable_index):
					act_val = self.action_value(state, action, values) 
					if act_val > best_value:
						best_value = act_val
						best_action = action
			new_policy[state] = best_action
		return new_policy

	def policy_iteration(self, policy_iters=1000, value_iters=100, \
						cnvrg_thresh=1e-6):
		"""
		Computes optimal policy and value functions for the MDP.

		This algorithm represents the full state space and therefore 
		requires time and space exponential in the size of the factored MDP.

		If policy_iters is reached, the algorithm has not converged and the
		results may be sub-optimal. For true policy iteration, value_iters
		should be set very high; for modified policy iteration, it can be
		relativley small.
		"""
		states = self.reachable_states()
		values = {s:0 for s in states}
		for _i in range(policy_iters):
			old_values = values
			policy = self.greedy_policy(values)
			values = self.state_values(policy, values, value_iters, \
										cnvrg_thresh)
			if converged(old_values, values, cnvrg_thresh):
				values_changed = False
		return policy, values


def converged(old_vals, new_vals, thresh=1e-6):
	return all((abs(new_vals[s] - old_vals[s]) < thresh for s in new_vals))


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
		return tuple(next_state)


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
		self.outcomes = sorted(outcome_dist.keys())
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
	mdp = random_MDP(min_vars=20, max_vars=20, min_acts=20, max_acts=20, \
					min_outs=50, max_outs=50)
	print mdp
	print 2**len(mdp.variables), "total states"
	print len(mdp.reachable_states()), "reachable states"
	try:
		lp = mdp.exact_LP()
		print "LP has", len(lp.getConstrs()), "constraints"
		lp.optimize()
		print "linear programming value estimate:", \
				mdp.lp_state_vars[mdp.initial].x
	except NameError:
		pass
#	policy, values = mdp.policy_iteration()
#	print "policy iteration value estimate:", values[mdp.initial]
