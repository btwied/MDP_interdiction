#!/usr/bin/python

from argparse import ArgumentParser
from random import randrange, sample
from itertools import product, chain, combinations

from numpy.random import uniform
from numpy import zeros, array

from CachedAttr import LazyCollection, CachedAttr

# import Gurobi but don't crash if it wasn't loaded
import warnings
warnings.formatwarning = lambda msg, *args: "warning: " + str(msg) + "\n"
try:
	import gurobipy as G
except ImportError:
	warnings.warn("Gurobi is required to solve MDPs by linear programming.")



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
		true_rwds: same, but v=False
		"""
		self.variables = LazyCollection(variables, sort=True)
		init_state = zeros(len(self.variables), dtype=bool)
		init_state[[self.variables.index(v) for v in initial]] = True
		self.initial = tuple(init_state)
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
		self.reachable = {self.initial:set()}
		while unvisited:
			state = unvisited.pop()
			visited.add(state)
			for action in self.actions:
				if action.prereq.consistent(state, self.variables):
					for outcome in action.outcomes:
						next_state = outcome.transition(state, \
										self.variables)
						parents = self.reachable.get(next_state,set())
						parents.add((state,action))
						self.reachable[next_state] = parents
						if next_state not in visited:
							unvisited.add(next_state)
		return self.reachable

	def exact_primal_LP(self):
		"""
		Construct an exponentially large LP to solve the MDP with Gurobi.

		This LP follows the standard construction given, for example, on
		p.25 of 'Competitive Markov Decision Processes' by Filar & Vrieze.

		The solution to this LP is the value of the initial state. The
		the value any other state can be extracted from the var.x of
		the state's lp variable. A list of these lp variables can be
		retreived using lp.getVars().
		"""
		lp = G.Model() # Throws a NameError if gurobipy wasn't loaded
		states = self.reachable_states()
		self.primal_state_vars = {}

		# add a variable to the LP to represent the value of each state
		for s in states:
			self.primal_state_vars[s] = lp.addVar(name=self.state_name(s), \
												lb=-float("inf"))
		lp.update()
		# since we only care about reachable states, this suffices:
		lp.setObjective(self.primal_state_vars[self.initial])
#		lp.setObjective(G.quicksum(lp.getVars()))

		# can always cash out
		for s,v in self.primal_state_vars.items():
			lp.addConstr(v >= self.terminal_reward(s))

		# backpropagation
		for state,action in product(states, self.actions):
			if action.prereq.consistent(state, self.variables) and \
					action.can_change(state, self.variables):
				const = action.stop_prob * self.terminal_reward(state)
				const -= action.cost
				expr = G.LinExpr(float(const))
				for outcome,prob in action.outcome_probs.items():
					lp_var = self.primal_state_vars[outcome.transition( \
									state, self.variables)]
					expr += prob * lp_var
				lp.addConstr(self.primal_state_vars[state] >= expr)
		lp.update()
		return lp

	def factored_primal_LP(self, basis):
		"""
		Construct a factored LP to approximately solve the MDP with gurobi.

		This LP follows the construction given by Guestrin, et al. in
		'Efficient Solution Algorithms for Factored MDPs', JAIR 2003.

		The constant function f(x)=1 will be added to the basis
		automatically to ensure LP feasibility. Using default_basis
		is probably a good place to start.
		"""
		lp = G.Model() # Throws a NameError if gurobipy wasn't loaded
		self.basis = sorted(set(basis).union({PartialState((),())}))

		self.lp_basis_vars = {}
		for b in self.basis:
			self.lp_basis_vars[b] = lp.addVar(name=str(b), lb=-float("inf"))
		lp.update()

		# set objective
		lp.setObjective()

		# set constraints
		
		#TODO: finish implementing this!
		raise NotImplementedError("TODO")
		return lp

	def exact_dual_LP(self):
		"""
		Construct an exponentially large LP to solve the MDP with Gurobi.

		This LP follows the standard construction given, for example, on
		p.25 of 'Competitive Markov Decision Processes' by Filar & Vrieze.

		The solution to this LP is the value of the initial state. After
		optimize() has been called, the variables of the LP indicate the
		optimal policy as follows: if the variable v has v.name=s_a, then
		action a is optimal in state s iff v.x > 0.
		"""
		lp = G.Model() # Throws a NameError if gurobipy wasn't loaded
		states = self.reachable_states()
		self.dual_sa_vars = G.tuplelist()

		for s in states:
			n = self.state_name(s)
			self.dual_sa_vars.append((s, "STOP", lp.addVar(name=n+"_STOP", \
					lb=0)))
			for a in self.actions:
				if a.prereq.consistent(s, self.variables):

					self.dual_sa_vars.append((s, a, lp.addVar(name=n+"_"+\
							a.name, lb=0)))
		lp.update()

		# set objective
		obj = G.LinExpr()
		for s,a,var in self.dual_sa_vars:
			rew = self.terminal_reward(s)
			if a == "STOP":
				obj += rew * var
			else:
				obj += (a.stop_prob * rew - a.cost) * var
		lp.setObjective(obj, G.GRB.MAXIMIZE)

		# set constraints
		for s in states:
			constr = G.quicksum([v for _,a,v in \
						self.dual_sa_vars.select(s)])
			for parent,action in self.reachable[s]:
				prob = action.trans_prob(parent, s, self.variables)
				var = self.dual_sa_vars.select(parent,action)[0][2]
				constr -= prob * var
			if s == self.initial:
				lp.addConstr(constr, G.GRB.EQUAL, G.LinExpr(1))
			else:
				lp.addConstr(constr, G.GRB.EQUAL, G.LinExpr(0))
		lp.update()
		return lp

	def factored_dual_LP(self, basis):
		"""
		Construct a factored LP to approximately solve the MDP with gurobi.

		This LP is the dual to construction given by Guestrin, et al. in
		'Efficient Solution Algorithms for Factored MDPs', JAIR 2003.
		"""
		raise NotImplementedError("TODO")

	def default_basis(self):
		"""
		Creates a basis function for each literal, prereq, and outcome.
		"""
		basis = [PartialState((v),()) for v in self.variables]
		for a in self.actions:
			basis.append(a.prereq)
			basis.extend(a.outcomes)
		return basis

	def state_name(self, state_vect):
		"""
		Gives the name of the LP variable used to represent the state's
		value in the exact_LPs.
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
			next_state = outcome.transition(state, self.variables)
			value += prob * values[next_state]
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
				if action == None:
					new_values[state] = self.terminal_reward(state)
				else:
					new_values[state] = self.action_value(state, action, \
															values)
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
			best_value = self.terminal_reward(state)
			for action in self.actions:
				if action.prereq.consistent(state, self.variables):
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


def powerset(s):
	return chain.from_iterable(combinations(s,i) for i in range(len(s)+1))


class PartialState(LazyCollection):
	"""Parent class for Outcome and Prereq."""
	def __init__(self, pos, neg):
		self.pos = LazyCollection(pos)
		self.neg = LazyCollection(neg)
		self.data = (self.pos, self.neg)

	@CachedAttr
	def as_str(self):
		return "+" + repr(self.pos) + "_-" + repr(self.neg)

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


class Prereq(PartialState):
	"""
	Specifies prerequisites that must hold for an action to be available.
	"""
	def __init__(self, *args):
		PartialState.__init__(self, *args)

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


class Basis(Prereq):
	"""
	A specification of a basis function for linear value approximation.

	Basis inherits from Prereq so that it can use consistent(), not because
	of any conceptual dependence. When a state is consistent with the
	basis, the basis function has value 1. For example, if pos=(v2,v4) and 
	neg=(v5,), then the function is as follows:
			{ 1 if s=..1.10.*
	f(s) =	{ 0 if s=..0....*
			{	or s=....0..*
			{	or s=.....1.*
	"""


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
	
	def trans_prob(self, pre_state, post_state, variables):
		prob = 0
		for o,p in self.outcome_probs.items():
			if o.transition(pre_state, variables) == post_state:
				prob += p
		return prob

	def can_change(self, state, variables):
		return any(o.changes(state, variables) for o in self.outcomes)

	def __repr__(self):
		return "MDP action: " + self.name

	def __hash__(self):
		return hash(self.name)

	def __cmp__(self, other):
		try:
			return cmp(self.name, other.name)
		except AttributeError:
			return cmp(self.name, other)


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
		act_prereq = Prereq(pos_prereqs, neg_prereqs)

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


def parse_args():
	parser = ArgumentParser ()
	parser.add_argument("-prm", type=str, default="", help="Optional "+\
						"Gurobi parameter file to import.")
	parser.add_argument("--exact_primal", action="store_true", help="Set "+\
						"this to compute values via linear programming.")
	parser.add_argument("--factored_primal", action="store_true", help= \
						"Set this compute approximate values via "+\
						"factored-MDP linear programming.")
	parser.add_argument("--exact_dual", action="store_true", help="Set "+\
						"this to compute an optimal policy via dual " +\
						"linear programming.")
	parser.add_argument("--factored_dual", action="store_true", help= \
						"Set this compute an approximately optimal " +\
						"policy via factored-MDP dual linear programming.")
	parser.add_argument("--policy_iter", action="store_true", help="Set"+\
						"this to solve the MDP via policy iteration.")
	parser.add_argument("-num_vars", type=int, default=0, help="Number" +\
						"of literals in the random MDP. If not set, the" +\
						"size is set so that chosen solution algorithms "+\
						"will take a reasonable amount of time.")
	return parser.parse_args()


def main(args):
	if args.num_vars > 0:
		num_vars = args.num_vars
	elif args.policy_iter:
		num_vars = 8
	elif args.exact_primal or args.exact_dual:
		num_vars = 16
	else:
		num_vars = 32
	mdp = random_MDP(min_vars=num_vars, max_vars=num_vars, min_acts= \
			num_vars, max_acts=num_vars, min_outs=2*num_vars, max_outs= \
			2*num_vars)

	if args.prm != "":
		G.readParams(args.prm)
	
	print mdp
	print "2^" + str(num_vars), "=",  2**num_vars, \
			"total states"
	if args.policy_iter or args.exact_primal or args.exact_dual:
		print len(mdp.reachable_states()), "reachable states"

	if args.exact_primal:
		lp = mdp.exact_primal_LP()
		print len(lp.getConstrs()), "primal LP constraints"
		lp.optimize()
		print "excact primal linear programming initial state value:", \
				mdp.primal_state_vars[mdp.initial].x

	if args.factored_primal:
		lp = mdp.factored_primalLP()
		lp.optimize()
		print "factored primal linear programming approximate initial "+\
				"state value:", mdp.lp_basis_vars[mdp.initial].x

	if args.exact_dual:
		lp = mdp.exact_dual_LP()
		lp.optimize()
		print "excact dual linear programming initial state value:", \
				lp.objVal
		try:
			print "excact dual linear programming initial state policy: "+\
					str(filter(lambda sav: sav[2].x > 0, \
					mdp.dual_sa_vars.select(mdp.initial))[0][1].name)
		except AttributeError:
			print "excact dual linear programming initial state policy: "+\
					"STOP"

	if args.factored_dual:
		lp = mdp.factored_dual_LP()
		lp.optimize()
		print "factored primal linear programming approximate initial "+\
				"state value:", mdp.lp_basis_vars[mdp.initial].x

	if args.policy_iter:
		policy, values = mdp.policy_iteration()
		print "policy iteration initial state value:", values[mdp.initial]
		try:
			print "policy iteration initial state policy: " + \
					str(policy[mdp.initial].name)
		except AttributeError:
			print "policy iteration initial state policy: STOP"


if __name__ == "__main__":
	args = parse_args()
	main(args)
