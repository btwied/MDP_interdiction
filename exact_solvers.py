from itertools import product

from useful_functions import converged

# import Gurobi but don't crash if it wasn't loaded
import warnings
warnings.formatwarning = lambda msg, *args: "warning: " + str(msg) + "\n"
try:
	import gurobipy as G
except ImportError:
	warnings.warn("Gurobi is required to solve MDPs by linear programming.")



def exact_primal_LP(mdp):
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
	state_vars = {}

	# add a variable to the LP to represent the value of each state
	for s in mdp.reachable_states:
		state_vars[s] = lp.addVar(name=str(s), lb=-float("inf"))
	lp.update()
	# objective is the value of the initial state
	lp.setObjective(state_vars[mdp.initial])

	# can always cash out
	for s,v in state_vars.items():
		lp.addConstr(v >= mdp.terminal_reward(s))

	# backpropagation
	for state,action in product(mdp.reachable_states, mdp.actions):
		if action.prereq <= state and action.can_change(state, mdp.variables):
			const = action.stop_prob * mdp.terminal_reward(state)
			const -= action.cost
			expr = G.LinExpr(float(const))
			for out,prob in action.outcome_probs.items():
				lp_var = state_vars[out.transition(state)]
				expr += prob * lp_var
			lp.addConstr(state_vars[state] >= expr)
	lp.update()
	lp.optimize()
	return lp, state_vars


def exact_dual_LP(mdp):
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
	sa_vars = G.tuplelist()

	for s in mdp.reachable_states:
		sa_vars.append((s, "STOP", lp.addVar(name=str(s)+"_STOP", lb=0)))
		for a in mdp.actions:
			if a.prereq <= s:
				sa_vars.append((s, a, lp.addVar(name=str(s)+"_"+a.name, lb=0)))
	lp.update()

	# set objective
	obj = G.LinExpr()
	for s,a,var in sa_vars:
		rew = mdp.terminal_reward(s)
		if a == "STOP":
			obj += rew * var
		else:
			obj += (a.stop_prob * rew - a.cost) * var
	lp.setObjective(obj, G.GRB.MAXIMIZE)

	# set constraints
	for s in mdp.reachable_states:
		constr = G.quicksum([v for _,__,v in sa_vars.select(s)])
		for parent,action in mdp.reachable_states[s]:
			prob = action.trans_prob(parent, s, mdp.variables)
			var = sa_vars.select(parent,action)[0][2]
			constr -= prob * var
		if s == mdp.initial:
			lp.addConstr(constr, G.GRB.EQUAL, G.LinExpr(1))
		else:
			lp.addConstr(constr, G.GRB.EQUAL, G.LinExpr(0))
	lp.update()
	lp.optimize()	
	return lp, sa_vars


def action_value(mdp, state, action, values):
	"""
	Expected next-state value of perfrming action in state.
	"""
	value = -action.cost
	for outcome,prob in action.outcome_probs.items():
		next_state = outcome.transition(state)
		value += prob * values[next_state]
	value += action.stop_prob * mdp.terminal_reward(state)
	return value


def state_values(mdp, policy, values, iters=1000, cnvrg_thresh=1e-6):
	"""
	Expected value estimate for each state when following policy.

	An accurate estimate requires convergence, which may require a
	a large number of iterations. For modified policy iteration, iters
	can be set relatively low to return before convergence.
	"""
	for _i in range(iters):
		new_values = {}
		for state in mdp.reachable_states:
			action = policy[state]
			if action == None:
				new_values[state] = mdp.terminal_reward(state)
			else:
				new_values[state] = action_value(mdp, state, action, values)
		if converged(values, new_values, cnvrg_thresh):
			break
		values = new_values
	return new_values


def greedy_policy(mdp, values):
	"""
	State-action map that is one-step optimal according to values.
	"""
	new_policy = {}
	for state in mdp.reachable_states:
		best_action = None
		best_value = mdp.terminal_reward(state)
		for action in mdp.actions:
			if action.prereq <= state:
				act_val = action_value(mdp, state, action, values) 
				if act_val > best_value:
					best_value = act_val
					best_action = action
		new_policy[state] = best_action
	return new_policy


def policy_iteration(mdp, policy_iters=1000, value_iters=100, \
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
	values = {s:0 for s in mdp.reachable_states}
	for _i in range(policy_iters):
		old_values = values
		policy = greedy_policy(mdp, values)
		values = state_values(mdp, policy, values, value_iters, cnvrg_thresh)
		if converged(old_values, values, cnvrg_thresh):
			values_changed = False
	return policy, values

