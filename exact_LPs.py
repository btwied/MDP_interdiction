from itertools import product


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
	states = mdp.reachable_states()
	state_vars = {}

	# add a variable to the LP to represent the value of each state
	for s in states:
		state_vars[s] = lp.addVar(name=mdp.state_name(s), lb=-float("inf"))
	lp.update()
	# objective is the value of the initial state
	lp.setObjective(state_vars[mdp.initial])

	# can always cash out
	for s,v in state_vars.items():
		lp.addConstr(v >= mdp.terminal_reward(s))

	# backpropagation
	for state,action in product(states, mdp.actions):
		if action.prereq.consistent(state, mdp.variables) and \
				action.can_change(state, mdp.variables):
			const = action.stop_prob * mdp.terminal_reward(state)
			const -= action.cost
			expr = G.LinExpr(float(const))
			for outcome,prob in action.outcome_probs.items():
				lp_var = state_vars[outcome.transition(state, \
									mdp.variables)]
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
	states = mdp.reachable_states()
	sa_vars = G.tuplelist()

	for s in states:
		n = mdp.state_name(s)
		sa_vars.append((s, "STOP", lp.addVar(name=n+"_STOP", lb=0)))
		for a in mdp.actions:
			if a.prereq.consistent(s, mdp.variables):
				sa_vars.append((s, a, lp.addVar(name=n +"_"+ a.name, lb=0)))
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
	for s in states:
		constr = G.quicksum([v for _,a,v in sa_vars.select(s)])
		for parent,action in mdp.reachable[s]:
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


