from useful_functions import powerset

# import Gurobi but don't crash if it wasn't loaded
import warnings
warnings.formatwarning = lambda msg, *args: "warning: " + str(msg) + "\n"
try:
	import gurobipy as G
except ImportError:
	warnings.warn("Gurobi is required to solve MDPs by linear programming.")



def default_basis(mdp):
	"""
	Creates a basis function for each literal, prereq, and outcome.
	"""
	basis_set = {Basis((),())}
	basis_set.update(Basis((v),()) for v in mdp.variables)
#	basis_set.update(Basis((),(v)) for v in mdp.variables)
	for a in mdp.actions:
		basis_set.add(Basis(*a.prereq))
		basis_set.update(map(lambda o: Basis(*o), a.outcomes))
	return basis


def factored_primal_LP(mdp):
	"""
	Construct a factored LP to approximately solve the MDP with gurobi.

	This LP follows the construction given by Guestrin, et al. in
	'Efficient Solution Algorithms for Factored MDPs', JAIR 2003.

	The constant function f(x)=1 will be added to the basis
	automatically to ensure LP feasibility. Using default_basis
	is probably a good place to start.
	"""
	lp = G.Model() # Throws a NameError if gurobipy wasn't loaded

	basis_vars = {}
	basis_funcs = default_basis(mdp)
	for b in basis_funcs:
		basis_vars[b] = lp.addVar(name=str(b),lb=-float("inf"),ub=float("inf"))
	lp.update()

	# set objective based on basis functions with f(Initial)=1
	lp.setObjective(G.quicksum([basis_vars[b] for b in filter(lambda b: \
					b.triggered_by(mdp.initial), basis_funcs)]))

	# add constraints
	for a in mdp.actions:
		backprojections = map(lambda b: b.backprojection(a), basis_funcs)
		max_constr = bucket_elimination(lp, backprojections, order)
		lp.addConstr(G.LinExpr(a.cost), G.GRB.GREATER_EQUAL, max_constr)
	raise NotImplementedError("TODO: handle STOP action")
	lp.optimize()	
	return lp, basis


def bucket_elimination(lp, funcs):
	"""
	Maximize sum(funcs) over the MDP's state space in sub-exponential time.

	Adds intermediate constraints to the LP and returns the expression for the
	right-hand side of the final constraint: cost >= max(sum(funcs)).

	lp:		gurobipy Model object. WILL BE MODIFIED.
	funcs:	list of (function, domain) pairs, where each function is a dict
			mapping States to numerical values and each domain is a collection
			of variables.
	"""
	# order variable eliminations from least to most common occurrence
	occurrence_counts = {}
	for _,dom in funcs:
		for var in dom:
			occurrence_counts[var] = occurrence_counts.get(var, 0) + 1
	order = sorted(occurrence_counts, key=occurrence_counts.get)

	lp_vars = []
	for func,dom in funcs:
		for state in func:
			lp.addVar(name=str(state),lb=-float("inf"),ub=float("inf"))#TODO: fix this
			lp.addConstr() #TODO: implement this

	for var in order:#TODO: fix this
		eliminated = filter(lambda f: var in f.iterkeys().next(), funcs)
		lp_vars = filter(lambda f: var in f.iterkeys().next(), funcs)
		funcs = filter(lambda func_dom: var not in func_dom[1], funcs)
		new_domain = reduce(set.union, eliminated_domains) - {var}
		new_func = {}
		for true_vars in powerset(new_domain, set):
			false_vars = new_domain - true_vars
			var_true_state = State(true_vars.union({var}), false_vars)
			var_false_state = State(true_vars, false_vars.union({var}))
			no_var_state = State(true_vars, false_vars)
			var_true_sum = sum(map(lambda f: f.get(var_true_state, 0), \
									eliminated))
			var_false_sum = sum(map(lambda f: f.get(var_false_state, 0), \
									eliminated))
			new_func(no_var_state) = max(var_true_sum, var_false_sum)
			funcs.add((new_func, new_domain))
#NEED TO DEAL WITH LP VARS and only LP vars!!!
	raise NotImplementedError("TODO: implement bucket elimination")


def factored_dual_LP(mdp):
	"""
	Construct a factored LP to approximately solve the MDP with gurobi.

	This LP is the dual to construction given by Guestrin, et al. in
	'Efficient Solution Algorithms for Factored MDPs', JAIR 2003.
	"""
	raise NotImplementedError("TODO")

