from useful_functions import powerset
from MDP_State import all_states

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
		backprojections = {b:b.backprojection(a) for b in basis_funcs}
		max_constr = bucket_elimination(lp, backprojections, basis_vars)
		lp.addConstr(G.LinExpr(a.cost), G.GRB.GREATER_EQUAL, max_constr)
	raise NotImplementedError("TODO: handle STOP action")#TODO: fix this
	lp.optimize()	
	return lp, basis


def bucket_elimination(lp, backprojections, basis_vars):
	"""
	Maximize sum(funcs) over the MDP's state space in sub-exponential time.

	Adds intermediate constraints to the LP and returns the expression for the
	right-hand side of the final constraint: cost >= max(sum(funcs)).

	lp:		gurobipy Model object. WILL BE MODIFIED.
	funcs:	list of (function, domain, basis_var) pairs, where each function is
			a dict mapping States to numerical values, each domain is a 
			collection of variables, and each basis_var is an LP variable.
	"""
	# order variable eliminations from least to most common occurrence
	occurrence_counts = {}
	for g in backprojections:
		if len(g) > 0:
			for var in g.itervalues().next():
				occurrence_counts[var] = occurrence_counts.get(var, 0) + 1
	order = sorted(occurrence_counts, key=occurrence_counts.get)

	interim_funcs = []
	for b,g in backprojections.itervalues():
		for state in g:
			ufz = lp.addVar(name="u^"+str(b)+"_"+str(state), \
							lb=-float("inf"), ub=float("inf"))
			interim_funcs.append((ufz, state))
			lp.addConstr(ufz, basis_vars[b] * g[state], G.GRB.EQUAL)

	raise NotImplementedError("TODO: eliminate variables")#TODO: fix this
	for i,var in enumerate(order):#TODO: fix this
		eliminated = filter(lambda pair: var in pair[1], interim_funcs)
		interim_funcs = filter(lambda pair: var not in pair[1], interim_funcs)
		new_domain = set()
		for u,z in eliminated:
			new_domain.update(z)
		new_domain -= {var}
		# add constraint for each assignment in new_domain
		# add new function to interim_funcs
		# add new function to LP vars???
#		for state in all_states(new_domain):
#			filter()
#			lp.addConstr(G.quicksum())
#			interim_funcs.append()
#			filter()
#			lp.addConstr(G.quicksum())
#			interim_funcs.append()
		
	return G.quicksum(interim_funcs)#WTF is this doing???


def factored_dual_LP(mdp):
	"""
	Construct a factored LP to approximately solve the MDP with gurobi.

	This LP is the dual to construction given by Guestrin, et al. in
	'Efficient Solution Algorithms for Factored MDPs', JAIR 2003.
	"""
	raise NotImplementedError("TODO")

