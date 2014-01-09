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

	The constant function f(x)=1 is also included in the basis to ensure LP 
	feasibility.
	"""
	basis_set = {Basis((),())}
	basis_set.update(Basis((v),()) for v in mdp.variables)
	basis_set.update(Basis((),(v)) for v in mdp.variables)
	for a in mdp.actions:
		basis_set.add(Basis(*a.prereq))
		basis_set.update(map(lambda o: Basis(*o), a.outcomes))
	return basis


def factored_primal_LP(mdp, basis_vars=None, order=None):
	"""
	Construct a factored LP to approximately solve the MDP with gurobi.

	This LP follows the construction given by Guestrin, et al. in
	'Efficient Solution Algorithms for Factored MDPs', JAIR 2003.
	"""
	lp = G.Model() # Throws a NameError if gurobipy wasn't loaded

	if basis_vars == None:
		basis_vars = [(b, lp.addVar(name='w_' + str(b), lb=-float("inf"), \
						ub=float("inf"))) for b in default_basis(mdp)]
		lp.update()
	if order == None:
		order = sorted(mdp.variables)

	for a in mdp.actions:
		# initialize ufz variables
		func_domains = set()
		for b,w in basis_vars:
			g = b.backprojection(a)
			for z,val in g.items(): # states,values in domain,range of g
				h = z <= b # h_i(x), aka does b trigger in state z?
				u = lp.addVar(name="u_"+str(a)+"_"+str(b)+"_"+str(z), \
							lb=-float("inf"), ub=float("inf"))
				lp.addConstr(u, '=', w * (val + h)) # u^{f_i}_z = w_i*c_i(z)
				func_domains.add((u,z))
		# convert max constraint to linear constraints
		for var in order:
			# eliminate variable
			relevant = filter(lambda f: var in f[1], func_domains):
			for z in all_states(free = (z.pos | z.neg) - {var})
				pos_dom = State(z.pos | {var}, z.neg)
				neg_dom = State(z.pos, z.neg | {var})
				pos_sum = [f[0] for f in relevant if f[1] <= pos_dom]
				neg_sum = [f[0] for f in relevant if f[1] <= neg_dom]
				u = lp.addVar(name="u_"+str(a)+"_"+str(var)+"_"+str(z), \
							lb=-float("inf"), ub=float("inf"))
				lp.addConstr(u, G.GRB.GREATER_EQUAL, G.quicksum(pos_sum))
				lp.addConstr(u, G.GRB.GREATER_EQUAL, G.quicksum(neg_sum))
				func_domains.append((u,z))
			func_domains -= set(relevant)
		# there should be only one constraint function left
		lp.addConstr(a.cost, G.GRB.GREATER_EQUAL, func_domains.pop())

	# set objective
	#TODO

	return lp


def factored_dual_LP(mdp):
	"""
	Construct a factored LP to approximately solve the MDP with gurobi.

	This LP is the dual to construction given by Guestrin, et al. in
	'Efficient Solution Algorithms for Factored MDPs', JAIR 2003.
	"""
	raise NotImplementedError("TODO")

