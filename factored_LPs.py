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
	basis = [Basis((v),()) for v in mdp.variables]
	for a in mdp.actions:
		basis.append(a.prereq)
		basis.extend(a.outcomes)
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
	for b in default_basis(mdp):
		basis_vars[b] = lp.addVar(name=str(b), lb=-float("inf"), \
				ub=float("inf"))
	lp.update()

	# set objective
	lp.setObjective()

	# set constraints
	
	#TODO: finish implementing this!
	raise NotImplementedError("TODO")
	lp.optimize()	
	return lp, basis


def factored_dual_LP(mdp):
	"""
	Construct a factored LP to approximately solve the MDP with gurobi.

	This LP is the dual to construction given by Guestrin, et al. in
	'Efficient Solution Algorithms for Factored MDPs', JAIR 2003.
	"""
	raise NotImplementedError("TODO")

