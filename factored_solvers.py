from MDP_State import Basis, State, all_states

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
	basis_set.update(Basis((v,),()) for v in mdp.variables)
	basis_set.update(Basis((),(v,)) for v in mdp.variables)
	for a in mdp.actions:
		basis_set.add(Basis(a.prereq.pos, a.prereq.neg))
		basis_set.update(map(lambda o: Basis(o.pos, o.neg), a.outcomes))
	return basis_set


def factored_primal_LP(mdp, basis_vars=None, order=None):
	"""
	Construct a factored LP to approximately solve the MDP with gurobi.

	This LP follows the construction given by Guestrin, et al. in
	'Efficient Solution Algorithms for Factored MDPs', JAIR 2003.
	"""
	lp = G.Model() # Throws a NameError if gurobipy wasn't loaded
	lp.modelSense = G.GRB.MINIMIZE

	if basis_vars == None:
		basis_vars = [(b, lp.addVar(name='w_' + str(i), lb=-float("inf"), \
				ub=float("inf"))) for i,b in enumerate(default_basis(mdp))]
		lp.update()
	if order == None:
		order = sorted(mdp.variables)

	for i,a in enumerate(mdp.actions):
		# initialize ufz variables
		func_domains = set()
		for j,bw in enumerate(basis_vars):
			b,w = bw # basis function, corresponding lp variable
			g = b.backprojection(a)
			new_constraints = []
			for k,zv in enumerate(g.items()): 
				z,val = zv # states,values in domain,range of g
				h = z <= b # h_i(x), aka: does b trigger in state z?
				u = lp.addVar(name="uf_"+str(i)+"_"+str(j)+"_"+str(k), \
							lb=-float("inf"), ub=float("inf"))
				new_constraints.append((u, w * (val + h)))
				func_domains.add((u,z))
			lp.update()
			for l,r in new_constraints:
				lp.addConstr(l, G.GRB.EQUAL, r) # u^{f_i}_z = w_i*c_i(z)
			lp.update()
		# convert max constraint to linear constraints
		for j,var in enumerate(order):
			# eliminate variable
			relevant = filter(lambda f: var in f[1], func_domains)
			rel_dom = reduce(lambda s,r: s|r[1].pos|r[1].neg, relevant, set())
			rel_dom.remove(var)
			new_constraints = []
			for k,z in enumerate(all_states(free = rel_dom)):
				pos_dom = State(z.pos | {var}, z.neg)
				neg_dom = State(z.pos, z.neg | {var})
				pos_sum = [f[0] for f in relevant if f[1] <= pos_dom]
				neg_sum = [f[0] for f in relevant if f[1] <= neg_dom]
				u = lp.addVar(name="ue_"+str(i)+"_"+str(j)+"_"+str(k), \
							lb=-float("inf"), ub=float("inf"))
				new_constraints.append((u,G.quicksum(pos_sum)))
				new_constraints.append((u,G.quicksum(neg_sum)))
				func_domains.add((u,z))
			lp.update()
			for l,r in new_constraints:
				lp.addConstr(l, G.GRB.GREATER_EQUAL, r)
				lp.addConstr(l, G.GRB.GREATER_EQUAL, r)
			lp.update()
			func_domains -= set(relevant)
		# there should be only one constraint function left
		assert len(func_domains) == 1, "elimination failed: " + \
				str(len(func_domains)) + " constraints remaining"
		lp.addConstr(a.cost, G.GRB.GREATER_EQUAL, func_domains.pop())

	# set objective
	lp.setObjective(G.quicksum([b[1] for b in basis_vars if b[0]<=mdp.initial]))

	return lp


def factored_dual_LP(mdp):
	"""
	Construct a factored LP to approximately solve the MDP with gurobi.

	This LP is the dual to construction given by Guestrin, et al. in
	'Efficient Solution Algorithms for Factored MDPs', JAIR 2003.
	"""
	raise NotImplementedError("TODO")

