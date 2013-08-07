#!/usr/bin/python

from argparse import ArgumentParser

from factored_MDP import random_MDP
from exact_solvers import *
from factored_solvers import *


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
		lp, state_vars = exact_primal_LP(mdp)
		print len(lp.getConstrs()), "primal LP constraints"
		print "excact primal linear programming initial state value:", \
				state_vars[mdp.initial].x

	if args.exact_dual:
		lp, sa_vars = exact_dual_LP(mdp)
		print "excact dual linear programming initial state value:", \
				lp.objVal
		try:
			print "excact dual linear programming initial state policy: "+\
					str(filter(lambda sav: sav[2].x > 0, \
					sa_vars.select(mdp.initial))[0][1].name)
		except AttributeError:
			print "excact dual linear programming initial state policy: "+\
					"STOP"

	if args.factored_primal:
		lp, basis_vars = factored_primalLP(mdp)
		print "factored primal linear programming approximate initial "+\
				"state value:", basis_vars[mdp.initial].x

	if args.factored_dual:
		lp, basis_vars = factored_dual_LP(mdp)
		print "factored primal linear programming approximate initial "+\
				"state value:", basis_vars[mdp.initial].x

	if args.policy_iter:
		policy, values = policy_iteration(mdp)
		print "policy iteration initial state value:", values[mdp.initial]
		try:
			print "policy iteration initial state policy: " + \
					str(policy[mdp.initial].name)
		except AttributeError:
			print "policy iteration initial state policy: STOP"


if __name__ == "__main__":
	args = parse_args()
	main(args)

