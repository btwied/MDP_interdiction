from itertools import chain, combinations


def converged(old_vals, new_vals, thresh=1e-6):
	return all((abs(new_vals[s] - old_vals[s]) < thresh for s in new_vals))


def powerset(s):
	return chain.from_iterable(combinations(s,i) for i in range(len(s)+1))

