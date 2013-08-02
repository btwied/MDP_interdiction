from useful_functions import converged

def action_value(mdp, state, action, values):
	"""
	Expected next-state value of perfrming action in state.
	"""
	value = -action.cost
	for outcome,prob in action.outcome_probs.items():
		next_state = outcome.transition(state, mdp.variables)
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
	states = mdp.reachable_states()
	for _i in range(iters):
		new_values = {}
		for state in states:
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
	states = mdp.reachable_states()
	new_policy = {}
	for state in states:
		best_action = None
		best_value = mdp.terminal_reward(state)
		for action in mdp.actions:
			if action.prereq.consistent(state, mdp.variables):
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
	states = mdp.reachable_states()
	values = {s:0 for s in states}
	for _i in range(policy_iters):
		old_values = values
		policy = greedy_policy(mdp, values)
		values = state_values(mdp, policy, values, value_iters, \
									cnvrg_thresh)
		if converged(old_values, values, cnvrg_thresh):
			values_changed = False
	return policy, values

