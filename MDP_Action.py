from CachedAttr import CachedAttr

class Action:
	"""
	MDP action.

	Prerequisites specify the states in which the action is available.
	Actions encode the MDP's transition function through their distributions
	over outcomes. They also partially specify the reward function through
	their costs.
	"""
	def __init__(self, name, cost, prereq, outcome_dist):
		"""
		cost > 0
		prereq: State.Prereq object
		outcome_dist: mapping of State.Outcome objects to probabilities
		"""
		self.name = name
		self.cost = cost
		self.prereq = prereq
		self.outcome_probs = outcome_dist

	@CachedAttr
	def stop_prob(self):
		return 1. - sum(self.outcome_probs.values())

	@CachedAttr
	def outcomes(self):
		return sorted(self.outcome_probs.keys())

	@CachedAttr
	def can_make_pos(self):
		"""Set of variables that can be added."""
		pos_vars = set()
		for o in self.outcomes:
			pos_vars.update(o.pos)
		return pos_vars

	@CachedAttr
	def can_make_neg(self):
		"""Set of variables that can be deleted."""
		neg_vars = set()
		for o in self.outcomes:
			neg_vars.update(o.neg)
		return neg_vars

	@CachedAttr
	def must_make_pos(self):
		"""Set of variables that always end up positive."""
		pos_vars = set(self.outcomes[0].pos)
		for o in self.outcomes:
			pos_vars.intersection_update(o.pos)
		return pos_vars

	@CachedAttr
	def must_make_neg(self):
		"""Set of variables that always end up negative."""
		neg_vars = set(self.outcomes[0].neg)
		for o in self.outcomes:
			neg_vars.intersection_update(o.neg)
		return neg_vars

	@CachedAttr
	def add_prob(self):
		"""Dict of probabilities that each variable will be added."""
		add_probs = defaultdict(lambda: 0)
		for o in self.outcomes:
			for v in o.pos:
				add_probs[v] += self.outcome_probs[o]
		return pos_probs

	@CachedAttr
	def del_prob(self):
		"""Dict of probabilities that each variable will be deleted."""
		del_probs = defaultdict(lambda: 0)
		for o in self.outcomes:
			for v in o.neg:
				del_probs[v] += self.outcome_probs[o]
		return del_probs

	def trans_prob(self, pre_state, post_state, variables):
		prob = 0
		for o,p in self.outcome_probs.items():
			if o.transition(pre_state, variables) == post_state:
				prob += p
		return prob

	def can_change(self, state, variables):
		return any(o.changes(state, variables) for o in self.outcomes)

	def __repr__(self):
		return "MDP action: name=" + self.name + ", prereq=" + \
				repr(self.prereq) + ", outcomes=" + repr(self.outcomes)

	def __hash__(self):
		return hash(self.name)

	def __cmp__(self, other):
		try:
			return cmp(self.name, other.name)
		except AttributeError:
			return cmp(self.name, other)
