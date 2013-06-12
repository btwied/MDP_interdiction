

class MDP:
	"""
	TODO: comment this!
	"""
	def __init__(self, variables, actions, rewards, basis):
		"""
		TODO: comment this!
		"""
		#TODO: implement this!
		pass

	#TODO: implement this!


class MDP_variable:
	"""
	TODO: comment this!
	"""
	def __init__(self, name, init_value=False, domain=[True,False]):
		"""
		TODO: comment this!
		"""
		self.name = name
		self.init_value = init_value
		self.domain = domain
		#TODO: finish implementing this!

	#TODO: finish implementing this!


class MDP_action:
	"""
	TODO: comment this!
	"""
	def __init__(self, name, prereqs, outcomes, probs, cost):
		"""
		TODO: comment this!
		"""
		self.name = name
		self.prereqs = prereqs
		self.outcomes = outcomes
		self.probs = probs
		self.cost = cost
		#TODO: finish implementing this!
		pass

	#TODO: finish implementing this!


if __name__ == "__main__":
	pass
