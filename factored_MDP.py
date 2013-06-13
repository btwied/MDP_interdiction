from collections import namedtuple

from HashableClasses import h_dict

class MDP:
	"""
	TODO: comment this!
	"""
	def __init__(self, variables, actions, rewards):
		"""
		variables: A collection of Variable objects.
		actions: A collection of Action objects. a.prereqs and a.outcomes
				should only include v's from variables
		rewards: A mapping of v's in variables to floats.

		may also need to specify a basis
		"""
		self.variables = variables
		self.actions = actions
		self.rewards = rewards

	#TODO: finish implementing this!

# value must be an element of domain
Variable = namedtuple("Variable", "name value domain")

# prereqs should be an h_dict mapping vars to vals
# outcomes should be an h_dict mapping vars to probs
Action = namedtuple("Action", "name prereqs outcomes cost")



if __name__ == "__main__":
	pass
