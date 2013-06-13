from collections import OrderedDict


def _blocked_attribute(*args, **kwargs):
	raise TypeError("hashing requires immutability")


class h_dict(OrderedDict):
	"""
	An sorted and immutable (and therefore hashable) subclass of OrderedDict.
	"""
	__setitem__ = update = setdefault = _blocked_attribute
	__delitem__ = clear = pop = popitem = _blocked_attribute

	def __init__(self, *args, **kwargs):
		OrderedDict.__init__(self)
		d = dict(*args, **kwargs)
		for k,v in sorted(d.items()):
			OrderedDict.__setitem__(self, k, v)

	def __hash__(self):
		try:
			return self._hash
		except AttributeError:
			self._hash = hash(tuple(self.items()))
			return self._hash

	def __repr__(self):
		return '{' + OrderedDict.__repr__(self)[8:-2] + '}'

