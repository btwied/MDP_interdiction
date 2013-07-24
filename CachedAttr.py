class CachedAttr(object):    
    '''Computes attribute value and caches it in the instance.
    Source: Python Cookbook 
    Author: Denis Otkidach
			http://stackoverflow.com/users/168352/denis-otkidach
    This decorator allows you to create a property which can be computed
	once and accessed many times.
    '''
    def __init__(self, method, name=None):
        self.method = method
        self.name = name or method.__name__
        self.__doc__ = method.__doc__
    def __get__(self, inst, cls): 
        if inst is None:
            return self
        elif self.name in inst.__dict__:
            return inst.__dict__[self.name]
        else:
            result = self.method(inst)
            inst.__dict__[self.name]=result
            return result    

class LazyCollection:
	"""
	Collection that can act like a tuple or set as needed.

	The input collection becomes self.data. If containment testing is
	performed, a set representation is constructed. If hashing or indexing
	is performed a tuple representation is constructed. If str or repr is
	called, a string representation is constructed. If an index is requested
	a reverse-index dict is constructed.

	This collection is not intended to be mutable: the whole point is to
	cache the various representations to speed up repeated operations. 
	Therefore no set or update methods have been provided.

	LazyCollection was designed under the assumption that elements would be
	unique. It may still be useable if this assumption is violated, but I
	make no promises.
	"""
	def __init__(self, data, sort=False):
		if sort:
			self.data = sorted(data)
		else:
			self.data = data

	@CachedAttr
	def as_tuple(self):
		if isinstance(self.data, tuple):
			return self.data
		return tuple(self.data)

	@CachedAttr
	def as_set(self):
		if isinstance(self.data, set):
			return self.data
		return set(self.data)
	
	@CachedAttr
	def as_index(self):
		return {j:i for i,j in enumerate(self.data)}

	@CachedAttr
	def as_str(self):
		return repr(self.data)

	@CachedAttr
	def as_hash(self):
		return hash(self.as_tuple)
	
	def __getitem__(self, index):
		return self.as_tuple[index]

	def __contains__(self, item):
		return item in self.as_set

	def index(self, item):
		return self.as_index[item]

	def __repr__(self):
		return self.as_str

	def __hash__(self):
		return self.as_hash

	def __len__(self):
		return len(self.data)
	
	def __cmp__(self, other):
		if isinstance(other, LazyCollection):
			return cmp(self.data, other.data)
		if isinstance(other, tuple):
			return cmp(self.as_tuple, other)
		if isinstance(other, set):
			return cmp(self.as_set, other)
		if isinstance(other, str):
			return cmp(self.as_str, other)
		return cmp(self.data, other)

