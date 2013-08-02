class CachedAttr(object):    
    """
	Computes attribute value and caches it in the instance.

    Source: Python Cookbook
    Author: Denis Otkidach
			http://stackoverflow.com/users/168352/denis-otkidach
	Retreived 7/13: http://stackoverflow.com/questions/3237678/how-to-
					create-decorator-for-lazy-initialization-of-a-property
	
    Use this as a decorator. If applied to a class member function, that
	function's name can be accessed as an attribute. The first time the
	attribute is accessed, the function is called, but the return value is
	cached, so that subsequent accesses don't re-compute.
    """
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

