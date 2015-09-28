
#import to define this file
from functools import update_wrapper, _c3_merge, _c3_mro, _compose_mro
from itertools import zip_longest
from operator import itemgetter
import collections


#original import
from abc import get_cache_token
from types import MappingProxyType
from weakref import WeakKeyDictionary

################################################################################
### singledispatch() - single-dispatch generic function decorator
################################################################################

class ClassContainer():
    def __new__(cls, data = None):
        new = super(ClassContainer, cls).__new__(cls)
        new.value = None
        return new

    def __init__(self, data = None):
        super().__init__()
        if isinstance(data, collections.abc.Mapping):
            for key, value in data.items():
                self.setitem(key, value)
        if isinstance(data, collections.abc.Sequence):
            for key, value in data:
                self.setitem(key, value)
        

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.getitem(key)
        else:
            return super().__getitem__(key)

    def getitem(self, keys):
        result = None
        if isinstance(keys, tuple):
            if len(keys) > 0:
                result = self[keys[0]][keys[1:]]
            else:
                result = self.value
        else:
            raise TypeError
        if result is None:
            raise KeyError
        return result

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            self.setitem(key, value)
        else:
            super().__setitem__(key, value)

    def setitem(self, keys, value):
        if isinstance(keys, tuple):
            if len(keys) > 0:
                if keys[0] not in self.keys():
                    self[keys[0]] = self.__class__()
                self[keys[0]][keys[1:]] = value
            else:
                self.value = value
        else:
            raise TypeError
    
    def allkeys(self):
        def _allkeys(parent, root = ()):
            if parent.value is not None:
                yield root
            for node, child in parent.items():
                for root_ in _allkeys(child, root + (node,)):
                    yield root_
        return _allkeys(self)
    
    def allvalues(self):
        return (self[key] for key in self.allkeys())

    def allitems(self):
        return zip(self.allkeys(), self.allvalues())
    

    def __repr__(self):
        return "{self.__class__.__name__}(\n{0})".format(
            "".join(
                "\t{key} : {value},\n".format(**locals())
                for key, value in self.allitems()),
            **locals())

class Dispatch_Cache(ClassContainer, WeakKeyDictionary):
    pass

class Registry(ClassContainer, dict):
    pass

def _find_impl(cls, registry):
    """Returns the best matching implementation from *registry* for type *cls*.

    Where there is no registered implementation for a specific type, its method
    resolution order is used to find a more generic implementation.

    Note: if *registry* does not contain an implementation for the base
    *object* type, this function may return None.

    """
    mro = _compose_mro(cls, registry.keys())
    match = None
    for t in mro:
        if match is not None:
            # If *match* is an implicit ABC but there is another unrelated,
            # equally matching implicit ABC, refuse the temptation to guess.
            if (t in registry and t not in cls.__mro__
                              and match not in cls.__mro__
                              and not issubclass(match, t)):
                raise RuntimeError("Ambiguous dispatch: {} or {}".format(
                    match, t))
            break
        if t in registry:
            match = t
    return registry.get(match)

def _find_impl_multi(clsTuple, registry):
    """Returns the best matching implementation from *registry* for type *cls*.

    Where there is no registered implementation for a specific type, its method
    resolution order is used to find a more generic implementation.

    Note: if *registry* does not contain an implementation for the base
    *object* type, this function may return None.

    """

    types_gen = (
        set(filter(lambda typ : typ is not None, types))
        for types in zip_longest(*registry.allkeys()))
    
    mros = tuple(_compose_mro(cls, types)
                 for cls, types in zip(clsTuple, types_gen))

    MaxMatchLen = 0
    IndexArgstypesList = []
    for argstypes in registry.allkeys():
        if not MaxMatchLen <= len(argstypes) <= len(mros):
            continue
        try:
            index = tuple(mro.index(argtype)
                          for mro, argtype in zip(mros, argstypes))
        except ValueError:
            continue
        
        if MaxMatchLen < len(index):
            MaxMatchLen = len(index)
            IndexArgstypesList.clear()
        IndexArgstypesList.append((index, argstypes))
    IndexArgstypesList.sort(key = itemgetter(0))

    match_indexs, match_typs = IndexArgstypesList.pop(0)
    for i, match_index_typ in enumerate(
        zip(match_indexs, match_typs)):
        match_index, match_typ = match_index_typ
        pivot = 0
        for temp, index_typ in enumerate(IndexArgstypesList):
            index, typ = map(itemgetter(i), index_typ)
            if typ not in clsTuple[i].__mro__\
               and match_typ not in clsTuple[i].__mro__\
               and not issubcluss(match_typ, typ):
                message = "Ambiguous dispatch: {} or {} at positional arguement {}"
                raise RuntimeError(message.format(match_typ, typ, i))
            if pivot == 0 and index > match_index:
                pivot = temp
        IndexArgstypesList = IndexArgstypesList[:pivot]
        
    return registry[match_typs]

def singledispatch(func):
    """Single-dispatch generic function decorator.

    Transforms a function into a generic function, which can have different
    behaviours depending upon the type of its first argument. The decorated
    function acts as the default implementation, and additional
    implementations can be registered using the register() attribute of the
    generic function.

    """
    registry = {}
    dispatch_cache = WeakKeyDictionary()
    cache_token = None

    def dispatch(cls):
        """generic_func.dispatch(cls) -> <function implementation>

        Runs the dispatch algorithm to return the best available implementation
        for the given *cls* registered on *generic_func*.

        """
        nonlocal cache_token
        if cache_token is not None:
            current_token = get_cache_token()
            if cache_token != current_token:
                dispatch_cache.clear()
                cache_token = current_token
        try:
            impl = dispatch_cache[cls]
        except KeyError:
            try:
                impl = registry[cls]
            except KeyError:
                impl = _find_impl(cls, registry)
            dispatch_cache[cls] = impl
        return impl

    def register(cls, func=None):
        """generic_func.register(cls, func) -> func

        Registers a new implementation for the given *cls* on a *generic_func*.

        """
        nonlocal cache_token
        if func is None:
            return lambda f: register(cls, f)
        registry[cls] = func
        if cache_token is None and hasattr(cls, '__abstractmethods__'):
            cache_token = get_cache_token()
        dispatch_cache.clear()
        return func

    def wrapper(*args, **kw):
        return dispatch(args[0].__class__)(*args, **kw)

    registry[object] = func
    wrapper.register = register
    wrapper.dispatch = dispatch
    wrapper.registry = MappingProxyType(registry)
    wrapper._clear_cache = dispatch_cache.clear
    update_wrapper(wrapper, func)
    return wrapper

def multidispatch(func):
    """Single-dispatch generic function decorator.

    Transforms a function into a generic function, which can have different
    behaviours depending upon the type of its first argument. The decorated
    function acts as the default implementation, and additional
    implementations can be registered using the register() attribute of the
    generic function.

    """

    registry = Registry()
    dispatch_cache = Dispatch_Cache()
    cache_token = None

    def dispatch(*clsTuple):
        """generic_func.dispatch(cls) -> <function implementation>

        Runs the dispatch algorithm to return the best available implementation
        for the given *cls* registered on *generic_func*.

        """
        nonlocal cache_token
        if cache_token is not None:
            current_token = get_cache_token()
            if cache_token != current_token:
                dispatch_cache.clear()
                cache_token = current_token
        
        try:
            impl = dispatch_cache[clsTuple]
        except KeyError:
            try:
                impl = registry[clsTuple]
            except KeyError:
                impl = _find_impl_multi(clsTuple, registry)
                
            dispatch_cache[clsTuple] = impl
        return impl

    def register(*clsTuple, func=None):
        """generic_func.register(cls, func) -> func

        Registers a new implementation for the given *cls* on a *generic_func*.

        """
        nonlocal cache_token
        if func is None:
            return lambda f: register(*clsTuple, func = f)
        registry[clsTuple] = func
        if cache_token is None and any(hasattr(cls, '__abstractmethods__')
                                       for cls in clsTuple):
            cache_token = get_cache_token()
        dispatch_cache.clear()
        return func

    def wrapper(*args, **kw):
        return dispatch(*map(type, args))(*args, **kw)
        #return dispatch(*args)(*args, **kw)

    registry[()] = func
    wrapper.register = register
    wrapper.dispatch = dispatch
    wrapper.registry = registry
    wrapper._clear_cache = dispatch_cache.clear
    update_wrapper(wrapper, func)
    return wrapper
