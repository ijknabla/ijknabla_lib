
from functools import update_wrapper, _find_impl, lru_cache
from itertools import chain, zip_longest
import collections

#sources copied from /Python/Lib/functools.py
from abc import get_cache_token
from types import MappingProxyType
from weakref import WeakKeyDictionary

################################################################################
### singledispatch() - single-dispatch generic function decorator
################################################################################

def _c3_merge(sequences):
    """Merges MROs in *sequences* to a single MRO using the C3 algorithm.

    Adapted from http://www.python.org/download/releases/2.3/mro/.

    """
    result = []
    while True:
        sequences = [s for s in sequences if s]   # purge empty sequences
        if not sequences:
            return result
        for s1 in sequences:   # find merge candidates among seq heads
            candidate = s1[0]
            for s2 in sequences:
                if candidate in s2[1:]:
                    candidate = None
                    break      # reject the current head, it appears later
            else:
                break
        if not candidate:
            raise RuntimeError("Inconsistent hierarchy")
        result.append(candidate)
        # remove the chosen candidate
        for seq in sequences:
            if seq[0] == candidate:
                del seq[0]

def _c3_mro(cls, abcs=None):
    """Computes the method resolution order using extended C3 linearization.

    If no *abcs* are given, the algorithm works exactly like the built-in C3
    linearization used for method resolution.

    If given, *abcs* is a list of abstract base classes that should be inserted
    into the resulting MRO. Unrelated ABCs are ignored and don't end up in the
    result. The algorithm inserts ABCs where their functionality is introduced,
    i.e. issubclass(cls, abc) returns True for the class itself but returns
    False for all its direct base classes. Implicit ABCs for a given class
    (either registered or inferred from the presence of a special method like
    __len__) are inserted directly after the last ABC explicitly listed in the
    MRO of said class. If two implicit ABCs end up next to each other in the
    resulting MRO, their ordering depends on the order of types in *abcs*.

    """
    for i, base in enumerate(reversed(cls.__bases__)):
        if hasattr(base, '__abstractmethods__'):
            boundary = len(cls.__bases__) - i
            break   # Bases up to the last explicit ABC are considered first.
    else:
        boundary = 0
    abcs = list(abcs) if abcs else []
    explicit_bases = list(cls.__bases__[:boundary])
    abstract_bases = []
    other_bases = list(cls.__bases__[boundary:])
    for base in abcs:
        if issubclass(cls, base) and not any(
                issubclass(b, base) for b in cls.__bases__
            ):
            # If *cls* is the class that introduces behaviour described by
            # an ABC *base*, insert said ABC to its MRO.
            abstract_bases.append(base)
    for base in abstract_bases:
        abcs.remove(base)
    explicit_c3_mros = [_c3_mro(base, abcs=abcs) for base in explicit_bases]
    abstract_c3_mros = [_c3_mro(base, abcs=abcs) for base in abstract_bases]
    other_c3_mros = [_c3_mro(base, abcs=abcs) for base in other_bases]
    return _c3_merge(
        [[cls]] +
        explicit_c3_mros + abstract_c3_mros + other_c3_mros +
        [explicit_bases] + [abstract_bases] + [other_bases]
    )

def _compose_mro(cls, types):
    """Calculates the method resolution order for a given class *cls*.

    Includes relevant abstract base classes (with their respective bases) from
    the *types* iterable. Uses a modified C3 linearization algorithm.

    """
    bases = set(cls.__mro__)
    # Remove entries which are already present in the __mro__ or unrelated.
    def is_related(typ):
        return (typ not in bases and hasattr(typ, '__mro__')
                                 and issubclass(cls, typ))
    types = [n for n in types if is_related(n)]
    # Remove entries which are strict bases of other entries (they will end up
    # in the MRO anyway.
    def is_strict_base(typ):
        for other in types:
            if typ != other and typ in other.__mro__:
                return True
        return False
    types = [n for n in types if not is_strict_base(n)]
    # Subclasses of the ABCs in *types* which are also implemented by
    # *cls* can be used to stabilize ABC ordering.
    type_set = set(types)
    mro = []
    for typ in types:
        found = []
        for sub in typ.__subclasses__():
            if sub not in bases and issubclass(cls, sub):
                found.append([s for s in sub.__mro__ if s in type_set])
        if not found:
            mro.append(typ)
            continue
        # Favor subclasses with the biggest number of useful bases
        found.sort(key=len, reverse=True)
        for sub in found:
            for subcls in sub:
                if subcls not in mro:
                    mro.append(subcls)
    return _c3_mro(cls, abcs=mro)

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


def _find_impl(clsTuple, registry):
    """Returns the best matching implementation from *registry* for type *cls*.

    Where there is no registered implementation for a specific type, its method
    resolution order is used to find a more generic implementation.

    Note: if *registry* does not contain an implementation for the base
    *object* type, this function may return None.

    """
    registry_list = sorted(registry.allitems(),
                           key = lambda key_value : len(key_value[0]))
    #print(registry_list)
    registries = tuple(
        frozenset(typ for typ in types if typ is not None)
        for types in zip_longest(*registry.allkeys()))
    #print(registries)
    mros = tuple(_compose_mro(cls, registry)
                for cls, registry in zip(clsTuple, registries))
    #mro = _compose_mro(cls, registry.keys())
    #print(mros)
    match = None
    
    while len(registry_list[-1][0]) > len(mros):
        registry_list.pop()
        
    for length in range(len(mros), -1, -1):
        best_fit, best_func = None, None
        while len(registry_list[-1][0]) == length:
            keys, func = (registry_list.pop())
            print(keys)
            temp_fit = []
            try:
                for i, key in enumerate(keys):
                    temp_fit.append(mros[i].index(key))
            except ValueError:
                continue
            print("match", keys, temp_fit)
            if best_fit is None or temp_fit < best_fit:
                best_fit, best_func = temp_fit, func
        if best_func is not None:
            print(best_fit)
            break
    else:
        best_func = registry[()]

    return best_func
    
            
            
    return 
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

    class Node():
        def __new__(cls):
            new = super(Node, cls).__new__(cls)
            new.value = None
            return new

        def __getitem__(self, *index):
            if len(index) > 1:
                return self[index[0]].__getitem__(*index[1:])
            if index == slice(None, None, None):
                return self.value
            else:
                return super().__getitem__(index)

        def __setitem__(self, *args):
            index, value = args[:-1], args[-1]
            if len(index) > 1:
                self[args[0]].__setitem__(*args[1:])
            else:
                index = index[0]
                if index == slice(None, None, None):
                    self.value = value
                else:
                    super().__setitem__(index, value)


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
                impl = _find_impl(clsTuple, registry)
                
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


#testcodes
@singledispatch
def f(*args):
    raise TypeError

@f.register(int)
def _(a, b):
    return a + b

@f.register(str)
def _(a, b):
    return a + b

@multidispatch
def g(*args):
    raise TypeError

def _(*args):
    print(*args)

class myint(int):
    pass
one = myint(1)

g.register(myint, int)(_)
g.register(int, int)(_)
g.register(int, str, int)(_)
g.register(str)(_)
