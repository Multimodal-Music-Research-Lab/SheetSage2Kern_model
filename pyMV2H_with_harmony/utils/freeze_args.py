

def freeze_args():

    """Transform mutable dictionnary
    Into immutable
    Useful to be compatible with cache
    """
    class HDict(dict):
        def __hash__(self):
            return hash(frozenset(self.items()))

    def runner(fn):
        def inner(*args, **kwargs):
            args = tuple([HDict(arg) if isinstance(arg, dict) else arg for arg in args])
            kwargs = {k: HDict(v) if isinstance(v, dict) else v for k, v in kwargs.items()}

            args = tuple([tuple(arg) if isinstance(arg, list) else arg for arg in args])
            kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in kwargs.items()}

            return fn(*args, **kwargs)
        return inner
    return runner
