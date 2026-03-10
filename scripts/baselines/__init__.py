"""Method registry for baseline evaluation."""

METHODS = {}


def register_method(name):
    """Decorator to register an evaluation method adapter."""
    def wrapper(cls):
        METHODS[name] = cls
        return cls
    return wrapper
