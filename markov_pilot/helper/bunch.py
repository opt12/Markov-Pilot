#see https://stackoverflow.com/a/2827734/2682209 and the linked "bunch" recipe
#see https://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of-a-bunch-of-named/?in=user-97991#c1
#see also builtin argparse.py class Namespace(_AttributeHolder) which is very similar

class Bunch:
    """
    Often we want to just collect a bunch of stuff together, naming each item
    of the bunch; a dictionary's OK for that, but a small do-nothing class 
    is even handier, and prettier to use.
    """
    __init__ = lambda self, **kw: setattr(self, '__dict__', kw)

    # __eq__ and __contains__ bluntly stolen from builtin argparse.py class Namespace(_AttributeHolder) which is very similar
    def __eq__(self, other):
        if not isinstance(other, Bunch):
            return NotImplemented
        return vars(self) == vars(other)

    def __contains__(self, key):
        return key in self.__dict__
