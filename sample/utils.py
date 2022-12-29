import itertools

def pairwise(iterator):
    a, b = itertools.tee(iterator)
    next(b, None)
    return zip(a,b)