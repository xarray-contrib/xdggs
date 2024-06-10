import itertools


def identity(x):
    return x


def groupby(iterable, key):
    return itertools.groupby(sorted(iterable, key=key), key=key)
