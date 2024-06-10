import itertools


def first(x):
    return x[0]


def identity(x):
    return x


def groupby(iterable, key):
    # to avoid re-evaluating `key` twice for each item, which might be expensive
    applied = ((key(item), item) for item in iterable)
    sorted_ = sorted(applied, key=first)

    return itertools.groupby(sorted_, key=first)
