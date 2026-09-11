import itertools
from collections.abc import Callable, Sequence


def identity(x):
    return x


def groupby(iterable, key):
    return itertools.groupby(sorted(iterable, key=key), key=key)


def partition(n, iterable, pad=None):
    iterators = [iter(iterable)] * n

    return itertools.zip_longest(*iterators, fillvalue=pad)


def pairwise_tree_reduce[T](func: Callable, values: Sequence[T]) -> T:
    def reduce(a, b):
        if b is None:
            return a

        return func(a, b)

    if len(values) == 0:
        raise ValueError("must receive at least one value")

    results = list(values)
    while len(results) > 1:
        results = [reduce(a, b) for a, b in partition(2, results, pad=None)]

    return results[0]
