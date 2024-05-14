import pytest

from xdggs.tests import matchers

try:
    ExceptionGroup
except NameError:
    from exceptiongroup import ExceptionGroup


def test_construct_simple():
    # simple
    matchers.Match(ValueError)
    matchers.Match((ValueError, NameError))
    matchers.Match(ExceptionGroup)
    with pytest.raises(
        TypeError, match="exception type must be one or more exceptions"
    ):
        matchers.Match(int)

    # with match string
    matchers.Match(TypeError, match="pattern")
    matchers.Match(ExceptionGroup, match="pattern")
    with pytest.raises(TypeError):
        matchers.Match(ValueError, match=int)

    # with submatchers
    with pytest.raises(TypeError):
        matchers.Match(ValueError, submatchers=[matchers.Match(ValueError)])


@pytest.mark.parametrize(
    ["mapping", "args", "kwargs"],
    (({"exceptions": ValueError}, (ValueError,), {}),),
)
def test_from_dict(mapping, args, kwargs):
    actual = matchers.Match.from_dict(mapping)
    expected = matchers.Match(*args, **kwargs)
    assert actual == expected


def test_single_level_match():
    pass
