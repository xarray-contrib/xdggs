import pytest

from xdggs.tests import matchers

try:
    ExceptionGroup
except NameError:
    from exceptiongroup import ExceptionGroup


class TestMatch:
    def test_construct_simple(self):
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
        (
            ({"exceptions": ValueError}, (ValueError,), {}),
            (
                {"exceptions": ValueError, "match": "abc"},
                (ValueError,),
                {"match": "abc"},
            ),
        ),
    )
    def test_from_dict(self, mapping, args, kwargs):
        actual = matchers.Match.from_dict(mapping)
        expected = matchers.Match(*args, **kwargs)
        assert actual == expected


def test_assert_exceptions_equal():
    actual = ValueError("error message")
    expected = ValueError("error message")

    matchers.assert_exceptions_equal(actual, expected)

    try:
        raise ValueError("error message")
    except ValueError as e:
        actual = e
    expected = ValueError("error message")

    matchers.assert_exceptions_equal(actual, expected)

    actual = ValueError("error message")
    expected = TypeError("error message")
    with pytest.raises(AssertionError):
        matchers.assert_exceptions_equal(actual, expected)

    actual = ValueError("error message1")
    expected = ValueError("error message2")
    with pytest.raises(AssertionError):
        matchers.assert_exceptions_equal(actual, expected)

    actual = ExceptionGroup("group message1", [ValueError("error message")])
    expected = ExceptionGroup("group message2", [ValueError("error message")])
    with pytest.raises(AssertionError):
        matchers.assert_exceptions_equal(actual, expected)

    actual = ExceptionGroup("group message", [ValueError("error message1")])
    expected = ExceptionGroup("group message", [ValueError("error message2")])
    with pytest.raises(AssertionError):
        matchers.assert_exceptions_equal(actual, expected)

    actual = ExceptionGroup("group message", [ValueError("error message")])
    expected = ExceptionGroup("group message", [TypeError("error message")])
    with pytest.raises(AssertionError):
        matchers.assert_exceptions_equal(actual, expected)
