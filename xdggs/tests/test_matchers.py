import pytest

from xdggs.tests import matchers

try:
    ExceptionGroup
except NameError:  # pragma: no cover
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

    @pytest.mark.parametrize(
        ["exc", "match", "expected"],
        (
            pytest.param(
                ValueError("e"), matchers.Match(ValueError), True, id="exc-match-wo pat"
            ),
            pytest.param(
                ValueError("error message"),
                matchers.Match(ValueError, match="message"),
                True,
                id="exc-match-w pat",
            ),
            pytest.param(
                ValueError("error message"),
                matchers.Match(ValueError, match="abc"),
                False,
                id="exc-no match-w pat",
            ),
            pytest.param(
                ExceptionGroup("eg", [ValueError("error")]),
                matchers.Match(ExceptionGroup),
                True,
                id="eg-match-wo pat-without submatchers",
            ),
            pytest.param(
                ExceptionGroup("error group", [ValueError("error")]),
                matchers.Match(ExceptionGroup, match="err"),
                True,
                id="eg-match-w pat-without submatchers",
            ),
            pytest.param(
                ExceptionGroup("eg", [ValueError("error")]),
                matchers.Match(ExceptionGroup, match="abc"),
                False,
                id="eg-no match-w pat-without submatchers",
            ),
            pytest.param(
                ExceptionGroup("eg", [ValueError("error")]),
                matchers.Match(
                    ExceptionGroup, submatchers=[matchers.Match(ValueError)]
                ),
                True,
                id="eg-match-wo pat-with submatchers-wo subpat",
            ),
            pytest.param(
                ExceptionGroup("eg", [ValueError("error")]),
                matchers.Match(ExceptionGroup, submatchers=[matchers.Match(TypeError)]),
                False,
                id="eg-no match-wo pat-with submatchers-wo subpat",
            ),
            pytest.param(
                ExceptionGroup("eg", [ValueError("error")]),
                matchers.Match(
                    ExceptionGroup,
                    submatchers=[matchers.Match(ValueError, match="err")],
                ),
                True,
                id="eg-match-w pat-with submatchers-wo subpat",
            ),
            pytest.param(
                ExceptionGroup("eg", [ValueError("error")]),
                matchers.Match(
                    ExceptionGroup,
                    submatchers=[matchers.Match(ValueError, match="abc")],
                ),
                False,
                id="eg-no match-w pat-with submatchers-wo subpat",
            ),
        ),
    )
    def test_matches(self, exc, match, expected):
        assert match.matches(exc) == expected


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
