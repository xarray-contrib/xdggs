import enum
import re
from dataclasses import dataclass, field

try:
    ExceptionGroup
except NameError:
    from exceptiongroup import BaseExceptionGroup, ExceptionGroup


class MatchResult(enum.Enum):
    match = 0
    mismatched_typing = 1
    mismatched_message = 2


def is_exception_spec(exc):
    if not isinstance(exc, tuple):
        exc = (exc,)

    return all(isinstance(e, type) and issubclass(e, BaseException) for e in exc)


def is_exceptiongroup_spec(exc):
    if not isinstance(exc, tuple):
        exc = (exc,)

    return all(isinstance(e, type) and issubclass(e, BaseExceptionGroup) for e in exc)


# necessary until pytest-dev/pytest#11538 is resolved
@dataclass
class Match:
    exc: Exception | ExceptionGroup | tuple[Exception | ExceptionGroup, ...]
    submatchers: list["Match"] = field(default_factory=list)
    match: str = None

    def __post_init__(self):
        if not is_exception_spec(self.exc):
            raise TypeError(
                f"exception type must be one or more exceptions, got: {self.exc}"
            )
        if not is_exceptiongroup_spec(self.exc) and self.submatchers:
            raise TypeError("can only pass sub-matchers for exception groups")
        if not isinstance(self.match, str) and self.match is not None:
            raise TypeError("match must be either `None` or a string pattern")

    @classmethod
    def from_dict(cls, mapping):
        children = [cls.from_dict(m) for m in mapping.get("children", [])]

        return cls(
            mapping["exceptions"],
            submatchers=children,
            match=mapping.get("match", None),
        )

    def assert_match(self, exc):
        assert type(exc) is type(self.exc)  # noqa: E721

        assert re.search(str(exc), self.match), "message does not match"


def compare_exceptions(a, b):
    if type(a) is not type(b):
        return False

    if isinstance(a, ExceptionGroup):
        comparison = a.args[0] == b.args[0] and all(
            compare_exceptions(_a, _b) for _a, _b in zip(a.args[1], b.args[1])
        )
    else:
        comparison = a.args == b.args

    return comparison


def assert_exceptions_equal(actual, expected):
    assert type(actual) is type(expected)

    assert compare_exceptions(actual, expected), "mismatching exceptions"
