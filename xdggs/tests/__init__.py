import enum
import re
from dataclasses import dataclass, field


class MatchResult(enum.Enum):
    match = 0
    mismatched_typing = 1
    mismatched_message = 2


# necessary until pytest-dev/pytest#11538 is resolved
@dataclass
class Match:
    exc: Exception | ExceptionGroup
    submatchers: list = field(default_factory=list)
    match: str = None

    def __post_init__(self):
        if not isinstance(self.exc, ExceptionGroup) and self.submatchers:
            raise TypeError("can only pass sub-matchers for exception groups")

    def assert_match(self, exc):
        assert type(exc) is type(self.exc)  # noqa: E721

        assert re.search(str(exc), self.match), "message does not match"


def assert_exceptions_equal(actual, expected):
    assert type(actual) is type(expected)
