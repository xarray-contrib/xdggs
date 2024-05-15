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


class Match: ...


MatchType = BaseException | Match | tuple[BaseException | Match, ...]


def extract_message(exc):
    return getattr(exc, "message", str(exc))


# necessary until pytest-dev/pytest#11538 is resolved
@dataclass
class Match:
    """match exceptions and exception groups

    Think of `Match` objects as an equivalent to `re.Pattern` classes.

    Providing a tuple in place of a single exception / matcher means logical "or".

    Parameters
    ----------
    exc : exception-like or tuple of exception-like
        The exceptions or exception groups to match.
    submatchers : match-like, optional
        The submatchers for exception groups. Note that matchers with a mixture of
        exception groups and exceptions can't provide submatchers. If that's what you
        need, provide a tuple containing multiple matchers.
    match : str or regex-like, optional
        A pattern for matching the message of the exception.
    """

    exc: type[BaseException] | tuple[type[BaseException], ...]
    submatchers: list[MatchType] = field(default_factory=list)
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

    def matches(self, exc):
        if not isinstance(exc, self.exc):
            return False

        if self.match is not None:
            message = extract_message(exc)
            match_ = re.search(self.match, message)
            if match_ is None:
                return False

        if self.submatchers and not isinstance(exc, BaseExceptionGroup):
            return False
        elif self.submatchers:
            if len(self.submatchers) != len(exc.exceptions):
                return False

            unmatched_matchers = []
            exceptions = list(exc.exceptions)
            for matcher in self.submatchers:
                for index, exception in enumerate(exceptions):
                    if matcher.matches(exception):
                        exceptions.pop(index)
                        break
                else:
                    unmatched_matchers.append(matcher)

            if unmatched_matchers:
                return False

        return True


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


def format_exception_diff(a, b):
    sections = []
    if type(a) is not type(b):
        sections.append("\n".join([f"L  {type(a).__name__}", f"R  {type(b).__name__}"]))
    else:
        sections.append(f"{type(a).__name__}")

    if isinstance(a, BaseExceptionGroup):
        if a.message != b.message:
            sections.append(
                "\n".join(["Message", f"L  {a.message}", f"R  {b.message}"])
            )
        if a.exceptions != b.exceptions:
            sections.append("\n".join(["Exceptions differ"]))
    elif str(a) != str(b):
        sections.append("\n".join(["Message", f"L  {str(a)}", f"R  {str(b)}"]))

    return "\n\n".join(sections)


def assert_exceptions_equal(actual, expected):
    assert compare_exceptions(actual, expected), format_exception_diff(actual, expected)
