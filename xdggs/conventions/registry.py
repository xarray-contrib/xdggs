import warnings

from xdggs.conventions.base import Convention

_conventions = {}


class ConventionWarning(UserWarning):
    pass


def register_convention(name: str):
    def register(cls: type[Convention]):
        if name in _conventions:
            warnings.warn(
                ConventionWarning(f"Overwriting existing convention {name!r}.")
            )

        _conventions[name] = cls()

        return cls

    return register
