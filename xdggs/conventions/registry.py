import warnings

from xdggs.conventions.base import Convention

_conventions = {}


class ConventionWarning(UserWarning):
    pass


def register_convention(name: str):
    """Decorator used to register a convention object

    Parameters
    ----------
    name : str
        Name of the registered convention

    Returns
    -------
    callable
        A callable that registers a :py:class:`Convention` object
    """

    def register(cls: type[Convention]):
        if name in _conventions:
            warnings.warn(
                ConventionWarning(f"Overwriting existing convention {name!r}.")
            )

        _conventions[name] = cls()

        return cls

    return register
