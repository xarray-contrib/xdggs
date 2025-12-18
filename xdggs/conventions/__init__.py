from xdggs.conventions import cf, easygems, xdggs  # noqa: F401
from xdggs.conventions.registry import (
    conventions as _conventions,
)
from xdggs.conventions.registry import register_convention  # noqa: F401


class DecoderError(Exception):
    pass


def detect_decoder(obj, grid_info, name):
    for name, convention in _conventions.items():
        try:
            return convention.decode(obj, grid_info=grid_info, name=name)
        except DecoderError:
            pass

    raise ValueError("cannot detect a matching convention")


__all__ = ["register_decoder", "register_encoder"]
