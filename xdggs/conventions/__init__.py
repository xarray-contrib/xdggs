from xdggs.conventions import cf, easygems, xdggs  # noqa: F401
from xdggs.conventions.base import Convention
from xdggs.conventions.errors import DecoderError
from xdggs.conventions.registry import _conventions, register_convention


def detect_decoder(obj, grid_info, name):
    for name, convention in _conventions.items():
        try:
            return convention.decode(obj, grid_info=grid_info, name=name)
        except DecoderError:
            pass

    raise ValueError("cannot detect a matching convention")


__all__ = ["register_convention", "detect_decoder", "DecoderError", "Convention"]
