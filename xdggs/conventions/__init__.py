from xdggs.conventions.registry import decoders as _decoders
from xdggs.conventions.registry import encoders as _encoders  # noqa: F401
from xdggs.conventions.registry import (
    register_decoder,
    register_encoder,
)


class DecoderError(Exception):
    pass


def detect_convention_decoder(obj, grid_info, name):
    for name, decoder in _decoders.items():
        try:
            return decoder(obj, grid_info=grid_info, name=name)
        except DecoderError:
            pass

    raise ValueError("cannot detect a matching convention")


__all__ = ["register_decoder", "register_encoder"]
