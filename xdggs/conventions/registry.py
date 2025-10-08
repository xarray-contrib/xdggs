import warnings

decoders = {}
encoders = {}


class DecoderWarning(UserWarning):
    pass


def register_decoder(name):
    def register(func):
        if name in decoders:
            warnings.warn(
                DecoderWarning(f"Overwriting existing convention decoder {name!r}.")
            )

        decoders[name] = func

        return func

    return register


def register_encoder(name):
    def register(func):
        if name in encoders:
            warnings.warn(
                DecoderWarning(f"Overwriting existing convention encoder {name!r}.")
            )

        encoders[name] = func

        return func

    return register
