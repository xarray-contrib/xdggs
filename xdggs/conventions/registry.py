import warnings

conventions = {}


class Convention:
    def decode(self, obj, grid_info, name, index_options):
        raise NotImplementedError

    def encode(self, obj, *, encoding=None):
        raise NotImplementedError


class DecoderWarning(UserWarning):
    pass


def register_convention(name):
    def register(cls):
        if name in conventions:
            warnings.warn(DecoderWarning(f"Overwriting existing convention {name!r}."))

        conventions[name] = cls()

        return cls

    return register
