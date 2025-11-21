from dataclasses import asdict, dataclass


@dataclass
class Ellipsoid:
    """In-memory representation of an ellipsoid.

    Parameters
    ----------
    name : str | None
        The name of the ellipsoid. Pass ``None`` if the ellipsoid is not named.
    semimajor_axis : float
        The semimajor axis of the ellipsoid.
    inverse_flattening : float
        The inverse flattening parameter of the ellipsoid.
    """

    name: str | None
    """The name of the ellipsoid, if any."""

    semimajor_axis: float
    """The semimajor axis of the ellipsoid."""
    inverse_flattening: float
    """The inverse flattening parameter of the ellipsoid"""

    @classmethod
    def from_dict(cls, mapping):
        return cls(**mapping)

    def to_dict(self):
        return asdict(self)


@dataclass
class Sphere:
    """In-memory representation of a sphere.

    Parameters
    ----------
    name : str | None
        The name of the sphere. Pass ``None`` if the sphere is not named.
    radius : float
        The radius of the sphere.
    """

    name: str | None
    """The name of the sphere, if any."""

    radius: float
    """The radius of the sphere."""

    @classmethod
    def from_dict(cls, mapping):
        return cls(**mapping)

    def to_dict(self):
        return asdict(self)
