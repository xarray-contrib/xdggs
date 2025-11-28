from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Ellipsoid:
    """In-memory representation of an ellipsoid.

    Ellipsoid parameters are taken as strings to reduce numerical errors as much
    as possible.

    Parameters
    ----------
    semimajor_axis : float
        The semimajor axis of the ellipsoid, in meters.
    inverse_flattening : float
        The inverse flattening parameter of the ellipsoid.
    name : str, optional
        The name of the ellipsoid. Pass ``None`` if the ellipsoid is not named.
    """

    semimajor_axis: float = field(kw_only=True)
    """The semimajor axis of the ellipsoid."""
    inverse_flattening: float = field(kw_only=True)
    """The inverse flattening parameter of the ellipsoid"""

    name: str | None = field(default=None, kw_only=True)
    """The name of the ellipsoid, if any."""

    @classmethod
    def from_dict(cls, mapping):
        return cls(**mapping)

    def to_dict(self):
        mapping = asdict(self)
        if self.name is None:
            del mapping["name"]

        return mapping

    def _serialize(self):
        if self.name is not None:
            return self.name

        return self.to_dict()


@dataclass
class Sphere:
    """In-memory representation of a sphere.

    Sphere parameters are taken as strings to reduce numerical errors as much as
    possible.

    Parameters
    ----------
    radius : float
        The radius of the sphere, in meters.
    name : str, optional
        The name of the sphere. Pass ``None`` if the sphere is not named.
    """

    radius: float = field(kw_only=True)
    """The radius of the sphere, in meters."""

    name: str | None = field(default=None, kw_only=True)
    """The name of the sphere, if any."""

    @classmethod
    def from_dict(cls, mapping):
        return cls(**mapping)

    def to_dict(self):
        mapping = asdict(self)
        if self.name is None:
            del mapping["name"]

        return mapping

    def _serialize(self):
        if self.name is not None:
            return self.name

        return self.to_dict()


def parse_ellipsoid(mapping: dict[str, Any]) -> Sphere | Ellipsoid:
    if "semimajor_axis" in mapping:
        return Ellipsoid.from_dict(mapping)
    elif "radius" in mapping:
        return Sphere.from_dict(mapping)
    else:
        raise ValueError(f"unknown ellipsoid definition: {mapping}")
