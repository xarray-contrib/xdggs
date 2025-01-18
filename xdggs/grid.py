import operator
from dataclasses import dataclass
from typing import Any, TypeVar

from xdggs.itertools import groupby, identity

try:
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

try:
    ExceptionGroup
except NameError:  # pragma: no cover
    from exceptiongroup import ExceptionGroup

T = TypeVar("T")


@dataclass(frozen=True)
class DGGSInfo:
    """Base class for DGGS grid information objects

    Parameters
    ----------
    level : int
        Grid hierarchical level. A higher value corresponds to a finer grid resolution
        with smaller cell areas. The number of cells covering the whole sphere usually
        grows exponentially with increasing level values, ranging from 5-100 cells at
        level 0 to millions or billions of cells at level 10+ (the exact numbers depends
        on the specific grid).
    """

    level: int

    @classmethod
    def from_dict(cls: type[T], mapping: dict[str, Any]) -> T:
        return cls(**mapping)

    def to_dict(self: Self) -> dict[str, Any]:
        return {"level": self.level}

    def cell_ids2geographic(self, cell_ids):
        raise NotImplementedError()

    def geographic2cell_ids(self, lon, lat):
        raise NotImplementedError()

    def cell_boundaries(self, cell_ids, backend="shapely"):
        raise NotImplementedError()


def translate_parameters(mapping, translations):
    def translate(name, value):
        new_name, translator = translations.get(name, (name, identity))

        return new_name, name, translator(value)

    translated = (translate(name, value) for name, value in mapping.items())
    grouped = {
        name: [(old_name, value) for _, old_name, value in group]
        for name, group in groupby(translated, key=operator.itemgetter(0))
    }
    duplicated_parameters = {
        name: group for name, group in grouped.items() if len(group) != 1
    }
    if duplicated_parameters:
        raise ExceptionGroup(
            "received multiple values for parameters",
            [
                ValueError(
                    f"Parameter {name} received multiple values: {sorted(n for n, _ in group)}"
                )
                for name, group in duplicated_parameters.items()
            ],
        )

    params = {
        name: group[0][1] for name, group in grouped.items() if name != "grid_name"
    }
    return params
