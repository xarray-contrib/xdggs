from dataclasses import dataclass
from typing import Any, TypeVar

try:
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

T = TypeVar("T")


@dataclass(frozen=True)
class DGGSInfo:
    """Base class for DGGS grid information objects

    Parameters
    ----------
    level : int
        The level within the grid hierarchy tree.
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
