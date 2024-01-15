from dataclasses import dataclass
from typing import Any, Self, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class DGGSInfo:
    resolution: int

    @classmethod
    def from_dict(cls: type[T], mapping: dict[str, Any]) -> T:
        return cls(**mapping)

    def to_dict(self: Self) -> dict[str, Any]:
        return {"resolution": self.resolution}
