from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class DGGSInfo:
    resolution: int

    @classmethod
    def from_dict(cls: type[T], mapping: dict) -> T:
        return cls(**mapping)
