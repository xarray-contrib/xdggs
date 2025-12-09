from dataclasses import dataclass
from typing import Any, Self

import matplotlib
import numpy as np


def normalize(var, params):
    from matplotlib.colors import CenteredNorm, Normalize

    if params.center is not None:
        halfrange = params.derive_limit(np.abs(var - params.center), kind="vmax")
        normalizer = CenteredNorm(vcenter=params.center, halfrange=halfrange)
    else:
        vmin = params.derive_limit(var, kind="vmin")
        vmax = params.derive_limit(var, kind="vmax")
        normalizer = Normalize(vmin=vmin, vmax=vmax)

    stats = {"vmin": normalizer.vmin, "vmax": normalizer.vmax}
    return normalizer(var.data), stats


def colorize(normalized, params):
    from lonboard.colormap import apply_continuous_cmap

    return apply_continuous_cmap(normalized, params.cmap, alpha=params.alpha)


def extract_colors(cmap: matplotlib.colors.Colormap):
    return cmap(np.linspace(0, 1, 256))[:, :3].tolist()


@dataclass
class ColorizeParameters:
    vmin: float | None = None
    vmax: float | None = None
    center: float | None = None
    robust: bool = False

    alpha: float | None = None
    cmap: str | matplotlib.colors.Colormap = "viridis"

    def __post_init__(self):
        if isinstance(self.cmap, str):
            self.cmap = matplotlib.colormaps[self.cmap]

    @classmethod
    def from_dict(cls, mapping: dict[str, Any]) -> Self:
        return cls(**mapping)

    def derive_limit(self, var, *, kind):
        kinds = {
            "vmin": ("min", 0.02),
            "vmax": ("max", 0.98),
        }

        stored = getattr(self, kind, None)
        if stored is not None:
            return stored

        method, quantile = kinds[kind]
        if self.robust:
            return var.quantile(q=quantile, skipna=True).item()
        else:
            return getattr(var, method)(skipna=True).item()
