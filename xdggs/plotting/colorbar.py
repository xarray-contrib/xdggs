import pathlib

import anywidget
import matplotlib
import numpy as np
import traitlets


def extract_colors(cmap: matplotlib.colors.Colormap):
    return cmap(np.linspace(0, 1, 256))[:, :3].tolist()


class Colorbar(anywidget.Anywidget):
    _esm = pathlib.Path("colorbar.js")

    vmin = traitlets.Float().tag(sync=True)
    vmax = traitlets.Float().tag(sync=True)

    colors = traitlets.List(trait=traitlets.List(trait=traitlets.Float())).tag(
        sync=True
    )

    label = traitlets.Unicode().tag(sync=True)
