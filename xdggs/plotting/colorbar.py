import pathlib

import anywidget
import traitlets


class Colorbar(anywidget.AnyWidget):
    _esm = pathlib.Path(__file__).parent / "colorbar.js"

    vmin = traitlets.Float().tag(sync=True)
    vmax = traitlets.Float().tag(sync=True)

    colors = traitlets.List(trait=traitlets.List(trait=traitlets.Float())).tag(
        sync=True
    )

    label = traitlets.Unicode().tag(sync=True)
