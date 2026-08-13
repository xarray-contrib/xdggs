import pathlib

import anywidget
import traitlets

root = pathlib.Path(__file__).parent


class DimensionSliders(anywidget.AnyWidget):
    _esm = root / "dimensions.js"

    dimensions = traitlets.Dict(values_trait=traitlets.Int()).tag(sync=True)
