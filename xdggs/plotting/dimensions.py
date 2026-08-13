import pathlib

import anywidget
import traitlets

root = pathlib.Path(__file__).parent


class DimensionSliders(anywidget.AnyWidget):
    _esm = root / "dimensions.js"
    _css = root / "dimensions.css"

    dimensions = traitlets.Dict(value_trait=traitlets.Int()).tag(sync=True)

    dimension_values = traitlets.Dict(value_trait=traitlets.Int()).tag(sync=True)
    dimension_available = traitlets.Dict(value_trait=traitlets.Bool()).tag(sync=True)
