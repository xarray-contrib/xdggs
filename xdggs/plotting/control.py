import pathlib

import anywidget

root = pathlib.Path(__file__).parent


class ControlPanel(anywidget.AnyWidget):
    _esm = root / "control.js"
    _css = root / "control.css"

    # controls:
    # - one dropdown (if available), otherwise disable?
    # - a grid of sliders (use a label to align sliders and descriptions)
    # - a colorbar

    variable_chooser = anywidget.WidgetTrait().tag(sync=True)
    dimension_sliders = anywidget.WidgetTrait().tag(sync=True)
    colorbar = anywidget.WidgetTrait().tag(sync=True)
