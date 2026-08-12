import pathlib

import anywidget
import traitlets


class MapWithControls(anywidget.AnyWidget):
    # controls:
    # - one dropdown (if available), otherwise disable?
    # - a grid of sliders (use a label to align sliders and descriptions)
    # - maybe a colorbar
    _esm = pathlib.Path(__file__).parent / "map.js"

    # the base map
    map = anywidget.WidgetTrait().tag(sync=True)

    # for choosing variables
    variables = anywidget.WidgetTrait().tag(sync=True)

    # for choosing values along dimensions
    dimensions = traitlets.Dict(value_trait=traitlets.Int()).tag(sync=True)
    coordinates = traitlets.Dict(
        value_trait=traitlets.List(trait=traitlets.Unicode())
    ).tag(sync=True)

    sliders = traitlets.Dict(value_trait=anywidget.WidgetTrait()).tag(sync=True)

    # the colorbar
    colorbar = anywidget.WidgetTrait().tag(sync=True)

    @property
    def layers(self):
        return self.map.layers

    def add_layer(self, layer):
        self.map.add_layer(layer)
