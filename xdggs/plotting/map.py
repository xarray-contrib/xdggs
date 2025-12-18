import pathlib

import anywidget
import ipywidgets as ipw
import traitlets


class MapWithControls(anywidget.AnyWidget):
    # controls:
    # - one dropdown (if available), otherwise disable?
    # - a grid of sliders (use a label to align sliders and descriptions)
    # - maybe a colorbar
    _esm = pathlib.Path(__file__).parent / "map.js"

    # the base map
    map = traitlets.Instance(ipw.DOMWidget).tag(sync=True, **ipw.widget_serialization)

    # for choosing variables
    variables = traitlets.Instance(ipw.DOMWidget).tag(
        sync=True, **ipw.widget_serialization
    )

    # for choosing values along dimensions
    dimensions = traitlets.Dict(value_trait=traitlets.Int()).tag(sync=True)
    coordinates = traitlets.Dict(
        value_trait=traitlets.List(trait=traitlets.Unicode())
    ).tag(sync=True)

    sliders = traitlets.Dict(value_trait=traitlets.Instance(ipw.DOMWidget)).tag(
        sync=True, **ipw.widget_serialization
    )

    # the colorbar
    colorbar = traitlets.Instance(ipw.DOMWidget).tag(
        sync=True, **ipw.widget_serialization
    )

    @property
    def layers(self):
        return self.map.layers

    def add_layer(self, layer):
        self.map.add_layer(layer)
