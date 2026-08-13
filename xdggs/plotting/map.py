import pathlib

import anywidget

root = pathlib.Path(__file__).parent


class MapWithControls(anywidget.AnyWidget):
    _esm = root / "map.js"
    _css = root / "map.css"

    # base map
    map = anywidget.WidgetTrait().tag(sync=True)

    # control panel
    control = anywidget.WidgetTrait().tag(sync=True)

    @property
    def layers(self):
        return self.map.layers

    def add_layer(self, layer):
        self.map.add_layer(layer)
