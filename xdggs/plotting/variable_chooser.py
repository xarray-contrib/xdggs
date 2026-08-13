import pathlib

import anywidget
import traitlets
from traitlets import validate

root = pathlib.Path(__file__).parent


class VariableChooser(anywidget.AnyWidget):
    _esm = root / "variable_chooser.js"

    variables = traitlets.List(value_trait=traitlets.Unicode()).tag(sync=True)
    value = traitlets.Unicode().tag(sync=True)

    @validate("value")
    def _valid_data(self, proposal: str) -> bool:
        if proposal["value"] not in self.variables:
            raise traitlets.TraitError(
                f"The selected value must be chosen from the list of variables: [{', '.join(self.variables)}]"
            )
        return proposal["value"]
