from xdggs.utils import GRID_REGISTRY


def infer_grid_name(index):
    for name, cls in GRID_REGISTRY.items():
        if cls is type(index):
            return name

    raise ValueError("unknown index")
