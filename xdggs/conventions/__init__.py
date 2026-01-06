from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, overload

from xdggs.conventions import cf, xdggs, zarr  # noqa: F401
from xdggs.conventions.base import Convention
from xdggs.conventions.errors import DecoderError
from xdggs.conventions.registry import _conventions, register_convention
from xdggs.utils import call_on_dataset

if TYPE_CHECKING:
    from collections.abc import Hashable
    from typing import Any

    import xarray as xr

    from xdggs.grid import GridInfoType


@overload
def decode(
    obj: xr.Dataset,
    grid_info: GridInfoType | None = None,
    *,
    name: Hashable | None = None,
    convention: str = "xdggs",
    index_options: dict[str, Any] | None = None,
    **index_kwargs: dict[str, Any],
) -> xr.Dataset: ...


@overload
def decode(
    obj: xr.DataArray,
    grid_info: GridInfoType | None = None,
    *,
    name: Hashable | None = None,
    convention: str = "xdggs",
    index_options: dict[str, Any] | None = None,
    **index_kwargs: dict[str, Any],
) -> xr.DataArray: ...


def decode(
    obj: xr.Dataset | xr.DataArray,
    grid_info: GridInfoType | None = None,
    *,
    name: Hashable | None = None,
    convention: str = "xdggs",
    index_options: dict[str, Any] | None = None,
    **index_kwargs: dict[str, Any],
) -> xr.Dataset | xr.DataArray:
    """
    decode grid parameters and create a DGGS index

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset. Must contain a coordinate for the cell ids with at
        least the attributes `grid_name` and `level`.
    grid_info : dict or DGGSInfo, optional
        Override the grid parameters on the dataset. Useful to set attributes on
        the dataset.
    name : str, default: "cell_ids"
        The name of the coordinate containing the cell ids.
    convention : str, default: "xdggs"
        The name of the metadata convention. Built-in conventions are:

        - "xdggs": the existing xdggs convention. ``name`` points to the
          coordinate containing cell ids, and which has all the grid
          metadata. The ``name`` parameter defaults to ``"cell_ids"``.
        - "cf": the upcoming CF convention standardization. While the
          convention extension is specialized on ``healpix`` for now, the
          decoder can work with other DGGS as well. For this, all metadata
          lives on a variable with a ``grid_mapping_name`` attribute, and
          the cell ids coordinate is indicated by the ``coordinates``
          attribute on data variables / other coordinates (this can be
          overridden by the ``name`` parameter).
    index_options, **index_kwargs : dict, optional
        Additional options to forward to the index.

    Returns
    -------
    decoded : xarray.DataArray or xarray.Dataset
        The input dataset with a DGGS index on the cell id coordinate.

    See Also
    --------
    xarray.Dataset.dggs.decode
    xarray.DataArray.dggs.decode
    """
    if index_options is None:
        index_options = {}

    if isinstance(convention, str):
        convention = _conventions.get(convention)
        if convention is None:
            valid_names = _conventions.keys()
            raise ValueError(
                f"unknown convention: {convention}."
                f" Choose a known convention: {', '.join(valid_names)}"
            )

    return call_on_dataset(
        partial(
            convention.decode,
            grid_info=grid_info,
            name=name,
            index_options=index_options | index_kwargs,
        ),
        obj,
    )


def detect_decoder(obj, grid_info, name):
    for name, convention in _conventions.items():
        try:
            return convention.decode(obj, grid_info=grid_info, name=name)
        except DecoderError:
            pass

    raise ValueError("cannot detect a matching convention")


@overload
def encode(
    obj: xr.DataArray, convention: str, *, encoding: dict[str, Any] | None = None
) -> xr.DataArray: ...


@overload
def encode(
    obj: xr.Dataset, convention: str, *, encoding: dict[str, Any] | None = None
) -> xr.Dataset: ...


def encode(
    obj: xr.DataArray | xr.Dataset,
    convention: str,
    *,
    encoding: dict[str, Any] | None = None,
) -> xr.DataArray | xr.Dataset:
    converter = _conventions.get(convention)
    if converter is None:
        raise ValueError(f"unknown convention: {convention}")

    return call_on_dataset(converter.encode, obj)


__all__ = ["register_convention", "detect_decoder", "DecoderError", "Convention"]
