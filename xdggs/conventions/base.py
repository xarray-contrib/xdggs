from collections.abc import Hashable
from typing import Any

import xarray as xr

from xdggs.grid import DGGSInfo


class Convention:
    def decode(
        self,
        obj: xr.Dataset | xr.DataArray,
        *,
        grid_info: DGGSInfo | None,
        name: Hashable | None,
        index_options: dict[str, Any] | None,
    ) -> xr.Dataset:
        """
        Decode the dataset according to the convention.

        This takes a dataset and performs all the necessary mutations to convert
        to the xdggs convention and attach the DGGSIndex.

        Parameters
        ----------
        ds : xr.Dataset
            The encoded dataset.
        grid_info : mapping or DGGSInfo, optional
             Overrides for the grid metadata.
        name : str, optional
            The name of the cell ids coordinate.
        index_options : mapping of str to Any, optional
            Additional options for the index.

        Returns
        -------
        decoded : xr.Dataset
            The decoded dataset with a DGGSIndex.
        """
        raise NotImplementedError

    def encode(
        self, obj: xr.Dataset | xr.DataArray, *, encoding: dict[str, Any] | None = None
    ) -> xr.Dataset:
        """
        Encode according to the convention.

        This takes a dataset with a DGGSIndex and performs the necessary
        mutations to convert from the xdggs convention to the convention in
        question.

        Note that this must drop the DGGSIndex.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset to encode. Must have a DGGSIndex.
        encoding : mapping of str to Any, optional
            Additional options for the convention.

        Returns
        -------
        encoded : xr.Dataset
            The encoded dataset.
        """
        raise NotImplementedError
