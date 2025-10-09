import numpy.typing as npt
import xarray as xr

from xdggs import conventions
from xdggs.grid import DGGSInfo
from xdggs.index import DGGSIndex
from xdggs.plotting import explore


@xr.register_dataset_accessor("dggs")
@xr.register_dataarray_accessor("dggs")
class DGGSAccessor:
    _obj: xr.Dataset | xr.DataArray
    _index: DGGSIndex | None
    _name: str

    def __init__(self, obj: xr.Dataset | xr.DataArray):
        self._obj = obj

        index = None
        name = ""
        for k, idx in obj.xindexes.items():
            if isinstance(idx, DGGSIndex):
                if index is not None:
                    raise ValueError(
                        "Only one DGGSIndex per dataset or dataarray is supported"
                    )
                index = idx
                name = k
        self._name = name
        self._index = index

    def decode(
        self,
        grid_info=None,
        *,
        name="cell_ids",
        convention="xdggs",
        index_options=None,
        **index_kwargs,
    ) -> xr.Dataset | xr.DataArray:
        """decode the DGGS cell ids

        Parameters
        ----------
        grid_info : dict or DGGSInfo, optional
            Override the grid parameters on the dataset. Useful to set attributes on
            the dataset.
        name : str, optional
            The name of the coordinate containing the cell ids. The default name
            depends on the convention.
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
        obj : xarray.DataArray or xarray.Dataset
            The object with a DGGS index on the cell id coordinate.
        """
        if callable(convention):
            decoder = convention
        else:
            decoder = conventions._decoders.get(convention)
            if decoder is None:
                valid_names = conventions._decoders.keys()
                raise ValueError(
                    f"unknown convention: {convention}."
                    f" Choose a known convention: {', '.join(valid_names)}"
                )

        if index_options is None:
            index_options = {}

        coords = decoder(
            self._obj, grid_info=grid_info, name=name, index_options=index_options
        )
        return self._obj.assign_coords(coords)

    @property
    def index(self) -> DGGSIndex:
        """The DGGSIndex instance for this Dataset or DataArray.

        Raises
        ------
        ValueError
            if no DGGSIndex can be found
        """
        if self._index is None:
            raise ValueError("no DGGSIndex found on this Dataset or DataArray")
        return self._index

    @property
    def coord(self) -> xr.DataArray:
        """The indexed DGGS (cell ids) coordinate as a DataArray.

        Raises
        ------
        ValueError
            if no such coordinate is found on the Dataset / DataArray
        """
        if not self._name:
            raise ValueError(
                "no coordinate with a DGGSIndex found on this Dataset or DataArray"
            )
        return self._obj[self._name]

    @property
    def params(self) -> dict:
        """The grid parameters after normalization."""
        return self.index.grid.to_dict()

    @property
    def grid_info(self) -> DGGSInfo:
        """The grid info object containing the DGGS type and its parameters.

        Returns
        -------
        xdggs.DGGSInfo
        """
        return self.index.grid_info

    def sel_latlon(
        self, latitude: npt.ArrayLike, longitude: npt.ArrayLike
    ) -> xr.Dataset | xr.DataArray:
        """Select grid cells from latitude/longitude data.

        Parameters
        ----------
        latitude : array-like
            Latitude coordinates (degrees).
        longitude : array-like
            Longitude coordinates (degrees).

        Returns
        -------
        subset
            A new :py:class:`xarray.Dataset` or :py:class:`xarray.DataArray`
            with all cells that contain the input latitude/longitude data points.
        """
        cell_indexers = {
            self._name: self.grid_info.geographic2cell_ids(lon=longitude, lat=latitude)
        }
        return self._obj.sel(cell_indexers)

    def assign_latlon_coords(self) -> xr.Dataset | xr.DataArray:
        """Return a new Dataset or DataArray with new "latitude" and "longitude"
        coordinates representing the grid cell centers."""

        lon_data, lat_data = self.index.cell_centers()

        return self._obj.assign_coords(
            latitude=(self.index._dim, lat_data),
            longitude=(self.index._dim, lon_data),
        )

    @property
    def cell_ids(self):
        """The indexed DGGS (cell ids) coordinate as a DataArray.

        Alias of ``coord``.

        Raises
        ------
        ValueError
            if no such coordinate is found on the Dataset / DataArray
        """
        return self.coord

    def cell_centers(self):
        """derive geographic cell center coordinates

        Returns
        -------
        coords : xarray.Dataset
            Dataset containing the cell centers in geographic coordinates.
        """
        lon_data, lat_data = self.index.cell_centers()

        return xr.Dataset(
            coords={
                "latitude": (self.index._dim, lat_data),
                "longitude": (self.index._dim, lon_data),
            }
        )

    def cell_boundaries(self):
        """derive cell boundary polygons

        Returns
        -------
        boundaries : xarray.DataArray
            The cell boundaries as shapely objects.
        """
        boundaries = self.index.cell_boundaries()

        return xr.DataArray(
            boundaries, coords={self._name: self.cell_ids}, dims=self.cell_ids.dims
        )

    def zoom_to(self, level: int):
        """Change the refinement level of the cell ids to `level`.

        Parameters
        ----------
        level : int
            The refinement level to change to. Can be smaller than the dataset's
            level to compute parents, or bigger to fetch the children. In the
            latter case, the array will have an additional `"children"`
            dimension.

        Returns
        -------
        zoomed : xr.DataArray
            The children or parents of the current cells.
        """
        zoomed = self.index.zoom_to(level=level)

        if zoomed.ndim == 1:
            dims = self.cell_ids.dims
        else:
            dims = [*self.cell_ids.dims, "children"]

        return xr.DataArray(zoomed, coords={self._name: self.cell_ids}, dims=dims)

    def explore(self, *, cmap="viridis", center=None, alpha=None, coords=None):
        """interactively explore the data using `lonboard`

        Requires `lonboard`, `matplotlib`, and `arro3.core` to be installed.

        Parameters
        ----------
        cmap : str
            The name of the color map to use
        center : int or float, optional
            If set, will use this as the center value of a diverging color map.
        alpha : float, optional
            If set, controls the transparency of the polygons.
        coords : list of str, default: ["latitude", "longitude"]
            Additional coordinates to contain in the table of contents.

        Returns
        -------
        map : lonboard.Map
            The rendered map.

        Notes
        -----
        Plotting currently is restricted to 1D `DataArray` objects.
        """
        if isinstance(self._obj, xr.Dataset):
            raise ValueError("does not work with Dataset objects, yet")

        return explore(
            self._obj,
            cmap=cmap,
            center=center,
            alpha=alpha,
            coords=coords,
        )

    def as_convention(self, convention: str):
        """Convert the dataset to a specific convention

        Parameters
        ----------
        convention : str
            The name of the convention. Supported are:
            - "easygems": ``grid_mapping`` coordinate and ``cell`` dimension and ``cell`` coordinate with a `pandas` index.
            - "cf": ``grid_mapping`` coordinate with ``cell_index`` coordinate and ``cell`` dimension.
            - "xdggs": ``cell_ids`` coordinate with grid metadata and a ``cells`` coordinate.

        Returns
        -------
        obj : xr.DataArray or xr.Dataset
            The object converted to the given dimension.
        """
        converter = conventions._encoders.get(convention)
        if converter is None:
            raise ValueError(f"unknown convention: {convention}")

        return converter(self._obj)
