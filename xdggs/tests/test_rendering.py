import numpy as np
import pytest
import xarray as xr

import xdggs
from xdggs import rendering


def healpix_dataarray(level=2, cell_ids=None, ellipsoid="WGS84"):
    if cell_ids is None:
        cell_ids = np.arange(12 * 4**level, dtype="uint64")

    ds = xr.Dataset(
        coords={
            "cell_ids": (
                "cells",
                cell_ids,
                {
                    "grid_name": "healpix",
                    "level": level,
                    "indexing_scheme": "nested",
                    "ellipsoid": ellipsoid,
                },
            )
        }
    ).dggs.decode()
    lon, lat = ds.dggs.grid_info.cell_ids2geographic(ds["cell_ids"].data)

    return xr.DataArray(
        np.sin(np.radians(lat)) * np.cos(np.radians(lon)),
        dims="cells",
        coords=ds.coords,
        name="data",
        attrs={"units": "K"},
    )


def assert_same_cells(actual, expected):
    # decoding normalizes the ellipsoid to an object, so compare the parameters
    assert actual.dggs.grid_info.to_dict() == expected.dggs.grid_info.to_dict()
    xr.testing.assert_identical(
        actual.sortby("cell_ids").drop_indexes("cell_ids"),
        expected.sortby("cell_ids").drop_indexes("cell_ids"),
    )


class FakeBackend:
    """map a 2 x 3 raster onto fixed cell ids"""

    pixel_cell_ids = np.ma.masked_array(
        np.array([[0, 1, 2], [3, 7, 0]], dtype="uint64"),
        mask=[[False, False, False], [False, False, True]],
    )

    def sample(self, grid_info, cell_ids, sampling_grid):
        template = xr.DataArray(
            np.zeros((2, 3), dtype=bool),
            dims=("y", "x"),
            coords={"y": [1.5, 0.5], "x": [0.5, 1.5, 2.5]},
            attrs={"crs_wkt": "fake"},
        )
        return self.pixel_cell_ids, template

    def raster_cell_ids(self, grid_info, raster):
        return self.pixel_cell_ids

    def plot(self, grid_info, cell_ids, data, **kwargs):
        return cell_ids, data, kwargs


@pytest.fixture
def fake_backend(monkeypatch):
    monkeypatch.setitem(rendering._backends, xdggs.HealpixInfo, FakeBackend)


class TestBackends:
    def test_unsupported_grid(self):
        grid_info = xdggs.H3Info(level=2)

        with pytest.raises(NotImplementedError, match="H3Info"):
            rendering.get_rendering_backend(grid_info)

    def test_register(self, monkeypatch):
        class Backend:
            pass

        monkeypatch.setattr(rendering, "_backends", {})
        rendering.register_rendering_backend(xdggs.H3Info)(Backend)

        assert isinstance(
            rendering.get_rendering_backend(xdggs.H3Info(level=2)), Backend
        )


@pytest.mark.usefixtures("fake_backend")
class TestToRasterGeneric:
    def test_dataarray(self):
        arr = healpix_dataarray(level=0, cell_ids=np.arange(5, dtype="uint64"))

        actual = arr.dggs.to_raster()

        expected_values = np.array([arr.data[[0, 1, 2]], [arr.data[3], np.nan, np.nan]])
        assert actual.dims == ("y", "x")
        np.testing.assert_equal(actual.data, expected_values)
        assert actual.attrs == {"crs_wkt": "fake", "units": "K"}
        assert "cell_ids" not in actual.coords
        np.testing.assert_equal(actual["x"].data, [0.5, 1.5, 2.5])

    def test_additional_dims(self):
        arr = healpix_dataarray(level=0, cell_ids=np.arange(5, dtype="uint64"))
        stacked = xr.concat([arr, 2 * arr], dim="time").assign_coords(
            time=[1, 2], latitude=("cells", np.arange(5))
        )

        actual = stacked.dggs.to_raster()

        assert actual.dims == ("time", "y", "x")
        np.testing.assert_equal(actual.isel(time=1).data, 2 * actual.isel(time=0).data)
        assert "latitude" not in actual.coords
        np.testing.assert_equal(actual["time"].data, [1, 2])

    def test_dataset(self):
        arr = healpix_dataarray(level=0, cell_ids=np.arange(5, dtype="uint64"))
        ds = xr.Dataset({"a": arr, "b": 2 * arr, "scalar": 1})

        actual = ds.dggs.to_raster()

        assert actual["a"].dims == ("y", "x")
        np.testing.assert_equal(actual["b"].data, 2 * actual["a"].data)
        assert actual["scalar"].dims == ()

    def test_xy_dims(self):
        arr = healpix_dataarray(level=0).expand_dims("x")

        with pytest.raises(ValueError, match="'x' or 'y'"):
            arr.dggs.to_raster()

    def test_interpolation(self):
        arr = healpix_dataarray(level=0)

        with pytest.raises(ValueError, match="interpolation"):
            arr.dggs.to_raster(interpolation="bilinear")


@pytest.mark.usefixtures("fake_backend")
class TestPlotGeneric:
    def test_forwards_arguments(self):
        arr = healpix_dataarray(level=0)

        cell_ids, data, kwargs = arr.dggs.plot(projection="Robinson", cmap="RdBu")

        np.testing.assert_equal(cell_ids, arr["cell_ids"].data)
        np.testing.assert_equal(data, arr.data)
        assert kwargs == {
            "projection": "Robinson",
            "sampling_grid": None,
            "ax": None,
            "cmap": "RdBu",
        }

    def test_dataset(self):
        ds = healpix_dataarray(level=0).to_dataset()

        with pytest.raises(TypeError, match="DataArray"):
            ds.dggs.plot()

    def test_multiple_dims(self):
        arr = healpix_dataarray(level=0).expand_dims(time=2)

        with pytest.raises(ValueError, match="1D"):
            arr.dggs.plot()


class TestHealpixPlot:
    @pytest.fixture(autouse=True)
    def requires_healpix_plot(self):
        pytest.importorskip("healpix_plot.raster")

    @pytest.mark.parametrize("ellipsoid", ["sphere", "WGS84"])
    def test_native_roundtrip(self, ellipsoid):
        pytest.importorskip("rioxarray")

        arr = healpix_dataarray(level=3, ellipsoid=ellipsoid)

        raster = arr.dggs.to_raster()
        actual = xdggs.from_raster(raster, arr.dggs.grid_info)

        assert raster.dims == ("y", "x")
        assert raster.rio.crs.to_dict()["proj"] == "healpix"
        assert int(raster.notnull().sum()) == arr.size
        # the rest of the bounding box is NaN
        assert_same_cells(actual.dropna("cells"), arr)

    def test_native_regional(self):
        pytest.importorskip("rioxarray")

        cell_ids = np.arange(4**3, 2 * 4**3, dtype="uint64")
        arr = healpix_dataarray(level=3, cell_ids=cell_ids)

        raster = arr.dggs.to_raster()
        actual = xdggs.from_raster(raster, arr.dggs.grid_info).dropna("cells")

        # one base cell: an 8 x 8 square of pixels
        assert raster.shape == (8, 8)
        assert_same_cells(actual, arr)

    def test_native_level_0(self):
        arr = healpix_dataarray(level=0)

        with pytest.raises(ValueError, match="level 1"):
            arr.dggs.to_raster()

    def test_latlon_grid(self):
        arr = healpix_dataarray(level=4)

        raster = arr.dggs.to_raster({"shape": (36, 18)})

        assert raster.sizes == {"y": 18, "x": 36}
        assert not raster.isnull().any()
        assert "healpix_crs_wkt" in raster.attrs

    def test_geotiff(self, tmp_path):
        rioxarray = pytest.importorskip("rioxarray")
        pytest.importorskip("rasterio")

        arr = healpix_dataarray(level=4, cell_ids=np.arange(256, 512, dtype="uint64"))
        path = tmp_path / "data.tif"

        arr.dggs.to_raster().rio.to_raster(path)
        # GeoTIFF cannot store +proj=healpix, the CRS is in the side-car or a tag
        for sidecar in tmp_path.glob("*.aux.xml"):
            sidecar.unlink()
        raster = rioxarray.open_rasterio(path).squeeze("band", drop=True)
        actual = xdggs.from_raster(raster, arr.dggs.grid_info).dropna("cells")

        np.testing.assert_equal(np.sort(actual["cell_ids"].data), arr["cell_ids"].data)
        np.testing.assert_allclose(actual.sortby("cell_ids").data, arr.data)

    def test_from_raster_too_fine(self):
        pytest.importorskip("rioxarray")

        arr = healpix_dataarray(level=1)
        raster = arr.dggs.to_raster({"shape": (72, 36)})

        with pytest.raises(ValueError, match="more than one pixel"):
            xdggs.from_raster(raster, arr.dggs.grid_info)

    @pytest.mark.parametrize(
        ["projection", "expected_type"],
        (
            # the native grid: the cells drawn as a mesh
            ("HEALPix", "QuadMesh"),
            # resampled to a longitude / latitude image
            ("Mollweide", "AxesImage"),
        ),
    )
    def test_plot(self, projection, expected_type):
        pytest.importorskip("cartopy")
        plt = pytest.importorskip("matplotlib.pyplot")
        import matplotlib.collections
        import matplotlib.image

        types = {
            "QuadMesh": matplotlib.collections.QuadMesh,
            "AxesImage": matplotlib.image.AxesImage,
        }
        arr = healpix_dataarray(level=2)

        mappable = arr.dggs.plot(projection=projection, colorbar=True)

        assert isinstance(mappable, types[expected_type])
        plt.close("all")
