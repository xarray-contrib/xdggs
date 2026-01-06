# Conventions

Different communities and file formats have different ways of encoding the metadata for discrete global grid systems.

The built-in format (called the `"xdggs"` convention) is to put all metadata on the coordinate containing the cell indices. When decoding, this metadata is used to construct a in-memory xarray index and removed from the coordinate:

```{jupyter-execute}
import xdggs
import xarray as xr

xdggs_encoded = xdggs.tutorial.open_dataset("air_temperature", "h3").load()
display(xdggs_encoded)
decoded = xdggs_encoded.dggs.decode()
display(decoded)
```

## Built-in conventions

`xdggs` comes with support for a few external conventions:

- `"cf"`: the convention for Healpix added in version 1.13 of the CF conventions
- `"zarr"`: the zarr `dggs` convention

### CF convention

The included convention object supports a generalized version of the [healpix grid mapping](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.13/cf-conventions.html#healpix) added in version 1.13 of the CF conventions. Thus, it applies the same encoding scheme to other DGGS like H3.

To use it, pass `convention="cf"` to the `decode` / `encode` methods:

```{jupyter-execute}
cf_encoded = decoded.dggs.encode(convention="cf")
display(cf_encoded)
cf_decoded = cf_encoded.dggs.decode(convention="cf")

xr.testing.assert_identical(cf_decoded, decoded)
```

### zarr convention

The [zarr `dggs` convention](https://github.com/zarr-conventions/dggs) makes use of the nesting allowed by zarr to encode all grid metadata into a single metadata object in the attributes.

To use it, pass `convention="zarr"` to the `decode` / `encode` methods:

```{jupyter-execute}
zarr_encoded = decoded.dggs.encode(convention="zarr")
display(zarr_encoded)
zarr_decoded = zarr_encoded.dggs.decode(convention="zarr")

xr.testing.assert_identical(zarr_decoded, decoded)
```

## Registering a custom convention

Conventions are defined as an object inheriting from {py:class}`xdggs.conventions.Convention`. It must define two methods:

- {py:meth}`xdggs.conventions.Convention.decode` for decoding into the in-memory structure
- {py:meth}`xdggs.conventions.Convention.encode` for encoding the in-memory format to the given convention

For example:

```{jupyter-execute}
import xdggs
from collections.abc import Hashable
from typing import Any
from xdggs.grid import DGGSInfo


@xdggs.conventions.register_convention("my-convention")
class MyConvention(xdggs.conventions.Convention):
    def decode(
        self,
        obj: xr.Dataset,
        *,
        grid_info: dict[str, Any] | DGGSInfo | None,
        name: Hashable | None,
        index_options: dict[str, Any] | None,
    ) -> xr.Dataset:
        # decode
        pass

    def encode(
        self, obj: xr.Dataset, *, encoding: dict[str, Any] | None = None
    ) -> xr.Dataset:
        # encode
        pass
```
