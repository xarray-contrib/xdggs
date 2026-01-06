from collections.abc import Hashable
from typing import Any, Literal

import xarray as xr

from xdggs.conventions.base import Convention
from xdggs.conventions.errors import DecoderError
from xdggs.conventions.registry import register_convention
from xdggs.utils import GRID_REGISTRY


def extract_convention_declaration(
    conventions: list[dict[str, Any]],
    uuid,
    schema_url,
    spec_url,
) -> dict[str, Any] | None:
    for convention in conventions:
        if (
            convention.get("uuid") == uuid
            or convention.get("schema_url") == schema_url
            or convention.get("spec_url") == spec_url
        ):
            return convention

    return None


@register_convention("zarr")
class Zarr(Convention):
    uuid = "7b255807-140c-42ca-97f6-7a1cfecdbc38"
    schema_url = "https://raw.githubusercontent.com/zarr-conventions/dggs/refs/tags/v1/schema.json"
    spec_url = "https://github.com/zarr-conventions/dggs/blob/v1/README.md"
    convention_metadata = {
        "uuid": uuid,
        "schema_url": schema_url,
        "spec_url": spec_url,
        "name": "dggs",
        "description": "Discrete Global Grid Systems convention for zarr",
    }

    def translate_metadata(
        self,
        metadata: dict[str, Any],
        direction: Literal["forward", "inverse"] = "forward",
    ) -> dict[str, Any]:
        key_translations = {
            "name": "grid_name",
            "refinement_level": "level",
        }
        if direction == "inverse":
            key_translations = {v: k for k, v in key_translations.items()}

        return {
            key_translations.get(key, key): value for key, value in metadata.items()
        }

    def decode(
        self,
        ds: xr.Dataset,
        *,
        grid_info: dict[str, Any] | None,
        name: Hashable | None,
        index_options: dict[str, Any] | None,
    ) -> xr.Dataset:
        # steps:
        # - find zarr conventions metadata (uuid, schema_url, spec_url)
        # - extract metadata object
        zarr_conventions = ds.attrs.get("zarr_conventions", [])
        convention = extract_convention_declaration(
            zarr_conventions, self.uuid, self.schema_url, self.spec_url
        )
        convention_index = (
            zarr_conventions.index(convention) if convention is not None else None
        )

        if grid_info is None:
            if convention is None:
                raise DecoderError(
                    "The zarr dggs convention was not declared. Aborting parsing"
                )

            grid_info = ds.attrs.get("dggs")
            if grid_info is None:
                raise DecoderError(
                    "No metadata found. Please make sure the dataset follows"
                    " the zarr dggs convention or pass a convention metadata"
                    " object to the `grid_info` parameter."
                )
        # copy to avoid mutating
        metadata = dict(grid_info)

        # information:
        # - "name" is the grid name
        # - "coordinate" if provided is the `name` (must be provided if missing in the metadata)
        # - "spatial_dimension" must be provided

        # required
        grid_name = metadata.pop("name", None)
        if grid_name is None:
            raise DecoderError("Required field `name` is missing or null.")

        spatial_dimension = metadata.pop("spatial_dimension", None)
        if spatial_dimension is None:
            raise DecoderError("Required field `spatial_dimension` is missing or null.")

        # optional, but required for now
        coordinate = metadata.pop("coordinate", None)
        if coordinate is None:
            raise NotImplementedError("missing coordinate is not supported for now")

        # optional, but required to be `"none"` for now
        compression = metadata.pop("compression", None)
        if compression != "none":
            raise NotImplementedError(
                "compressed coordinates are not supported for now"
            )

        # construct index
        metadata_ = self.translate_metadata(metadata)

        var = ds.variables[coordinate].copy(deep=False)
        var.attrs = metadata_

        if grid_name not in GRID_REGISTRY:
            raise DecoderError(f"cf convention: unknown grid name: {grid_name}")
        index_cls = GRID_REGISTRY[grid_name]
        index = index_cls.from_variables({name: var}, options=index_options)

        new_ds = ds.copy(deep=False).assign_coords(xr.Coordinates.from_xindex(index))
        new_ds.attrs.pop("dggs", None)
        if convention_index is not None:
            zarr_conventions = new_ds.attrs.get("zarr_conventions", [])
            del zarr_conventions[convention_index]
            if not zarr_conventions:
                del new_ds.attrs["zarr_conventions"]
        return new_ds

    def encode(
        self, ds: xr.Dataset, *, encoding: dict[str, Any] | None = None
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
        grid_info = self.translate_metadata(
            ds.dggs.grid_info.to_dict(), direction="inverse"
        )

        # encoding contains:
        # - compression type (ignored for now)

        # additional keys:
        # - coordinate
        # - spatial_dimension
        # - compression

        coordinate = ds.dggs.index._name

        additional_metadata = {
            "spatial_dimension": ds.dggs.index._dim,
            "coordinate": coordinate,
            "compression": "none",
        }

        result = ds.drop_indexes(coordinate)
        result.attrs["dggs"] = grid_info | additional_metadata
        conventions = result.attrs.setdefault("zarr_conventions", [])
        conventions.append(self.convention_metadata)

        return result
