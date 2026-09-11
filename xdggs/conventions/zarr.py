import copy
from collections.abc import Hashable
from typing import Any, ClassVar, TypedDict

import xarray as xr

from xdggs.conventions.base import Convention, translate_metadata_keys
from xdggs.conventions.errors import DecoderError
from xdggs.conventions.registry import register_convention
from xdggs.typing import TranslationTable
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


class ZarrConventionHeader(TypedDict):
    uuid: str
    schema_url: str
    spec_url: str
    name: str
    description: str


@register_convention("zarr")
class Zarr(Convention):
    uuid: ClassVar[str] = "7b255807-140c-42ca-97f6-7a1cfecdbc38"
    schema_url: ClassVar[str] = (
        "https://raw.githubusercontent.com/zarr-conventions/dggs/refs/tags/v1/schema.json"
    )
    spec_url: ClassVar[str] = (
        "https://github.com/zarr-conventions/dggs/blob/v1/README.md"
    )
    convention_metadata: ClassVar[ZarrConventionHeader] = {
        "uuid": uuid,
        "schema_url": schema_url,
        "spec_url": spec_url,
        "name": "dggs",
        "description": "Discrete Global Grid Systems convention for zarr",
    }

    translation_table: ClassVar[TranslationTable] = {
        "refinement_level": "level",
        "name": "grid_name",
        "ellipsoid": {
            "semi_major_axis": "semimajor_axis",
            "semi_minor_axis": "semiminor_axis",
        },
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
        metadata = copy.deepcopy(grid_info)

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
        compression = metadata.pop("compression", "none")
        variables_to_drop = []
        if compression != "none":
            index_options["compression"] = compression
            index_options["dim"] = spatial_dimension
            variables_to_drop.append(coordinate)

        # construct index
        translation_table = self._create_translation_table(direction="xdggs")
        metadata_ = translate_metadata_keys(metadata, translation_table)

        var = ds.variables[coordinate].copy(deep=False)
        var.attrs = metadata_

        if grid_name not in GRID_REGISTRY:
            raise DecoderError(f"cf convention: unknown grid name: {grid_name}")
        index_cls = GRID_REGISTRY[grid_name]
        index = index_cls.from_variables({name: var}, options=index_options)

        new_ds = (
            ds.assign_coords(xr.Coordinates.from_xindex(index))
            .drop_vars(variables_to_drop)
            .assign_attrs(copy.deepcopy(ds.attrs))
        )
        # remove redundant attrs
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
        if encoding is None:
            encoding = {}

        # prepare the output dataset
        index = ds.dggs.index
        coordinate = index.name

        coords = index.serialize(encoding=encoding)
        result = ds.drop_indexes(coordinate).drop_vars(coordinate).assign_coords(coords)

        # grid metadata
        translation_table = self._create_translation_table(direction="self")
        raw_grid_metadata = index.grid_info.to_dict()
        grid_metadata = translate_metadata_keys(raw_grid_metadata, translation_table)

        # additional metadata
        variables = coords.variables
        if len(variables) != 1:
            raise ValueError(
                f"expected exactly one coordinate while encoding, but got {len(variables)}."
                " This most likely a problem with the implementation of the index. Please open an issue."
            )
        coord_name, coord = next(iter(variables.items()))
        for key in raw_grid_metadata:
            coord.attrs.pop(key, None)

        compression = coord.attrs.pop("compression", "none")
        additional_metadata = {
            "coordinate": coord_name,
            "compression": compression,
            "spatial_dimension": index._dim,
        }

        # assign the metadata
        result.attrs["dggs"] = grid_metadata | additional_metadata
        conventions = result.attrs.setdefault("zarr_conventions", [])
        conventions.append(self.convention_metadata)

        return result
