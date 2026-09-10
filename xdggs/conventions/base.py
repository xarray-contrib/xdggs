from collections.abc import Hashable
from typing import Any, ClassVar, Literal

import xarray as xr

from xdggs.grid import DGGSInfo
from xdggs.typing import TranslationTable


def invert_translation_table(mapping: TranslationTable) -> TranslationTable:
    return dict(
        (
            (value, key)
            if isinstance(value, str)
            else (key, invert_translation_table(value))
        )
        for key, value in mapping.items()
    )


def translate_metadata_keys(mapping: dict[str, Any], table: TranslationTable):
    def _translate(key, value, table):
        replacement = table.get(key, key)
        if isinstance(replacement, str):
            return replacement, value

        renamed_object = {
            _translate(subkey, subvalue, replacement)
            for subkey, subvalue in value.items()
        }
        return key, renamed_object

    return dict(_translate(key, value, table) for key, value in mapping.items())


class Convention:
    translation_table: ClassVar[TranslationTable]

    def _create_translation_table(
        self, direction: Literal["xdggs", "self"]
    ) -> TranslationTable:
        match direction:
            case "xdggs":
                return self.translation_table
            case "self":
                return invert_translation_table(self.translation_table)
            case _:
                raise ValueError(f"unknown direction: {direction}")

    def decode(
        self,
        obj: xr.Dataset,
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
        ds : xarray.Dataset
            The encoded dataset.
        grid_info : mapping or xdggs.DGGSInfo, optional
             Overrides for the grid metadata.
        name : str, optional
            The name of the cell ids coordinate.
        index_options : mapping of str to Any, optional
            Additional options for the index.

        Returns
        -------
        decoded : xarray.Dataset
            The decoded dataset with a DGGSIndex.
        """
        raise NotImplementedError

    def encode(
        self, obj: xr.Dataset, *, encoding: dict[str, Any] | None = None
    ) -> xr.Dataset:
        """
        Encode according to the convention.

        This takes a dataset with a DGGSIndex and performs the necessary
        mutations to convert from the xdggs convention to the convention in
        question.

        Note that this must drop the DGGSIndex.

        Parameters
        ----------
        ds : xarray.Dataset
            The dataset to encode. Must have a DGGSIndex.
        encoding : mapping of str to Any, optional
            Additional options for the convention.

        Returns
        -------
        encoded : xarray.Dataset
            The encoded dataset.
        """
        raise NotImplementedError
