from typing import Literal

Compression = Literal["none", "compacted", "ranges"]
TranslationTable = dict[str, str | dict[str, str]]
