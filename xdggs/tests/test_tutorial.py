import pytest

from xdggs import tutorial


@pytest.mark.parametrize(
    ["ds_name", "grid_name"],
    (
        ("air_temperature", "h3"),
        ("air_temperature", "healpix"),
    ),
)
def test_download_from_github(tmp_path, ds_name, grid_name):
    cache_dir = tmp_path / tutorial._default_cache_dir_name
    ds = tutorial.open_dataset(ds_name, grid_name, cache_dir=cache_dir).load()

    assert cache_dir.is_dir() and len(list(cache_dir.iterdir())) == 1
    assert ds["air"].count() > 0
