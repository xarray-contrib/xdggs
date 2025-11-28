import hypothesis.strategies as st

from xdggs import ellipsoid

axis_sizes = st.floats(
    min_value=0.0,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
    exclude_min=True,
)
inverse_flattening = st.floats(
    min_value=5,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
    exclude_min=True,
)

names = st.none() | st.sampled_from(["WGS84", "airy", "bessel", "sphere", "unitsphere"])

_ellipsoids = st.builds(
    ellipsoid.Ellipsoid,
    semimajor_axis=axis_sizes,
    inverse_flattening=inverse_flattening,
    name=names,
)
_spheres = st.builds(ellipsoid.Sphere, radius=axis_sizes, name=names)


def ellipsoids(variant="all"):
    serialized = _ellipsoids.map(lambda x: x.to_dict()) | _spheres.map(
        lambda x: x.to_dict()
    )
    in_memory = _ellipsoids | _spheres

    variants = {
        "serialized_only": names | serialized,
        "in_memory_only": names | in_memory,
    }
    return variants.get(variant, names | serialized | in_memory)
