import string

import hypothesis.strategies as st

from xdggs import ellipsoid

axis_sizes = st.floats(
    min_value=0,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
    exclude_min=True,
)
names = st.none() | st.text(
    alphabet=st.sampled_from(string.ascii_letters + string.digits),
    min_size=1,
)
ellipsoids = st.one_of(
    st.sampled_from(["WGS84", "airy", "bessel", "sphere", "unitsphere"]),
    st.builds(ellipsoid.Sphere, radius=axis_sizes, name=names),
    st.builds(
        ellipsoid.Ellipsoid,
        semimajor_axis=axis_sizes,
        inverse_flattening=axis_sizes,
        name=names,
    ),
)
