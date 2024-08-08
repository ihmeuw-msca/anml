import numpy as np
import pytest
from anml.getter.spline import SplineGetter
from xspline import XSpline


@pytest.fixture
def data():
    np.random.seed(123)
    return np.random.randn(100)


@pytest.mark.parametrize("knots", [np.linspace(0.0, 1.0, 5)])
@pytest.mark.parametrize("degree", [3])
@pytest.mark.parametrize("ldegree", [0, 1])
@pytest.mark.parametrize("rdegree", [0, 1])
@pytest.mark.parametrize("knots_type", ["rel_domain", "rel_freq", "abs"])
def test_splinegetter(data, knots, degree, ldegree, rdegree, knots_type):
    splinegetter = SplineGetter(knots, degree, ldegree, rdegree, knots_type)
    spline = splinegetter.get_spline(data)

    assert isinstance(spline, XSpline)
    # assert spline.num_spline_bases == splinegetter.num_spline_bases
    if knots_type.startswith("rel"):
        assert np.isclose(spline.knots[0], data.min())
        assert np.isclose(spline.knots[-1], data.max())
    else:
        assert np.allclose(spline.knots, splinegetter.knots)


@pytest.mark.parametrize("knots", [np.linspace(0.0, 1.0, 5)])
@pytest.mark.parametrize("degree", [3])
@pytest.mark.parametrize("knots_type", ["abs_domain"])
def test_knots_type_setter_illegal(knots, degree, knots_type):
    with pytest.raises(ValueError):
        SplineGetter(knots, degree, knots_type=knots_type)
