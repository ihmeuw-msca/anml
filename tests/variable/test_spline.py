import numpy as np
import pandas as pd
import pytest
from anml.getter.prior import SplinePriorGetter
from anml.getter.spline import SplineGetter
from anml.prior.main import UniformPrior
from anml.variable.spline import SplineVariable
from xspline import XSpline


@pytest.fixture
def data():
    return pd.DataFrame({"cov": np.random.randn(5)})


@pytest.mark.parametrize("spline",
                         [XSpline(knots=np.array([0.0, 0.5, 1.0]), degree=3),
                          SplineGetter(knots=np.array([0.0, 0.5, 1.0]), degree=3)])
def test_spline_setter_legal(spline):
    sv = SplineVariable("cov", spline=spline)
    assert sv.spline.degree == 3


@pytest.mark.parametrize("spline", [1])
def test_spline_setter_illegal(spline):
    with pytest.raises(TypeError):
        SplineVariable("cov", spline=spline)


@pytest.mark.parametrize("knots", [np.linspace(0.0, 1.0, 5)])
@pytest.mark.parametrize("degree", [3])
@pytest.mark.parametrize("l_linear", [True, False])
@pytest.mark.parametrize("r_linear", [True, False])
@pytest.mark.parametrize("include_first_basis", [True, False])
def test_spline_size(knots, degree, l_linear, r_linear, include_first_basis):
    splinegetter = SplineGetter(knots, degree, l_linear, r_linear, include_first_basis)
    spline = XSpline(knots, degree, l_linear, r_linear, include_first_basis)
    sv = SplineVariable("cov", spline=splinegetter)

    assert sv.size == spline.num_spline_bases


@pytest.mark.parametrize("spline", [SplineGetter(knots=np.array([0.0, 0.5, 1.0]), degree=3)])
@pytest.mark.parametrize("priors", [[SplinePriorGetter(UniformPrior(0.0, 1.0), order=1)]])
def test_attach(spline, priors, data):
    sv = SplineVariable("cov", spline, priors=priors)
    sv.attach(data)
    assert isinstance(sv.spline, XSpline)
    assert isinstance(sv.priors[0], UniformPrior)


@pytest.mark.parametrize("spline", [SplineGetter(knots=np.array([0.0, 0.5, 1.0]), degree=3)])
def test_get_design_mat(data, spline):
    sv = SplineVariable("cov", spline=spline)
    design_mat = sv.get_design_mat(data)
    assert design_mat.shape == (data.shape[0], sv.size)
