import numpy as np
import pytest
from anml.getter.prior import SplinePriorGetter
from anml.prior.main import Prior
from xspline import XSpline


@pytest.fixture
def spline():
    return XSpline(knots=np.linspace(0.0, 2.0, 5), degree=3)


@pytest.fixture
def prior():
    return Prior([0.0, 1.0])


def test_prior_setter_legal(prior):
    spline_prior_getter = SplinePriorGetter(prior)
    assert spline_prior_getter.prior.mat is None


@pytest.mark.parametrize("prior", [0.0, Prior([0.0, 1.0], mat=np.ones((2, 1)))])
def test_prior_setter_legal(prior):
    with pytest.raises((TypeError, ValueError)):
        SplinePriorGetter(prior)


@pytest.mark.parametrize("size", [1.0, 1, 1.1])
def test_size_setter_legal(prior, size):
    spline_prior_getter = SplinePriorGetter(prior, size=size)
    assert isinstance(spline_prior_getter.size, int)
    assert spline_prior_getter.size == 1


@pytest.mark.parametrize("size", ["a", 0, -1])
def test_size_setter_illegal(prior, size):
    with pytest.raises(ValueError):
        SplinePriorGetter(prior, size=size)


@pytest.mark.parametrize("order", [1.0, 1, 1.1])
def test_order_setter_legal(prior, order):
    spline_prior_getter = SplinePriorGetter(prior, order=order)
    assert isinstance(spline_prior_getter.order, int)
    assert spline_prior_getter.order == 1


@pytest.mark.parametrize("order", ["a", -1])
def test_order_setter_illegal(prior, order):
    with pytest.raises(ValueError):
        SplinePriorGetter(prior, order=order)


@pytest.mark.parametrize("domain", [(0.0, 1.0), [0.0, 1.0]])
def test_domain_setter_legal(prior, domain):
    spline_prior_getter = SplinePriorGetter(prior, domain=domain)
    assert spline_prior_getter.domain == (0.0, 1.0)


@pytest.mark.parametrize("domain", [(0.0, 1.0, 2.0), [2.0, 1.0]])
def test_domain_setter_illegal(prior, domain):
    with pytest.raises(ValueError):
        SplinePriorGetter(prior, domain=domain)


@pytest.mark.parametrize("domain_type", ["rel", "abs"])
def test_domain_type_setter_legal(prior, domain_type):
    spline_prior_getter = SplinePriorGetter(prior, domain_type=domain_type)
    assert spline_prior_getter.domain_type == domain_type


@pytest.mark.parametrize("domain_type", ["other"])
def test_domain_type_setter_illegal(prior, domain_type):
    with pytest.raises(ValueError):
        SplinePriorGetter(prior, domain_type=domain_type)


@pytest.mark.parametrize("domain_type", ["rel", "abs"])
def test_get_prior_input_legal(prior, domain_type, spline):
    spline_prior_getter = SplinePriorGetter(prior, domain_type=domain_type)
    prior = spline_prior_getter.get_prior(spline)
    assert prior.mat is not None
