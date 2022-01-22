import numpy as np
import pandas as pd
import pytest
from anml.data.component import Component
from anml.prior.main import GaussianPrior, UniformPrior
from anml.variable.main import Variable


@pytest.fixture
def df():
    np.random.seed(123)
    return pd.DataFrame({"cov": np.random.randn(5)})


@pytest.fixture
def v():
    return Variable("cov")


@pytest.mark.parametrize("component", ["cov", Component("cov")])
def test_component_setter_legal(component):
    v = Variable(component)
    assert v.component.key == "cov"


@pytest.mark.parametrize("component", [1, 1.0])
def test_component_setter_illegal(component):
    with pytest.raises(TypeError):
        Variable(component)


@pytest.mark.parametrize("priors", [None,
                                    [GaussianPrior(mean=0.0, sd=1.0)],
                                    [UniformPrior(lb=0.0, ub=1.0)]])
def test_priors_setter_legal(priors):
    v = Variable("cov", priors=priors)
    if priors is None:
        assert len(v.priors) == 0
    else:
        assert len(v.priors) == 1


@pytest.mark.parametrize("priors", [[1, 2]])
def test_priors_setter_illegal(priors):
    with pytest.raises(TypeError):
        Variable("cov", priors=priors)


def test_size(v):
    assert v.size == 1


def test_attach(v, df):
    v.attach(df)
    assert np.allclose(v.component.value, df["cov"])


def test_get_design_mat(v, df):
    mat = v.get_design_mat(df)
    assert np.allclose(mat, df[["cov"]])


@pytest.mark.parametrize("priors", [None, [GaussianPrior(0.0, 1.0)]])
def test_get_direct_gaussian_prior_params(v, priors):
    v.priors = priors
    params = v.get_direct_prior_params("GaussianPrior")
    if priors is None:
        assert np.allclose(params, np.array([[0.0], [np.inf]]))
    else:
        assert np.allclose(params, np.array([[0.0], [1.0]]))


@pytest.mark.parametrize("priors", [None, [UniformPrior(0.0, 1.0)]])
def test_get_direct_uniform_prior_params(v, priors):
    v.priors = priors
    params = v.get_direct_prior_params("UniformPrior")
    if priors is None:
        assert np.allclose(params, np.array([[-np.inf], [np.inf]]))
    else:
        assert np.allclose(params, np.array([[0.0], [1.0]]))


@pytest.mark.parametrize("priors", [[GaussianPrior(0.0, 1.0, np.ones((3, 1))),
                                     GaussianPrior(0.0, 1.0, np.ones((3, 1)))]])
def test_get_linear_gaussian_prior_params(v, priors):
    v.priors = priors
    params = v.get_linear_prior_params("GaussianPrior")
    assert np.allclose(params[0], np.repeat(np.array([[0.0], [1.0]]), 6, axis=1))
    assert np.allclose(params[1], np.ones((6, 1)))


@pytest.mark.parametrize("priors", [[UniformPrior(0.0, 1.0, np.ones((3, 1))),
                                     UniformPrior(0.0, 1.0, np.ones((3, 1)))]])
def test_get_linear_uniform_prior_params(v, priors):
    v.priors = priors
    params = v.get_linear_prior_params("UniformPrior")
    assert np.allclose(params[0], np.repeat(np.array([[0.0], [1.0]]), 6, axis=1))
    assert np.allclose(params[1], np.ones((6, 1)))
