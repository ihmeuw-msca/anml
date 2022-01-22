import numpy as np
import pandas as pd
import pytest
from anml.data.component import Component
from anml.parameter.main import Parameter
from anml.parameter.smoothmapping import Exp
from anml.prior.main import GaussianPrior, UniformPrior
from anml.variable.main import Variable


@pytest.fixture
def variables():
    return [Variable("cov1"), Variable("cov2")]


@pytest.fixture
def p(variables):
    return Parameter(variables=variables)


@pytest.fixture
def df():
    return pd.DataFrame({"cov1": np.random.randn(5), "cov2": np.random.randn(5)})


def test_variables_setter_legal(variables):
    p = Parameter(variables=variables)
    assert p.variables[0].component.key == "cov1"
    assert p.variables[1].component.key == "cov2"


@pytest.mark.parametrize("variables", [[1], ["a"]])
def test_variables_setter_illegal(variables):
    with pytest.raises(TypeError):
        Parameter(variables=variables)


@pytest.mark.parametrize("transform", [None, Exp()])
def test_transfrom_legal(variables, transform):
    p = Parameter(variables=variables, transform=transform)
    if transform is None:
        assert p.transform.__repr__() == "Identity()"
    else:
        assert p.transform.__repr__() == "Exp()"


@pytest.mark.parametrize("transform", [1, np.exp])
def test_transform_illegal(variables, transform):
    with pytest.raises(TypeError):
        Parameter(variables=variables, transform=transform)


@pytest.mark.parametrize("offset", ["cov", Component("cov"), None])
def test_offset_legal(variables, offset):
    p = Parameter(variables, offset=offset)
    if offset is not None:
        assert isinstance(p.offset, Component)
    else:
        assert p.offset is None


@pytest.mark.parametrize("offset", [1])
def test_offset_illegal(variables, offset):
    with pytest.raises(TypeError):
        Parameter(variables=variables, offset=offset)


@pytest.mark.parametrize("priors", [None,
                                    [GaussianPrior(mean=np.zeros(2), sd=np.ones(2))],
                                    [UniformPrior(lb=np.zeros(2), ub=np.ones(2))]])
def test_priors_setter_legal(variables, priors):
    p = Parameter(variables=variables, priors=priors)
    if priors is None:
        assert len(p.priors) == 0
    else:
        assert len(p.priors) == 1


@pytest.mark.parametrize("priors", [[1, 2]])
def test_priors_setter_illegal(variables, priors):
    with pytest.raises(TypeError):
        Parameter(variables=variables, priors=priors)


def test_size(p):
    assert p.size == 2


def test_attach(p, df):
    p.attach(df)
    assert all(v.component.value is not None for v in p.variables)
    assert p.design_mat is not None
    for prior_category in ["direct", "linear"]:
        assert prior_category in p.prior_dict
        for prior_type in ["GaussianPrior", "UniformPrior"]:
            assert prior_type in p.prior_dict[prior_category]


@pytest.mark.parametrize("prior_type", ["GaussianPrior", "UniformPrior"])
def test_get_direct_prior(p, prior_type, df):
    p.attach(df)
    p._get_direct_prior(prior_type=prior_type)
    assert p.prior_dict["direct"][prior_type].shape == (2, p.size)


@pytest.mark.parametrize("prior_type", ["GaussianPrior", "UniformPrior"])
def test_get_linear_prior(p, prior_type, df):
    p.attach(df)
    p._get_linear_prior(prior_type=prior_type)
    prior = p.prior_dict["linear"][prior_type]
    assert prior.params.shape == (2, 0)
    assert prior.mat.shape == (0, p.size)


@pytest.mark.parametrize("x", [np.ones(2)])
def test_get_params_order_0(p, x, df):
    params = p.get_params(x, df, order=0)
    assert np.allclose(params, df.sum(axis=1))


@pytest.mark.parametrize("x", [np.ones(2)])
def test_get_params_order_1(p, x, df):
    dparams = p.get_params(x, df, order=1)
    assert np.allclose(dparams, df.values)


@pytest.mark.parametrize("x", [np.ones(2)])
def test_get_params_order_2(p, x, df):
    dparams = p.get_params(x, df, order=2)
    assert np.allclose(dparams, np.zeros((df.shape[0], 2, 2)))


@pytest.mark.parametrize("x", [np.ones(2)])
def test_prior_objective(p, x, df):
    p.attach(df)
    assert p.prior_objective(x) == 0.0


@pytest.mark.parametrize("x", [np.ones(2)])
def test_prior_gradient(p, x, df):
    p.attach(df)
    assert np.allclose(p.prior_gradient(x), 0.0)


@pytest.mark.parametrize("x", [np.ones(2)])
def test_prior_hessian(p, x, df):
    p.attach(df)
    assert np.allclose(p.prior_hessian(x), 0.0)
