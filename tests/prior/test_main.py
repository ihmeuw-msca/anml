import numpy as np
import pytest
from anml.prior.main import GaussianPrior, Prior, UniformPrior


def ad_jacobian(fun, x, out_shape=(), eps=1e-10):
    c = x + 0j
    g = np.zeros((*out_shape, *x.shape))
    if len(out_shape) == 0:
        for i in np.ndindex(x.shape):
            c[i] += eps*1j
            g[i] = fun(c).imag/eps
            c[i] -= eps*1j
    else:
        for j in np.ndindex(out_shape):
            for i in np.ndindex(x.shape):
                c[i] += eps*1j
                g[j][i] = fun(c)[j].imag/eps
                c[i] -= eps*1j
    return g


@pytest.mark.parametrize("params", [[1.0, np.ones(2)], [np.ones(2), np.ones(2)]])
def test_params_setter_legal(params):
    prior = Prior(params)
    assert prior.params.shape == (2, 2)


@pytest.mark.parametrize("params", [[np.ones(3), np.ones(2)]])
def test_params_setter_illegal(params):
    with pytest.raises(ValueError):
        Prior(params)


@pytest.mark.parametrize("params", [[1.0, np.ones(3)]])
@pytest.mark.parametrize("mat", [None, np.ones((3, 2))])
def test_mat_setter_legal(params, mat):
    prior = Prior(params, mat)
    if mat is None:
        assert prior.shape == (3, 3)
    else:
        assert prior.shape == (3, 2)


@pytest.mark.parametrize("params", [[1.0, np.ones(3)]])
@pytest.mark.parametrize("mat", [[], np.ones((2, 3))])
def test_mat_setter_illegal(params, mat):
    with pytest.raises(ValueError):
        Prior(params, mat)


@pytest.mark.parametrize("mean", [1.0, [1.0]])
@pytest.mark.parametrize("sd", [1.0, [1.0]])
def test_gaussian_prior_init_legal(mean, sd):
    prior = GaussianPrior(mean=mean, sd=sd)
    assert np.allclose(prior.mean, 1.0)
    assert np.allclose(prior.sd, 1.0)


@pytest.mark.parametrize("mean", [1.0])
@pytest.mark.parametrize("sd", [-1.0, 0.0])
def test_gaussian_prior_init_illegal(mean, sd):
    with pytest.raises(ValueError):
        GaussianPrior(mean=mean, sd=sd)


@pytest.mark.parametrize("x", [np.array([1.0]), np.array([2.0])])
def test_gaussian_prior_objective(x):
    mean, sd = 2.0, 0.5
    prior = GaussianPrior(mean, sd)
    assert prior.objective(x) == 0.5*np.sum(((x - mean) / sd)**2)


@pytest.mark.parametrize("x", [np.array([1.0]), np.array([2.0])])
@pytest.mark.parametrize("mat", [None, np.ones((2, 1))])
def test_gaussian_prior_gradient(x, mat):
    mean, sd = 2.0, 0.5
    prior = GaussianPrior(mean, sd, mat=mat)
    assert np.allclose(prior.gradient(x), ad_jacobian(prior.objective, x))


@pytest.mark.parametrize("x", [np.array([1.0]), np.array([2.0])])
@pytest.mark.parametrize("mat", [None, np.ones((2, 1))])
def test_gaussian_prior_hessian(x, mat):
    mean, sd = 2.0, 0.5
    prior = GaussianPrior(mean, sd, mat=mat)
    assert np.allclose(prior.hessian(x), ad_jacobian(prior.gradient, x, out_shape=(1,)))


@pytest.mark.parametrize("lb", [1.0, [1.0]])
@pytest.mark.parametrize("ub", [1.0, [1.0]])
def test_uniform_prior_init_legal(lb, ub):
    prior = UniformPrior(lb, ub)
    assert np.allclose(prior.lb, 1.0)
    assert np.allclose(prior.ub, 1.0)


@pytest.mark.parametrize("lb", [1.0])
@pytest.mark.parametrize("ub", [-1.0])
def test_uniform_prior_init_illegal(lb, ub):
    with pytest.raises(ValueError):
        UniformPrior(lb, ub)


def test_uniform_prior_objective():
    prior = UniformPrior(lb=0.0, ub=5.0)
    with pytest.raises(NotImplementedError):
        prior.objective(None)


def test_uniform_prior_gradient():
    prior = UniformPrior(lb=0.0, ub=5.0)
    with pytest.raises(NotImplementedError):
        prior.gradient(None)


def test_uniform_prior_hessian():
    prior = UniformPrior(lb=0.0, ub=5.0)
    with pytest.raises(NotImplementedError):
        prior.hessian(None)
