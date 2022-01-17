import numpy as np
import pytest
from anml.prior.main import GaussianPrior, UniformPrior
from anml.prior.utils import (combine_priors_params, filter_priors,
                              get_prior_type)


@pytest.fixture
def priors():
    return [GaussianPrior(mean=0.0, sd=1.0),
            UniformPrior(lb=0.0, ub=1.0),
            GaussianPrior(mean=0.0, sd=1.0, mat=np.identity(5)),
            UniformPrior(lb=0.0, ub=1.0, mat=np.identity(5))]


@pytest.mark.parametrize(("prior_type", "tr_prior_type"),
                         [("GaussianPrior", GaussianPrior),
                          ("UniformPrior", UniformPrior)])
def test_get_prior_type(prior_type, tr_prior_type):
    prior_type = get_prior_type(prior_type)
    assert prior_type == tr_prior_type


@pytest.mark.parametrize("prior_type", ["GaussianPrior", "UniformPrior"])
@pytest.mark.parametrize("with_mat", [True, False])
def test_filter_priors(priors, prior_type, with_mat):
    filtered_priors = filter_priors(priors, prior_type, with_mat)
    prior_type = get_prior_type(prior_type)
    assert all(isinstance(prior, prior_type) for prior in filtered_priors)
    if with_mat:
        assert all(prior.mat is not None for prior in filtered_priors)
    else:
        assert all(prior.mat is None for prior in filtered_priors)


@pytest.mark.parametrize("priors", [[], [1]])
def test_combine_prior_params_priors_illegal(priors):
    with pytest.raises((TypeError, ValueError)):
        combine_priors_params(priors)


def test_combine_prior_params_case_1():
    priors = [GaussianPrior(mean=0.0, sd=1.0)]
    params, mat = combine_priors_params(priors)
    assert np.allclose(params, np.array([[0.0], [1.0]]))
    assert mat is None


def test_combine_prior_params_case_2():
    priors = [GaussianPrior(mean=np.zeros(2), sd=np.ones(2)),
              GaussianPrior(mean=0.0, sd=1.0, mat=np.ones((3, 2)))]
    params, mat = combine_priors_params(priors)
    assert params.shape == (2, 5)
    assert mat.shape == (5, 2)
