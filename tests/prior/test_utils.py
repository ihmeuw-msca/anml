import numpy as np
import pytest
from anml.prior.main import GaussianPrior, UniformPrior
from anml.prior.utils import filter_priors, get_prior_type


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
