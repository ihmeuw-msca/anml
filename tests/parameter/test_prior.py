import pytest
import numpy as np

from anml.parameter.prior import Prior, GaussianPrior
from anml.parameter.likelihood import LikelihoodError


def test_prior():
    prior = Prior()
    assert prior.lower_bound == [-np.inf]
    assert prior.upper_bound == [np.inf]


@pytest.mark.parametrize(
    "lower,upper", [
        (-1., 1.),
        (-np.inf, 1.),
        (-1., np.inf)
    ]
)
def test_priors(lower, upper):
    prior = Prior(lower_bound=[lower], upper_bound=[upper])
    assert prior.lower_bound == [lower]
    assert prior.upper_bound == [upper]


def test_gaussian_prior():
    prior = GaussianPrior()
    assert prior.mean == [0.]
    assert prior.std == [1.]


def test_bad_gaussian_prior():
    with pytest.raises(LikelihoodError):
        GaussianPrior(std=[-1.])


