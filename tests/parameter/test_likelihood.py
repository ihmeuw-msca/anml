import pytest
from scipy.stats import norm
import numpy as np

from anml.parameter.likelihood import Likelihood, LikelihoodError, GaussianLikelihood


np.random.seed(1)
# What constants were subtracted from the likelihood
# of each distribution's negative log likelihood since they
# don't influence estimation of the parameters?
GAUSSIAN_CONSTANT = 0.5 * np.log(2 * np.pi)


def test_likelihood():
    Likelihood()


def test_likelihood_error():
    with pytest.raises(LikelihoodError):
        lik = Likelihood()
        lik.get_neg_log_likelihood(vals=0.)


def test_likelihood_lik():
    with pytest.raises(NotImplementedError):
        Likelihood._likelihood(vals=0., parameters=[0.])


def test_likelihood_nll():
    with pytest.raises(NotImplementedError):
        Likelihood._neg_log_likelihood(vals=0., parameters=[0.])


@pytest.mark.parametrize("x", [-1, 0, 1])
@pytest.mark.parametrize("mean", [-1, 0, 1])
@pytest.mark.parametrize("std", np.random.rand(3) + 0.1)
def test_gaussian_likelihood(x, mean, std):
    np.testing.assert_almost_equal(
        GaussianLikelihood._likelihood(vals=[x], parameters=[[mean], [std]]),
        norm.pdf(x=x, loc=mean, scale=std)
    )
    np.testing.assert_almost_equal(
        GaussianLikelihood._neg_log_likelihood(vals=[x], parameters=[[mean], [std]]),
        -np.log(norm.pdf(x=x, loc=mean, scale=std))
    )


def test_gaussian_likelihood_array():
    np.testing.assert_array_almost_equal(
        GaussianLikelihood._neg_log_likelihood(vals=np.array([0., 1., 2.]), parameters=[[0], [1]]),
        -np.log(norm.pdf(x=np.array([0., 1., 2.])))
    )