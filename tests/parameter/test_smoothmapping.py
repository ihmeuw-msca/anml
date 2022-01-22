from functools import partial

import numpy as np
import pytest
from anml.parameter.smoothmapping import Exp, Expit, Identity, Log, Logit


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


funs = [
    Identity(),
    Exp(),
    Log(),
    Expit(),
    Logit(),
]


@pytest.fixture
def x():
    np.random.seed(123)
    return np.random.rand(5)


@pytest.mark.parametrize("order", [-1, 3])
@pytest.mark.parametrize("fun", funs)
def test_illegal_order(order, fun, x):
    with pytest.raises(ValueError):
        fun(x, order=order)


@pytest.mark.parametrize("fun", funs)
def test_first_order_derivative(fun, x):
    d = np.diag(ad_jacobian(fun, x, out_shape=(x.size,)))
    my_d = fun(x, order=1)
    assert np.allclose(my_d, d)


@pytest.mark.parametrize("fun", funs)
def test_second_order_derivative(fun, x):
    d2 = np.diag(ad_jacobian(partial(fun, order=1), x, out_shape=(x.size,)))
    my_d2 = fun(x, order=2)
    assert np.allclose(my_d2, d2)


@pytest.mark.parametrize("fun", funs)
def test_inverse(fun, x):
    assert np.allclose(x, fun(fun.inverse(x)))
    assert np.allclose(x, fun.inverse(fun(x)))


@pytest.mark.parametrize("x", [np.array([0.0]), np.array([-1.0])])
def test_log_illegal_input(x):
    log = Log()
    with pytest.raises(ValueError):
        log(x)


@pytest.mark.parametrize("x", [np.array([-1.0]),
                               np.array([0.0]),
                               np.array([1.0]),
                               np.array([2.0])])
def test_logit_illegal_input(x):
    logit = Logit()
    with pytest.raises(ValueError):
        logit(x)
