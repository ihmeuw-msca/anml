import numpy as np
import pandas as pd
import pytest
from anml.data.example import DataExample
from anml.model.example import ModelExample
from anml.variable.main import Variable


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


@pytest.fixture
def data():
    return DataExample(obs="obs", obs_se="obs_se")


@pytest.fixture
def variables():
    return [Variable("cov0"), Variable("cov1")]


@pytest.fixture
def df():
    np.random.seed(123)
    return pd.DataFrame({
        "obs": np.random.randn(5),
        "obs_se": np.ones(5),
        "cov0": np.random.randn(5),
        "cov1": np.random.randn(5)
    })


@pytest.fixture
def model(data, variables, df):
    return ModelExample(data, variables, df)


def test_init(data, variables, df):
    model = ModelExample(data, variables, df)
    assert np.allclose(model.data.obs.value, df["obs"])
    assert np.allclose(model.data.obs_se.value, df["obs_se"])


def test_objective(model, df):
    my_value = model.objective(np.zeros(2))
    value = 0.5*np.sum((df["obs"] / df["obs_se"])**2)
    assert np.isclose(my_value, value)


@pytest.mark.parametrize("x", [np.zeros(2), np.ones(2)])
def test_gradient(model, x):
    my_value = model.gradient(x)
    value = ad_jacobian(model.objective, x)
    assert np.allclose(my_value, value)


@pytest.mark.parametrize("x", [np.zeros(2), np.ones(2)])
def test_hessian(model, x):
    my_value = model.hessian(x)
    value = ad_jacobian(model.gradient, x, out_shape=(2,))
    assert np.allclose(my_value, value)


@pytest.mark.parametrize("x", [np.zeros(2), np.ones(2)])
def test_jacobian2(model, x):
    r = model.data.obs.value - model.parameters[0].get_params(x, order=0)
    r = r / model.data.obs_se.value**2
    m = model.parameters[0].design_mat
    my_value = model.jacobian2(x)
    value = (m.T * r**2).dot(m)
    assert np.allclose(my_value, value)


def test_fit(model):
    model.fit()
    m = model.parameters[0].design_mat
    y = model.data.obs.value
    s = model.data.obs_se.value
    x = np.linalg.solve((m.T / s**2).dot(m), (m.T / s**2).dot(y))
    assert np.allclose(model.result["x"], x)


@pytest.mark.parametrize("x", [None, np.ones(2)])
@pytest.mark.parametrize("df_pred", [None,
                                     pd.DataFrame({"cov0": np.ones(3),
                                                   "cov1": np.ones(3)})])
def test_predict(model, x, df, df_pred):
    model.fit()
    prediction = model.predict(x, df_pred)
    assert len(prediction) == 1
    if x is None:
        x = model.result["x"]
    if df_pred is None:
        df_pred = df
    assert np.allclose(df_pred[["cov0", "cov1"]].values.dot(x), prediction[0])
