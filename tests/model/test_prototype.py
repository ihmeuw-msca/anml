import numpy as np
import pandas as pd
import pytest
from anml.data.component import Component
from anml.data.prototype import DataPrototype
from anml.model.prototype import ModelPrototype
from anml.parameter.main import Parameter
from anml.variable.main import Variable

ModelPrototype.__abstractmethods__ = set()


@pytest.fixture
def data():
    return DataPrototype(components={
        "obs": Component("obs"),
        "obs_se": Component("obs_se")
    })


@pytest.fixture
def parameters():
    return [Parameter([Variable("cov")])]


@pytest.fixture
def df():
    return pd.DataFrame({
        "obs": np.random.randn(5),
        "obs_se": np.ones(5),
        "cov": np.random.randn(5)
    })


def test_data_setter_legal(data, parameters, df):
    model = ModelPrototype(data, parameters, df)
    assert np.allclose(model.data.obs.value, df["obs"])
    assert np.allclose(model.data.obs_se.value, df["obs_se"])


@pytest.mark.parametrize("data", [df])
def test_data_setter_illegal(data, parameters, df):
    with pytest.raises(TypeError):
        ModelPrototype(data, parameters, df)


def test_parameters_setter_legal(data, parameters, df):
    model = ModelPrototype(data, parameters, df)
    assert len(model.parameters) == 1
    assert len(model.parameters[0].variables) == 1
    assert np.allclose(model.parameters[0].variables[0].component.value, df["cov"])


@pytest.mark.parametrize("parameters", [[1]])
def test_parameters_setter_illegal(data, parameters, df):
    with pytest.raises(TypeError):
        ModelPrototype(data, parameters, df)
