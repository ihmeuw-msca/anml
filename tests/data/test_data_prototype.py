import numpy as np
import pytest
from anml.data.component import Component
from anml.data.prototype import DataPrototype
from anml.data.validator import NoNans, Positive
from pandas import DataFrame


@pytest.fixture
def df():
    np.random.seed(123)
    return DataFrame({
        "obs": np.random.randn(5),
        "obs_se": np.random.rand(5)
    })


@pytest.fixture
def components():
    return {
        "obs": Component("obs", [NoNans()]),
        "obs_se": Component("obs_se", [NoNans(), Positive()])
    }


def test_components_legal(components):
    data = DataPrototype(components)
    assert data.obs.key == "obs"
    assert data.obs_se.key == "obs_se"


@pytest.mark.parametrize("components", [{1: Component("obs")}, {"obs": "obs"}])
def test_components_illegal(components):
    with pytest.raises(TypeError):
        DataPrototype(components)
