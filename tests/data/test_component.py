import numpy as np
import pandas as pd
import pytest
from anml.data.component import Component
from anml.data.validator import NoNans, Validator


@pytest.fixture
def df():
    np.random.seed(123)
    return pd.DataFrame({
        "col": np.random.randn(10)
    })


@pytest.mark.parametrize("key", ["col"])
def test_key_setter_legal(key):
    comp = Component(key)
    assert comp.key == key


@pytest.mark.parametrize("key", [1, 1.0, True])
def test_key_setter_illegal(key):
    with pytest.raises(TypeError):
        Component(key)


@pytest.mark.parametrize("validators", [None, [NoNans()]])
def test_validators_setter_legal(validators):
    comp = Component("col", validators)
    assert isinstance(comp.validators, list)
    assert all(isinstance(validator, Validator)
               for validator in comp.validators)


@pytest.mark.parametrize("validators", [1, [1, 2, 3]])
def test_validator_setter_illegal(validators):
    with pytest.raises(TypeError):
        Component("col", validators)


def test_attach_with_key(df):
    comp = Component("col")
    comp.attach(df)
    assert np.allclose(comp.value, df["col"])


def test_attach_no_key_with_default_value(df):
    comp = Component("intercept", default_value=1.0)
    comp.attach(df)
    assert np.allclose(comp.value, 1.0)
    assert np.allclose(df["intercept"], 1.0)


def test_attach_no_key_no_default_value(df):
    comp = Component("intercept")
    with pytest.raises(KeyError):
        comp.attach(df)


def test_clear(df):
    comp = Component("col")
    comp.attach(df)
    comp.clear()
    assert comp.value is None
