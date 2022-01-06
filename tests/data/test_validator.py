import numpy as np
import pytest
from _pytest.monkeypatch import V
from anml.data.validator import NoNans, Positive, Unique


def test_nonans_legal():
    validator = NoNans()
    key, value = "col", np.ones(5)
    validator(key, value)


def test_nonans_illegal():
    validator = NoNans()
    key, value = "col", np.array([1.0, 1.0, np.nan])
    with pytest.raises(ValueError):
        validator(key, value)


def test_positive_legal():
    validator = Positive()
    key, value = "col", np.ones(5)
    validator(key, value)


@pytest.mark.parametrize("value", [np.zeros(5), -np.ones(5)])
def test_positive_illegal(value):
    validator = Positive()
    with pytest.raises(ValueError):
        validator("col", value)


def test_unique_legal():
    validator = Unique()
    key, value = "col", np.arange(5)
    validator(key, value)


def test_unique_illegal():
    validator = Unique()
    key, value = "col", np.ones(3)
    with pytest.raises(ValueError):
        validator(key, value)
