import pandas as pd
import numpy as np
import pytest
from dataclasses import dataclass
from typing import List

from anml.data.data_specs import DataSpecCompatibilityError
from anml.data.data_specs import DataSpecs, _check_compatible_specs


@pytest.fixture
def df():
    return pd.DataFrame({
        'obs': np.random.randn(5),
        'obs_se': np.random.randn(5),
        'group': np.arange(5)
    })


def test_compatible_data(df):
    specs = DataSpecs(
        col_obs='obs', col_obs_se='obs_se', col_groups=['group']
    )
    specs._validate_df(df)


def test_incompatible_data(df):
    with pytest.raises(DataSpecCompatibilityError):
        specs = DataSpecs(
            col_obs='obs', col_obs_se='obs_standard_error', col_groups=['group']
        )
        specs._validate_df(df)


def test_col_attributes():
    specs = DataSpecs(
        col_obs='obs', col_obs_se='obs_se', col_groups=['group']
    )
    assert sorted(specs._col_attributes) == sorted(['col_obs', 'col_obs_se', 'col_groups'])


@dataclass
class DataSpecsSubclass(DataSpecs):
    col_pop: str = None


@dataclass
class DataSpecsSubclass2(DataSpecs):
    col_pop: List[str] = None


def test_compatible_specs():
    specs1 = DataSpecs(
        col_obs='obs', col_obs_se='obs_se', col_groups=['group']
    )
    specs2 = DataSpecs(
        col_obs='obs', col_obs_se='obs_se_1', col_groups=['group']
    )
    specs3 = DataSpecsSubclass(
        col_obs='obs', col_obs_se='obs_se',
        col_groups=['group'], col_pop='population'
    )
    _check_compatible_specs([specs1, specs2])
    with pytest.raises(DataSpecCompatibilityError):
        _check_compatible_specs([specs1, specs3])


def test_incompatible_spec_types():
    specs1 = DataSpecsSubclass(
        col_obs='obs', col_obs_se='obs_se',
        col_groups=['group'], col_pop='population'
    )
    specs2 = DataSpecsSubclass2(
        col_obs='obs', col_obs_se='obs_se',
        col_groups=['group'], col_pop=['population']
    )
    with pytest.raises(DataSpecCompatibilityError):
        _check_compatible_specs([specs1, specs2])
