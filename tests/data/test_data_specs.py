import pandas as pd
import numpy as np
import pytest

from placeholder.data.data_specs import DataSpecCompatibilityError
from placeholder.data.data_specs import DataSpecs


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
