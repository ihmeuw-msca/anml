import pandas as pd
import numpy as np
import pytest

from placeholder.data.data import Data, DataTypeError
from placeholder.data.data_specs import DataSpecs


@pytest.fixture
def df():
    return pd.DataFrame({
        'obs': np.random.randn(5),
        'obs_se': np.random.randn(5),
        'group': np.arange(5)
    })


def test_attach_detach_specs():

    d = Data()
    assert isinstance(d._data_specs, list)
    assert len(d._data_specs) == 0

    specs = DataSpecs(
        col_obs='obs', col_obs_se='obs_se', col_groups=['group']
    )

    d.set_data_specs(specs)
    assert specs == d._data_specs[0]

    d.detach_data_specs()
    assert len(d._data_specs) == 0


def test_validate_data(df):

    d = Data()
    specs = DataSpecs(
        col_obs='obs', col_obs_se='obs_se', col_groups=['group']
    )
    d.set_data_specs(specs)

    d.process_data(df)


def test_data_type():

    with pytest.raises(DataTypeError):
        d = Data()
        d.process_data(np.arange(10))
