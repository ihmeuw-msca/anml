import numpy as np
import pandas as pd
import pytest
from anml.data.example import DataExample


@pytest.fixture
def df():
    return pd.DataFrame({"obs": np.random.randn(5),
                         "obs_se": np.ones(5)})


@pytest.mark.parametrize("obs", ["obs"])
@pytest.mark.parametrize("obs_se", ["obs_se", "random"])
def test_init(obs, obs_se, df):
    data = DataExample(obs=obs, obs_se=obs_se)
    data.attach(df)
    assert np.allclose(data.obs.value, df["obs"])
    assert np.allclose(data.obs_se.value, 1.0)
