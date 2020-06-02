import pandas as pd
import numpy as np
import pytest

from anml.data.data import Data, DataTypeError, EmptySpecsError
from anml.data.data_specs import DataSpecs
from anml.parameter.parameter import Intercept, Parameter, ParameterSet


@pytest.fixture
def df():
    return pd.DataFrame({
        'observations': np.random.randn(5),
        'obs_std_err': np.random.randn(5),
        'group': np.arange(5),
        'obs_se': np.random.randn(5),
        'cov1': np.random.randn(5),
        'cov2': np.random.randn(5)
    })


@pytest.fixture(scope='module')
def SimpleParam():
    return ParameterSet(
        parameters=[
            Parameter(
                variables=[Intercept()],
                link_fun=lambda x: x,
                param_name='foo'
            )
        ]
    )


def test_attach_detach_specs():

    d = Data()
    assert isinstance(d._data_specs, list)
    assert len(d._data_specs) == 0

    specs = DataSpecs(
        col_obs='observations', col_obs_se='obs_std_err', col_groups=['group']
    )

    d.set_data_specs(specs)
    assert specs == d._data_specs[0]

    d.detach_data_specs()
    assert len(d._data_specs) == 0


def test_attach_detach_params(SimpleParam):

    d = Data()
    assert isinstance(d._param_set, list)
    assert len(d._param_set) == 0

    d.set_param_set(SimpleParam)
    assert SimpleParam == d._param_set[0]

    d.detach_param_set()
    assert len(d._param_set) == 0


def test_process_data(df):

    d = Data()
    specs = DataSpecs(
        col_obs='observations', col_obs_se='obs_std_err', col_groups=['group']
    )
    d.set_data_specs(specs)
    d.process_data(df)
    np.testing.assert_array_equal(
        d.data['obs'], np.asarray(df['observations'])
    )
    np.testing.assert_array_equal(
        d.data['obs_se'], np.asarray(df['obs_std_err'])
    )
    np.testing.assert_array_equal(
        d.data['groups'], np.asarray(df[['group']])
    )


def test_process_data_multi_spec(df):
    d = Data()

    specs1 = DataSpecs(col_obs='observations', col_obs_se='obs_std_err', col_groups=['group'])
    specs2 = DataSpecs(col_obs='observations', col_obs_se='obs_se', col_groups=['group'])

    d.set_data_specs([specs1, specs2])
    d.process_data(df)
    np.testing.assert_array_equal(
        d.data['obs'][0], np.asarray(df['observations'])
    )
    np.testing.assert_array_equal(
        d.data['obs'][1], np.asarray(df['observations'])
    )
    np.testing.assert_array_equal(
        d.data['obs_se'][0], np.asarray(df['obs_std_err'])
    )
    np.testing.assert_array_equal(
        d.data['obs_se'][1], np.asarray(df['obs_se'])
    )
    np.testing.assert_array_equal(
        d.data['groups'][0], np.asarray(df[['group']])
    )
    np.testing.assert_array_equal(
        d.data['groups'][0], np.asarray(df[['group']])
    )


def test_process_data_empty(df):
    d = Data()
    with pytest.raises(EmptySpecsError):
        d.process_data(df)


def test_data_type():

    with pytest.raises(DataTypeError):
        d = Data()
        d.process_data(np.arange(10))


def test_col_to_attribute():
    assert Data._col_to_attribute('col_obs') == 'obs'


def test_process_params(df, SimpleParam):

    d = Data()
    d.set_param_set(SimpleParam)
    d.process_params(df)

    np.testing.assert_array_equal(
        d.covariates[0]['foo'][0],
        np.ones(5).reshape((5, 1))
    )
    assert d._param_set[0].parameters[0].variables[0].covariate == 'intercept'
