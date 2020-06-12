import pandas as pd
import numpy as np
import pytest

from anml.data.data import Data, DataTypeError, EmptySpecsError
from anml.data.data_specs import DataSpecs
from anml.parameter.parameter import Intercept, Parameter, ParameterSet, Spline, SplineLinearConstr


@pytest.fixture
def df():
    return pd.DataFrame({
        'observations': np.random.randn(5),
        'obs_std_err': np.random.randn(5),
        'group': ['1', '2', '3', '3', '1'],
        'obs_se': np.random.randn(5),
        'cov1': np.random.randn(5),
        'cov2': np.random.randn(5)
    })


@pytest.fixture(scope='module')
def SimpleParam():
    return ParameterSet(
        parameters=[
            Parameter(
                variables=[Intercept(add_re=True, col_group='group')],
                link_fun=lambda x: x,
                param_name='foo'
            )
        ]
    )


@pytest.fixture(scope='module')
def SplineParam():
    constr_cvx = SplineLinearConstr(order=2, y_bounds=[0.0, np.inf], grid_size=5)
    spline_var = Spline(
        covariate='cov1',
        derivative_constr=[constr_cvx],
        degree=2,
    )
    parameter = Parameter(variables=[spline_var, Intercept(add_re=True, col_group='group')], param_name='foo')
    return ParameterSet([parameter])


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
    design_matrix, re_matrix, constr_matrix, _, _ = d.process_params(df)
    assert design_matrix.shape == (5, 1)
    
    assert re_matrix.shape == (5, 3)
    assert re_matrix[0, 0] == 1
    assert re_matrix[1, 1] == 1
    assert re_matrix[2, 2] == 1
    assert re_matrix[3, 2] == 1
    assert re_matrix[4, 0] == 1
    
    assert constr_matrix.shape == (1, 1)


def test_process_params_spline(df, SplineParam):
    d = Data()
    d.set_param_set(SplineParam)
    design_matrix, re_matrix, constr_matrix, lb, ub = d.process_params(df)
    assert design_matrix.shape == (5, 4)
    assert re_matrix.shape == (5, 3)
    assert constr_matrix.shape == (6, 4)
    assert len(lb) == len(ub) == 6
