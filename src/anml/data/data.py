"""
===============
Data Management
===============

Data is managed and processed using :class:`~anml.data.data.Data`
with specifications provided through one or more
instances of :class:`~anml.data.data_specs.DataSpecs`.
"""

from collections import defaultdict
from typing import Union, List, Optional, Dict, Any
import pandas as pd
import numpy as np
from scipy.linalg import block_diag

from anml.parameter.parameter import ParameterSet
from anml.data.data_specs import DataSpecs, _check_compatible_specs
from anml.exceptions import ANMLError


class DataError(ANMLError):
    """Base error for the data module."""
    pass


class DataTypeError(DataError):
    """Error raised when the data type is not understood."""
    pass


class EmptySpecsError(DataError):
    """Error raise when an operation can't be performed
    because there are no specifications associated with the Data instance."""
    pass


class Data:
    """A data manager that takes data as inputs along with data specs and
    transforms into primitive types for use in the optimization.

    Parameters
    ----------
    data_specs
        A data specification object, or list of data specification objects
        that indicate what the columns of a data frame represent.
    param_set
        A parameter set that has covariate specifications, or list of these sets.

    Attributes
    ----------
    data
        A dictionary of numpy ndarrays keyed by the column attribute in
        _data_specs, extracted from the data frame after doing self.process_data().
        If _data_specs has multiple elements, then the values will be a list
        of numpy ndarrays, in the order of _data_specs.
    covariates
    """
    def __init__(self,
                 data_specs: Optional[Union[DataSpecs, List[DataSpecs]]] = None,
                 param_set: Optional[Union[ParameterSet, List[ParameterSet]]] = None):

        self._data_specs = []
        self._param_set = []

        if data_specs is not None:
            self.set_data_specs(data_specs)
        if param_set is not None:
            self.set_param_set(param_set)

        self.data: Dict[str, Union[np.ndarray, List[np.ndarray]]] = dict()
        self.covariates: List[Dict[str, Any]] = list()
        self.groups_info = defaultdict(dict)

    @property
    def data_spec_col_attributes(self):
        return self._data_specs[0]._col_attributes

    @property
    def _unique_covariates(self):
        covariates = [p_set._flat_covariates for p_set in self._param_set]
        return set([item for sublist in covariates for item in sublist])

    @property
    def multi_spec(self):
        return len(self._data_specs) > 1

    @property
    def multi_param_set(self):
        return len(self._param_set) > 1

    @staticmethod
    def _col_to_attribute(x: str) -> str:
        return ''.join(x.split('col_')[1:])

    def set_data_specs(self, data_specs: Union[DataSpecs, List[DataSpecs]]):
        """Updates the data specifications, or sets them if they are empty.

        Parameters
        ----------
        data_specs
            A data specification object, or list of data specification objects
            that indicate what the columns of a data frame represent.

        """
        if isinstance(data_specs, list):
            _check_compatible_specs(data_specs)
            self._data_specs = data_specs
        else:
            self._data_specs = [data_specs]

    def encode_groups(self, col_group, df: pd.DataFrame):
        group_assign = df[col_group].to_numpy()
        groups = np.unique(group_assign)
        group_id_dict = {grp: i for i, grp in enumerate(groups)}
        self.groups_info[col_group] = group_id_dict

    def set_param_set(self, param_set: Union[ParameterSet, List[ParameterSet]]):
        if isinstance(param_set, list):
            self._param_set = param_set
        else:
            self._param_set = [param_set]

    def detach_data_specs(self):
        """Remove existing data specs."""
        self._data_specs = list()

    def detach_param_set(self):
        """Remove existing parameter set."""
        self._param_set = list()

    def process_data(self, df: pd.DataFrame):
        """Process a data frame and attach to this instance with existing data specs.

        Parameters
        ----------
        df
            A pandas.DataFrame with all of the information that the existing data specifications
            needs.

        """
        if not isinstance(df, pd.DataFrame):
            raise DataTypeError("Data to attach must be in the form of a pandas.DataFrame.")

        if len(self._data_specs) == 0:
            raise EmptySpecsError("Need to attach data specs before processing data.")

        for spec in self._data_specs:
            spec._validate_df(df=df)

        for attribute in self.data_spec_col_attributes:
            name = self._col_to_attribute(attribute)
            self.data[name] = list()
            for spec in self._data_specs:
                self.data[name].append(
                    df[getattr(spec, attribute)].to_numpy()
                )
            if not self.multi_spec:
                self.data[name] = self.data[name][0]

    def process_params(self, df: pd.DataFrame):
        """Process a data frame's covariates and attach to this instance with existing parameter sets.

        Parameters
        ----------
        df
            A pandas.DataFrame with all of the information that the existing parameter sets encode
            for the covariates.

        """
        if len(self._param_set) == 0:
            raise EmptySpecsError("Need to attach parameter sets before processing data.")

        for param_set in self._param_set:
            param_set._validate_df(df=df)
        
        design_mat_blocks = []
        constr_mat_blocks = []
        re_mat_groups = defaultdict(dict)
        lbs = []
        ubs = []
        self.fe_variables_names = []
        for param_set in self._param_set:
            for parameter in param_set.parameters:
                for variable in parameter.variables:
                    design_mat = variable.design_mat(df=df)
                    design_mat_blocks.append(design_mat)
                    var_name = parameter.param_name + '_' + variable.covariate
                    self.fe_variables_names.append(var_name)
                    if variable.add_re:
                        re_mat_groups[variable.col_group][var_name] = design_mat
                    mat, lb, ub = variable.get_constraint_matrix()
                    constr_mat_blocks.append(mat)
                    lbs.append(lb)
                    ubs.append(ub)
        self.design_matrix = np.hstack(design_mat_blocks)
        self.constr_matrix = block_diag(*constr_mat_blocks)
        self.constr_lower_bounds = np.hstack(lbs)
        self.constr_upper_bounds = np.hstack(ubs)
        assert self.design_matrix.shape[1] == self.constr_matrix.shape[1]
        assert len(self.constr_lower_bounds) == len(self.constr_upper_bounds) == self.constr_matrix.shape[0]

        if len(re_mat_groups) == 0:
            self.re_matrix = None
        else:
            self.groups_info = defaultdict(dict)
            re_mat_blocks = []
            self.re_variables_names = []
            for col_group, dct in re_mat_groups.items():
                self.encode_groups(col_group, df)
                grp_assign = [self.groups_info[col_group][g] for g in df[col_group]]
                n_group = len(self.groups_info[col_group])

                self.re_variables_names.append(list(dct.keys()))
                mat = np.hstack(dct.values())
                n_coefs = mat.shape[1]
                re_mat = np.zeros((mat.shape[0], n_coefs * n_group))
                for i, row in enumerate(mat):
                    grp = grp_assign[i]
                    re_mat[i, grp * n_coefs: (grp + 1) * n_coefs] = row 
                re_mat_blocks.append(re_mat)
            
            self.re_matrix = np.hstack(re_mat_blocks)

    def collect_priors(self):
        s = 0
        def prior_fun(x):
            assert len(x) == self._param_set.num_fe
            s = 0
            val = 0.0
            for param in self._param_set.parameters:
                for variable in param.variables:
                    x_dim = variable.fe_prior.x_dim
                    val += variable.fe_prior.error_value(x[s: s + x_dim])
                    s += x_dim
            return val 
        self.priors_fun = lambda x: prior_fun(x)
