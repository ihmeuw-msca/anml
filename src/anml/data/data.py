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

import numpy as np
import pandas as pd

from anml.data.data_specs import DataSpecs, _check_compatible_specs
from anml.exceptions import ANMLError
from anml.parameter.parameter import ParameterSet


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
        self._df = None

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
        self._df = df.copy()

        if len(self._data_specs) == 0:
            raise EmptySpecsError("Need to attach data specs before processing data.")

        for spec in self._data_specs:
            spec._validate_df(df=self._df)

        for attribute in self.data_spec_col_attributes:
            name = self._col_to_attribute(attribute)
            self.data[name] = list()
            for spec in self._data_specs:
                self.data[name].append(
                    self._df[getattr(spec, attribute)].to_numpy()
                )
            if not self.multi_spec:
                self.data[name] = self.data[name][0]
