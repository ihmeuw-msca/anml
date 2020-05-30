"""
===============
Data Management
===============

Data is managed and processed using :class:`~placeholder.data.data.Data`
with specifications provided through one or more
instances of :class:`~placeholder.data.data_specs.DataSpecs`.
"""

from typing import Union, List, Optional, Dict
import pandas as pd
import numpy as np

from placeholder.data.data_specs import DataSpecs, _check_compatible_specs
from placeholder.exceptions import PlaceholderError


class DataError(PlaceholderError):
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

    Attributes
    ----------
    data
        A dictionary of numpy ndarrays keyed by the column attribute in
        _data_specs, extracted from the data frame after doing self.process_data().
        If _data_specs has multiple elements, then the values will be a list
        of numpy ndarrays, in the order of _data_specs.
    """
    def __init__(self, data_specs: Optional[Union[DataSpecs, List[DataSpecs]]] = None):
        self._data_specs = []
        if data_specs is not None:
            self.set_data_specs(data_specs)

        self.data: Dict[str, Union[np.ndarray, List[np.ndarray]]] = {}

    def set_data_specs(self, data_specs):
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

    def detach_data_specs(self):
        """Remove existing data specs."""
        self._data_specs = list()

    def _validate_specs(self, df: pd.DataFrame):
        """Validate the existing data specifications and their compatibility with
        a data frame to be processed.

        Parameters
        ----------
        df
            A pandas.DataFrame to be validated with the existing specs.

        """
        for spec in self._data_specs:
            spec._validate_df(df=df)

    @staticmethod
    def _convert_to_numpy(df: pd.DataFrame, column: Union[str, List[str]]) -> np.ndarray:
        return np.asarray(df[column])

    def _process_data_with_spec(self, df: pd.DataFrame, spec: DataSpecs):
        """Processes a data frame according to this specification.
        Turns the pandas Series from the df into numpy arrays
        and stores them in self.data dictionary.

        Parameters
        ----------
        df
            pandas DataFrame that has columns to extract

        spec
            data specifications indicating which
            columns to extract and how to label them

        """
        cols = spec._col_attributes
        for col in cols:
            name = spec._col_to_name(col)
            if len(self._data_specs) == 1:
                self.data.update({
                    name: self._convert_to_numpy(df, getattr(spec, col))
                })
            elif len(self._data_specs) > 1:
                if name not in self.data:
                    current = list()
                else:
                    current = self.data[name]
                current.append(self._convert_to_numpy(df, getattr(spec, col)))
                self.data.update({
                    name: current
                })

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

        self._validate_specs(df=df)
        for spec in self._data_specs:
            self._process_data_with_spec(df=df, spec=spec)
