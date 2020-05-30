"""
===============
Data Management
===============

Data is managed and processed using :class:`~placeholder.data.data.Data`
with specifications provided through one or more
instances of :class:`~placeholder.data.data_specs.DataSpecs`.
"""

from typing import Union, List, Optional
import pandas as pd

from placeholder.data.data_specs import DataSpecs
from placeholder.exceptions import PlaceholderError


class DataError(PlaceholderError):
    """Base error for the data module."""
    pass


class DataTypeError(DataError):
    """Error raised when the data type is not understood."""
    pass


class Data:
    """A data manager that takes data as inputs along with data specs and
    transforms into primitive types for use in the optimization.

    Parameters
    ----------
    data_specs
        A data specification object, or list of data specification objects
        that indicate what the columns of a data frame represent.
    """
    def __init__(self, data_specs: Optional[Union[DataSpecs, List[DataSpecs]]] = None):
        self._data_specs = []
        if data_specs is not None:
            self.set_data_specs(data_specs)

    def set_data_specs(self, data_specs):
        """Updates the data specifications, or sets them if they are empty.

        Parameters
        ----------
        data_specs
            A data specification object, or list of data specification objects
            that indicate what the columns of a data frame represent.

        """
        if isinstance(data_specs, list):
            self._data_specs = data_specs
        else:
            self._data_specs = [data_specs]

    def detach_data_specs(self):
        """Remove existing data specs."""
        self._data_specs = list()

    def _validate_specs(self, df):
        """Validate the existing data specifications and their compatibility with
        a data frame to be processed.

        Parameters
        ----------
        df
            A pandas.DataFrame to be validated with the existing specs.

        """
        for spec in self._data_specs:
            spec._validate_df(df=df)

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

        self._validate_specs(df=df)

