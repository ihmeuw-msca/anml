"""
===================
Data Specifications
===================

Gives data specifications that are used in
:class:`~anml.data.data.Data`.

A :class:`~anml.data.data.Data` class can be subclassed
for use in applications that have other standard columns outside
of the three default
"""

from dataclasses import dataclass
from typing import List
import pandas as pd

from anml.exceptions import ANMLError


class DataSpecCompatibilityError(ANMLError):
    """Error raised when the data specs are not compatible with the data frame to be used."""
    pass


@dataclass
class DataSpecs:

    col_obs: str
    col_obs_se: str = None
    col_groups: List[str] = None

    def __post_init__(self):
        pass

    @property
    def _attrs(self):
        return vars(self)

    @property
    def _col_attributes(self):
        return list(k for k in self._attrs if self._attrs[k] is not None)

    @property
    def _data_attributes(self):
        return list(k for k in self._attrs.values() if k is not None)

    def _validate_df(self, df: pd.DataFrame):
        """Validates the existing

        Parameters
        ----------
        df
            A pandas.DataFrame to be validated with these specifications.

        """
        for column in self._data_attributes:
            if column is None:
                continue
            if isinstance(column, str):
                column = [column]
            for col in column:
                if col not in df.columns:
                    raise DataSpecCompatibilityError(f"{col} is not in data columns: {df.columns}")


def _check_compatible_specs(specs: List[DataSpecs]):
    first_spec = specs[0]
    for i, spec in enumerate(specs[1:]):
        if sorted(spec._col_attributes) != sorted(first_spec._col_attributes):
            raise DataSpecCompatibilityError(
                "At least one data spec is different."
                f"Columns in spec 1 are {spec._col_attributes}."
                f"Columns in spec {i+2} are {spec._col_attributes}."
            )
    for attribute in first_spec._col_attributes:
        attr_type = type(getattr(first_spec, attribute))
        for spec in specs[1:]:
            if not isinstance(getattr(spec, attribute), attr_type):
                raise DataSpecCompatibilityError(
                    "At least one data spec type is different."
                    f"The attribute {attribute} should be of type {attr_type}."
                )
