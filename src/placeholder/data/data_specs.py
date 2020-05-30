"""
===================
Data Specifications
===================

Gives data specifications that are used in
:class:`~placeholder.data.data.Data`.
"""

from dataclasses import dataclass
from typing import List

from placeholder.exceptions import PlaceholderError


class DataSpecCompatibilityError(PlaceholderError):
    """Error raised when the data specs are not compatible with the data frame to be used."""
    pass


@dataclass
class DataSpecs:

    col_obs: str
    col_groups: List[str]
    col_obs_se: str

    def __post_init__(self):
        pass

    def _validate_data(self, df):
        """Validates the existing

        Parameters
        ----------
        df
            A pandas.DataFrame to be validated with these specifications.

        """
        for column in [self.col_obs, self.col_obs_se] + self.col_groups:
            if column not in df.columns:
                raise DataSpecCompatibilityError(f"{column} is not in data columns: {df.columns}")

