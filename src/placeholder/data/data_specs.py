"""
===================
Data Specifications
===================

Data is managed and processed using :class:`~placeholder.data.data.Data`
with specifications provided through one or more
instances of :class:~placeholder.data.data_specs.DataSpecs`.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class DataSpecs:

    col_obs: str
    col_groups: List[str]
    col_obs_se: str

    def __post_init__(self):
        pass

