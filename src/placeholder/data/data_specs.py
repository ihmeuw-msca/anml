"""
===================
Data Specifications
===================

Gives data specifications that are used in
:class:`~placeholder.data.data.Data`.
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

