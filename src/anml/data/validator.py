from abc import ABC

import numpy as np
from numpy.typing import NDArray


class Validator(ABC):
    """Validator class validates the data satisfy the condition. The instance is
    callable. And if the condition is not met, the call will raise value error.
    """

    def __call__(self, key: str, value: NDArray):
        pass

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class NoNans(Validator):

    def __call__(self, key: str, value: NDArray):
        if np.isnan(value).any():
            raise ValueError(f"Column '{key}' contains nans.")


class Positive(Validator):

    def __call__(self, key: str, value: NDArray):
        if (value <= 0).any():
            raise ValueError(f"Column '{key}' contains nonpositive numbers.")


class Unique(Validator):

    def __call__(self, key: str, value: NDArray):
        if len(np.unique(value, axis=0)) < value.shape[0]:
            raise ValueError(f"Column '{key}' contains duplicated values.")
