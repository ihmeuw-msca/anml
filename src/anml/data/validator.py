from inspect import signature
from typing import Callable

import numpy as np
from numpy import ndarray


def validate_validator(validator: Callable) -> None:
    if not callable(validator):
        raise TypeError("Validator must be callable.")
    if tuple(signature(validator).parameters.key()) != ("key", "value"):
        raise TypeError(f"Validator[{validator.__name__}] must have arguments "
                        "('key', 'value').")


def validate_no_nans(key: str, value: ndarray) -> None:
    if np.isnan(value).any():
        raise ValueError(f"{key} contains nans.")


def validate_positive(key: str, value: ndarray) -> None:
    if (value <= 0).any():
        raise ValueError(f"{key} contains nonpositive numbers.")


def validate_unique(key: str, value: ndarray) -> None:
    if len(np.unique(value, axis=0)) < value.shape[0]:
        raise ValueError(f"{key} contains duplicated value or rows.")
