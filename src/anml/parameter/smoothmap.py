from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class SmoothMap(ABC):

    def _validate_order(self, order: int = 0):
        if order not in [0, 1, 2]:
            raise ValueError("Order must be 0, 1 or 2.")

    @property
    def inverse(self) -> SmoothMap:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, x: NDArray, order: int = 0) -> NDArray:
        pass

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Identity(SmoothMap):

    @property
    def inverse(self) -> SmoothMap:
        return Identity()

    def __call__(self, x: NDArray, order: int = 0) -> NDArray:
        self._validate_order(order)
        return x


class Exponential(SmoothMap):

    @property
    def inverse(self) -> SmoothMap:
        return Log()

    def __call__(self, x: NDArray, order: int = 0) -> NDArray:
        self._validate_order(order)
        return np.exp(x)


class Log(SmoothMap):

    @property
    def inverse(self) -> SmoothMap:
        return Exponential()

    def __call__(self, x: NDArray, order: int = 0) -> NDArray:
        self._validate_order(order)
        if not (x > 0).all():
            raise ValueError("All values for log function must be positive.")
        if order == 0:
            return np.log(x)
        elif order == 1:
            return 1 / x
        return -1 / x**2


class Expit(SmoothMap):

    @property
    def inverse(self) -> SmoothMap:
        return Logit()

    def __call__(self, x: NDArray, order: int = 0) -> NDArray:
        self._validate_order(order)
        z = np.exp(-x**2)
        if order == 0:
            return 1 / (1 + z)
        elif order == 1:
            return z / (1 + z)**2
        return z * (z - 1) / (z + 1)**3


class Logit(SmoothMap):

    @property
    def inverse(self) -> SmoothMap:
        return Expit()

    def __call__(self, x: NDArray, order: int = 0) -> NDArray:
        self._validate_order(order)
        if not ((x > 0).all() and (x < 1).all()):
            raise ValueError("All values for logit function must be between "
                             "0 and 1.")
        if order == 0:
            return np.log(x / (1 - x))
        elif order == 1:
            return 1 / (x * (1 - x))
        return (2 * x - 1) / (x * (1 - x))**2
