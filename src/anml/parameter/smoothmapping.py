from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class SmoothMapping(ABC):
    """Smooth mapping that contains function, first and second derivative
    information.

    """

    def _validate_order(self, order: int = 0):
        """Validate input order for the function call.

        Parameters
        ----------
        order
            Order of the derivative of the function, by default 0. When it is 0
            call will return function value, if it is 1, call will return the
            first order derivative, if it is 2, call will return the second
            order derviative.

        Raises
        ------
        ValueError
            Raised when order is not 0, 1 or 2.

        """
        if order not in [0, 1, 2]:
            raise ValueError("Order must be 0, 1 or 2.")

    @property
    def inverse(self) -> SmoothMapping:
        """Inverse function of the current instance. The inverse function is
        also an instance of :class:`SmoothMapping`.

        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, x: NDArray, order: int = 0) -> NDArray:
        pass

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Identity(SmoothMapping):
    """Identity smooth mapping."""

    @property
    def inverse(self) -> SmoothMapping:
        """Inverse of :class:`Identity` is :class:`Identity`."""
        return Identity()

    def __call__(self, x: NDArray, order: int = 0) -> NDArray:
        self._validate_order(order)
        if order == 0:
            return x
        if order == 1:
            return np.ones(x.shape)
        return np.zeros(x.shape)


class Exp(SmoothMapping):
    """Exponential smooth mapping."""

    @property
    def inverse(self) -> SmoothMapping:
        """Inverse of :class:`Exp` is :class:`Log`."""
        return Log()

    def __call__(self, x: NDArray, order: int = 0) -> NDArray:
        self._validate_order(order)
        return np.exp(x)


class Log(SmoothMapping):
    """Logarithm smooth mapping.

    Raises
    ------
    ValueError
        Raised when the argument is not all positive.

    """

    @property
    def inverse(self) -> SmoothMapping:
        """Inverse of :class:`Log` is :class:`Exp`."""
        return Exp()

    def __call__(self, x: NDArray, order: int = 0) -> NDArray:
        self._validate_order(order)
        if not (x > 0).all():
            raise ValueError("All values for log function must be positive.")
        if order == 0:
            return np.log(x)
        elif order == 1:
            return 1 / x
        return -1 / x**2


class Expit(SmoothMapping):
    """Expit smooth mapping.

    .. math:: \\mathrm{expit}(x) = \\frac{1}{1 + \\exp(-x)}

    """

    @property
    def inverse(self) -> SmoothMapping:
        """Inverse of :class:`Expit` is :class:`Logit`."""
        return Logit()

    def __call__(self, x: NDArray, order: int = 0) -> NDArray:
        self._validate_order(order)
        z = np.exp(-x)
        if order == 0:
            return 1 / (1 + z)
        elif order == 1:
            return z / (1 + z) ** 2
        return z * (z - 1) / (z + 1) ** 3


class Logit(SmoothMapping):
    """Logit smooth mapping.

    .. math:: \\mathrm{logit}(x) = \\log\\left(\\frac{x}{1 - x}\\right)

    Raises
    ------
    ValueError
        Raised when the argument is not all strictly between 0 and 1.

    """

    @property
    def inverse(self) -> SmoothMapping:
        """Inverse of :class:`Logit` is :class:`Expit`."""
        return Expit()

    def __call__(self, x: NDArray, order: int = 0) -> NDArray:
        self._validate_order(order)
        if not ((x > 0).all() and (x < 1).all()):
            raise ValueError(
                "All values for logit function must be strictly " "between 0 and 1."
            )
        if order == 0:
            return np.log(x / (1 - x))
        elif order == 1:
            return 1 / (x * (1 - x))
        return (2 * x - 1) / (x * (1 - x)) ** 2
