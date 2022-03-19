"""
Matrix
======

Matrix classes for consistent interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, Tuple, Union

import numpy
from numpy.typing import NDArray
from scipy.sparse import csc_matrix, csr_matrix, spdiags
from scipy.sparse.linalg import spsolve


class Array(Protocol):

    @property
    def T(self) -> Array:
        """Transpose of the array."""

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the array."""

    @property
    def size(self) -> int:
        """Size of the array, total number of elements."""

    @property
    def ndim(self) -> int:
        """Number of dimensions of the array."""

    def dot(self, x: Array) -> Array:
        """Dot product with another array."""


def asmatrix(data: Array) -> Matrix:
    if isinstance(data, Matrix):
        return data
    if isinstance(data, numpy.ndarray):
        return NumpyMatrix(data)
    if isinstance(data, csr_matrix):
        return CSRMatrix(data)
    if isinstance(data, csc_matrix):
        return CSCMatrix(data)
    raise TypeError(f"Cannot convert {type(data)} to a matrix.")


class Matrix(ABC):

    def __init__(self, data: Array):
        if data.ndim != 2:
            raise ValueError("Matrix must have two dimensions")
        self.data = data

    @property
    def T(self) -> Matrix:
        return asmatrix(self.data.T)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def dot(self, x: Union[NDArray, Matrix]) -> Union[NDArray, Matrix]:
        if isinstance(x, numpy.ndarray) and x.ndim == 1:
            return self.data.dot(x)
        x = asmatrix(x)
        return asmatrix(self.data.dot(x.data))

    @abstractmethod
    def scale_rows(self, x: NDArray) -> Matrix:
        pass

    @abstractmethod
    def scale_cols(self, x: NDArray) -> Matrix:
        pass

    @abstractmethod
    def solve(self, x: NDArray) -> NDArray:
        pass

    @abstractmethod
    def to_numpy(self) -> NDArray:
        pass

    def __add__(self, x: Matrix) -> Matrix:
        return asmatrix(self.data + x.data)

    def __sub__(self, x: Matrix) -> Matrix:
        return asmatrix(self.data - x.data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(shape={self.shape})"


class NumpyMatrix(Matrix):

    def __init__(self, data: NDArray):
        super().__init__(numpy.asarray(data))

    def scale_rows(self, x: NDArray) -> NumpyMatrix:
        return NumpyMatrix(x[:, numpy.newaxis] * self.data)

    def scale_cols(self, x: NDArray) -> NumpyMatrix:
        return NumpyMatrix(self.data * x)

    def solve(self, x: NDArray) -> NDArray:
        return numpy.linalg.solve(self.data, x)

    def to_numpy(self) -> NDArray:
        return self.data.copy()


class CSRMatrix(Matrix):

    def __init__(self, data: csr_matrix):
        super().__init__(csr_matrix(data))

    def scale_rows(self, x: NDArray) -> CSRMatrix:
        return CSRMatrix(spdiags(x, 0, len(x), len(x)) * self.data)

    def scale_cols(self, x: NDArray) -> CSRMatrix:
        result = self.data.copy()
        result.data *= x[result.indices]
        return CSRMatrix(result)

    def solve(self, x: NDArray) -> NDArray:
        return spsolve(self.data, x)

    def to_numpy(self) -> NDArray:
        return self.data.toarray()


class CSCMatrix(Matrix):

    def __init__(self, data: csc_matrix):
        super().__init__(csc_matrix(data))

    def scale_rows(self, x: NDArray) -> CSCMatrix:
        result = self.data.copy()
        result.data *= x[result.indices]
        return CSCMatrix(result)

    def scale_cols(self, x: NDArray) -> CSRMatrix:
        return CSCMatrix(self.data * spdiags(x, 0, len(x), len(x)))

    def solve(self, x: NDArray) -> NDArray:
        return spsolve(self.data, x)

    def to_numpy(self) -> NDArray:
        return self.data.toarray()
