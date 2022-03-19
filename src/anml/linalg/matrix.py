"""
Matrix
======

Matrix classes for consistent interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Tuple, Union

import numpy
from numpy.typing import NDArray
from scipy.sparse import csc_matrix, csr_matrix, spdiags
from scipy.sparse.linalg import spsolve


class Matrix(ABC):

    @abstractproperty
    def T(self) -> Matrix:
        pass

    @abstractproperty
    def shape(self) -> Tuple[int, int]:
        pass

    @abstractproperty
    def size(self) -> int:
        pass

    @abstractmethod
    def dot(self, x: Union[NDArray, Matrix]) -> Union[NDArray, Matrix]:
        pass

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

    def __repr__(self) -> str:
        return f"{type(self).__name__}(shape={self.shape})"


class SparseMatrix(Matrix):

    @abstractproperty
    def nnz(self) -> int:
        pass

    def __repr__(self) -> str:
        return f"{type(self).__name__}(shape={self.shape}, nnz={self.nnz})"


def asmatrix(data: Any) -> Matrix:
    if isinstance(data, numpy.ndarray):
        return NumpyMatrix(data)
    if isinstance(data, csr_matrix):
        return CSRMatrix(data)
    if isinstance(data, csc_matrix):
        return CSCMatrix(data)
    raise TypeError("Unknown data type, cannot convert to a matrix.")


class NumpyMatrix(Matrix):

    def __init__(self, data: NDArray):
        data = numpy.asarray(data)
        if data.ndim != 2:
            raise ValueError("Matrix must have two dimensions")
        self.data = data

    @property
    def T(self) -> NumpyMatrix:
        return NumpyMatrix(self.data.T)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape

    @property
    def size(self) -> int:
        return self.data.size

    def dot(self, x: Union[NDArray, Matrix]) -> Union[NDArray, Matrix]:
        if isinstance(x, numpy.ndarray):
            return self.data.dot(x)
        return asmatrix(self.data.dot(x.data))

    def scale_rows(self, x: NDArray) -> NumpyMatrix:
        return NumpyMatrix(x[:, numpy.newaxis] * self.data)

    def scale_cols(self, x: NDArray) -> NumpyMatrix:
        return NumpyMatrix(self.data * x)

    def solve(self, x: NDArray) -> NDArray:
        return numpy.linalg.solve(self.data, x)

    def to_numpy(self) -> NDArray:
        return self.data.copy()


class CSRMatrix(SparseMatrix):

    def __init__(self, data: csr_matrix):
        if not isinstance(data, csr_matrix):
            raise TypeError("Data of CSRMatrix must be an instance of "
                            "csr_matrix.")
        self.data = data

    @property
    def T(self) -> CSCMatrix:
        return CSCMatrix(self.data.T)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def nnz(self) -> int:
        return self.data.nnz

    def dot(self, x: Union[NDArray, Matrix]) -> Union[NDArray, Matrix]:
        if isinstance(x, numpy.ndarray):
            return self.data.dot(x)
        return asmatrix(self.data.dot(x.data))

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


class CSCMatrix(SparseMatrix):

    def __init__(self, data: csc_matrix):
        if not isinstance(data, csc_matrix):
            raise TypeError("Data of CSCMatrix must be an instance of "
                            "csc_matrix.")
        self.data = data

    @property
    def T(self) -> CSRMatrix:
        return CSRMatrix(self.data.T)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def nnz(self) -> int:
        return self.data.nnz

    def dot(self, x: Union[NDArray, Matrix]) -> Union[NDArray, Matrix]:
        if isinstance(x, numpy.ndarray):
            return self.data.dot(x)
        return asmatrix(self.data.dot(x.data))

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
