from __future__ import annotations

from operator import attrgetter
from typing import List, Optional, Tuple, Union

import numpy as np
from anml.data.component import Component
from anml.data.validator import NoNans
from anml.parameter.smoothmapping import Identity, SmoothMapping
from anml.prior.main import Prior
from anml.prior.utils import filter_priors
from anml.variable.main import Variable
from numpy.typing import NDArray
from pandas import DataFrame
from scipy.linalg import block_diag


class Parameter:
    """Parameter class contains information from a list of variables that are
    used to parametrize the distribution parameter. Parameter class also include
    optional transformation function, offset and list of priors.

    Parameters
    ----------
    variables
        A list of variables that parametrized the linear predictor of the
        parameter.
    transform
        A :class:`SmoothMapping` instance that transforms the linear prediction
        from the variables to the parameter space. Default is `None`, which will
        be converted into identity mapping.
    offset
        A data component contains the offset information for the parameter.
        Default is `None` which indicates no offset. Offset will be applied
        onto the linear predictor.
    priors
        A list of additional priors that directly apply to the parameter linear
        predictor. Default is `None`, where no additional priors will be added.

    """

    variables = property(attrgetter("_variables"))
    """A list of variables that parametrized the linear predictor of the
    parameter.

    Raises
    ------
    TypeError
        Raised if the input variables are not all instances of
        :class:`Variable`.

    """
    transform = property(attrgetter("_transform"))
    """A function that transforms the linear prediction to the parameter space.

    Raises
    ------
    TypeError
        Raised when the input transform is not an instance of
        :class:`SmoothMapping`.

    """
    offset = property(attrgetter("_offset"))
    """Offset for the linear predictor.

    Raises
    ------
    TypeError
        Raised when the input offset is not `None`, or a string, or an instance
        of :class:`DataComponent`.

    """
    priors = property(attrgetter("_priors"))
    """A list of additional priors that apply to the linear predictor.

    Raises
    ------
    TypeError
        Raised when the input priors are not `None` or a list of instances of
        :class:`Prior`.

    """

    def __init__(self,
                 variables: List[Variable],
                 transform: Optional[SmoothMapping] = None,
                 offset: Optional[Union[str, Component]] = None,
                 priors: Optional[List[Prior]] = None):
        self.variables = variables
        self.transform = transform
        self.offset = offset
        self.priors = priors
        self._design_mat = None

    @variables.setter
    def variables(self, variables: List[Variable]):
        variables = list(variables)
        if not all(isinstance(variable, Variable) for variable in variables):
            raise TypeError("Parameter input variables must be a list of "
                            "instances of Variable.")
        self._variables = variables

    @transform.setter
    def transform(self, transform: Optional[SmoothMapping]):
        if transform is not None and not isinstance(transform, SmoothMapping):
            raise TypeError("Parameter input transform must be an instance "
                            "of SmoothMapping or None.")
        if transform is None:
            transform = Identity()
        self._transform = transform

    @offset.setter
    def offset(self, offset: Optional[Union[str, Component]]):
        if offset is not None:
            if not isinstance(offset, (str, Component)):
                raise TypeError("Parameter input offset has to be a string or "
                                "an instance of Component.")
            if isinstance(offset, str):
                offset = Component(offset, validators=[NoNans()])
        self._offset = offset

    @priors.setter
    def priors(self, priors: Optional[List[Prior]]):
        priors = list(priors) if priors is not None else []
        if not all(isinstance(prior, Prior) for prior in priors):
            raise TypeError("Parameter input priors must be a list of "
                            "instances of Prior.")
        self._priors = priors

    @property
    def size(self) -> int:
        """Size of the parameter coefficients. It is the sum of all sizes
        for variables.

        """
        return sum([variable.size for variable in self.variables])

    def attach(self, df: DataFrame):
        """Attach data frame to variables and offset.

        Parameters
        ----------
        df
            Given data frame.

        """
        for variable in self.variables:
            variable.attach(df)
        if self.offset is not None:
            self.offset.attach(df)

    def get_design_mat(self, df: Optional[DataFrame] = None) -> NDArray:
        """Get the design matrix for the linear predictor. If the data frame is
        provided it will compute the design matrix and cache it. If no data
        frame is provided, it will try to use the cached design matrix. If no
        no data frame is provided and the instance does not contain a cached
        design matrix, it will raise error.


        Parameters
        ----------
        df
            Given data frame. Default is `None`. 
        Returns
        -------
        NDArray
            The design matrix for the linear predictor.

        Raises
        ------
        ValueError
            Raised when `df=None` and there is no cached design matrix.

        """
        if df is None and self._design_mat is None:
            raise ValueError("Must provide a data frame, do not have cache for "
                             "the design matrix.")
        if df is None:
            return self._design_mat
        self._design_mat = np.hstack([variable.get_design_mat(df)
                                      for variable in self.variables])
        return self._design_mat

    def get_direct_prior_params(self, prior_type: str) -> NDArray:
        """Get the direct prior parameters. The direct prior refers to the
        priors that do not have a linear map and direct act on the variable.
        This function will ignore the direct priors provided by the additional
        priors in the parameter. Please add direct priors on :class:`Variable`
        instances.

        Parameters
        ----------
        prior_type
            Given name of the prior type.

        """
        return np.hstack([variable.get_direct_prior_params(prior_type)
                          for variable in self.variables])

    def get_linear_prior_params(self, prior_type: str) -> Tuple[NDArray, NDArray]:
        """Get the linear prior parameters. The linear prior refers to the
        priors that contain a linear map. This function will combine the linear
        priors from the list of variables and the ones in the additional priors
        provided by the parameter.

        Parameters
        ----------
        prior_type
            Given name of the prior type.

        """

        params, mat = tuple(zip(*[variable.get_linear_prior_params(prior_type)
                                  for variable in self.variables]))
        params = np.hstack(params)
        mat = block_diag(*mat)

        linear_priors = filter_priors(self.priors, prior_type, with_mat=True)
        if len(linear_priors) == 0:
            extra_params = np.empty((2, 0))
            extra_mat = np.empty((0, self.size))
        else:
            extra_params = np.hstack([prior.params for prior in linear_priors])
            extra_mat = np.vstack([prior.mat for prior in linear_priors])

        return np.hstack([params, extra_params]), np.vstack([mat, extra_mat])

    def get_params(self,
                   x: NDArray,
                   df: Optional[DataFrame] = None,
                   order: int = 0) -> NDArray:
        """Compute and return the parameter. Denote :math:`x` as the
        coefficients, :math:`A` as the design matrix, :math:`z` as the offset,
        :math:`f` as the transformation function, the parameter :math:`p` can
        be represented as

        .. math:: p = f(z + Ax)

        Here we call :math:`Ax` as the linear predictor.

        Parameters
        ----------
        x
            Coefficients for the design matrix.
        df
            Given data frame used for compute the design matrix. Default is
            `None`.
        order
            Order of the derivative. Default is 0.

        Returns
        -------
        NDArray
            When `order=0`, it will return the parameter value. When `order=1`,
            it will return the Jacobian matrix. And when `order=2`, it will
            return the second order Jacobian tensor.
        """
        design_mat = self.get_design_mat(df)
        y = design_mat.dot(x)
        if self.offset is not None:
            y += self.offset.value
        z = self.transform(y, order=order)

        if order == 0:
            return z
        if order == 1:
            return z[:, np.newaxis] * design_mat
        return z[:, np.newaxis, np.newaxis] * \
            (design_mat[..., np.newaxis] * design_mat[:, np.newaxis, :])

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(variables={self.variables}, "
                f"transform={self.transform}, offset={self.offset}, "
                f"priors={self.priors})")
