from __future__ import annotations

from operator import attrgetter
from typing import List, Optional, Union

import numpy as np
from anml.data.component import Component
from anml.data.validator import NoNans
from anml.parameter.smoothmapping import Identity, SmoothMapping
from anml.prior.main import Prior
from anml.prior.utils import filter_priors, get_prior_type
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

    def __init__(
        self,
        variables: List[Variable],
        transform: Optional[SmoothMapping] = None,
        offset: Optional[Union[str, Component]] = None,
        priors: Optional[List[Prior]] = None,
    ):
        self.variables = variables
        self.transform = transform
        self.offset = offset
        self.priors = priors

        self.design_mat = None
        self.prior_dict = {"direct": {}, "linear": {}}

    @variables.setter
    def variables(self, variables: List[Variable]):
        variables = list(variables)
        if not all(isinstance(variable, Variable) for variable in variables):
            raise TypeError(
                "Parameter input variables must be a list of " "instances of Variable."
            )
        self._variables = variables

    @transform.setter
    def transform(self, transform: Optional[SmoothMapping]):
        if transform is not None and not isinstance(transform, SmoothMapping):
            raise TypeError(
                "Parameter input transform must be an instance "
                "of SmoothMapping or None."
            )
        if transform is None:
            transform = Identity()
        self._transform = transform

    @offset.setter
    def offset(self, offset: Optional[Union[str, Component]]):
        if offset is not None:
            if not isinstance(offset, (str, Component)):
                raise TypeError(
                    "Parameter input offset has to be a string or "
                    "an instance of Component."
                )
            if isinstance(offset, str):
                offset = Component(offset, validators=[NoNans()])
        self._offset = offset

    @priors.setter
    def priors(self, priors: Optional[List[Prior]]):
        priors = list(priors) if priors is not None else []
        if not all(isinstance(prior, Prior) for prior in priors):
            raise TypeError(
                "Parameter input priors must be a list of " "instances of Prior."
            )
        self._priors = priors

    @property
    def size(self) -> int:
        """Size of the parameter coefficients. It is the sum of all sizes
        for variables.

        """
        return sum([variable.size for variable in self.variables])

    def attach(self, df: DataFrame):
        """Attach data frame to offset and cache the design matrix and gather
        the prior information.

        Parameters
        ----------
        df
            Given data frame.

        """
        if self.offset is not None:
            self.offset.attach(df)
        self.design_mat = np.hstack(
            [variable.get_design_mat(df) for variable in self.variables]
        )
        for prior_category in ["direct", "linear"]:
            for prior_type in ["UniformPrior", "GaussianPrior"]:
                getattr(self, f"_get_{prior_category}_prior")(prior_type)

    def _get_direct_prior(self, prior_type: str):
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
        params = np.hstack(
            [
                variable.get_direct_prior_params(prior_type)
                for variable in self.variables
            ]
        )
        self.prior_dict["direct"][prior_type] = get_prior_type(prior_type)(
            params[0], params[1]
        )

    def _get_linear_prior(self, prior_type: str):
        """Get the linear prior parameters. The linear prior refers to the
        priors that contain a linear map. This function will combine the linear
        priors from the list of variables and the ones in the additional priors
        provided by the parameter.

        Parameters
        ----------
        prior_type
            Given name of the prior type.

        """

        params, mat = tuple(
            zip(
                *[
                    variable.get_linear_prior_params(prior_type)
                    for variable in self.variables
                ]
            )
        )
        params = np.hstack(params)
        mat = block_diag(*mat)

        linear_priors = filter_priors(self.priors, prior_type, with_mat=True)
        if len(linear_priors) == 0:
            extra_params = np.empty((2, 0))
            extra_mat = np.empty((0, self.size))
        else:
            extra_params = np.hstack([prior.params for prior in linear_priors])
            extra_mat = np.vstack([prior.mat for prior in linear_priors])

        params = np.hstack([params, extra_params])
        mat = np.vstack([mat, extra_mat])

        self.prior_dict["linear"][prior_type] = get_prior_type(prior_type)(
            params[0], params[1], mat
        )

    def get_params(
        self, x: NDArray, df: Optional[DataFrame] = None, order: int = 0
    ) -> NDArray:
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

        Raises
        ------
        ValueError
            Raised when there is not cache of the design matrix and no data
            frame is provided.

        """
        if df is not None:
            self.attach(df)
        if self.design_mat is None:
            raise ValueError("Must provide a data frame to attach data.")
        y = self.design_mat.dot(x)
        if self.offset is not None:
            y += self.offset.value
        z = self.transform(y, order=order)

        if order == 0:
            return z
        if order == 1:
            return z[:, np.newaxis] * self.design_mat
        return z[:, np.newaxis, np.newaxis] * (
            self.design_mat[..., np.newaxis] * self.design_mat[:, np.newaxis, :]
        )

    def prior_objective(self, x: NDArray) -> float:
        """Objective function from the prior.

        Parameters
        ----------
        x
            Coefficients for the design matrix.

        Returns
        -------
        float
            Objective value from the prior.

        """
        value = 0.0
        for prior_category in ["direct", "linear"]:
            prior = self.prior_dict[prior_category]["GaussianPrior"]
            value += prior.objective(x)
        return value

    def prior_gradient(self, x: NDArray) -> NDArray:
        """Gradient function from the prior.

        Parameters
        ----------
        x
            Coefficients for the design matrix.

        Returns
        -------
        NDArray
            Gradient value from the prior.

        """
        value = np.zeros(x.size, dtype=x.dtype)
        for prior_category in ["direct", "linear"]:
            prior = self.prior_dict[prior_category]["GaussianPrior"]
            value += prior.gradient(x)
        return value

    def prior_hessian(self, x: NDArray) -> NDArray:
        """Hessian function from the prior.

        Parameters
        ----------
        x
            Coefficients for the design matrix.

        Returns
        -------
        NDArray
            Hessian value from the prior.

        """
        value = np.zeros((x.size, x.size), dtype=x.dtype)
        for prior_category in ["direct", "linear"]:
            prior = self.prior_dict[prior_category]["GaussianPrior"]
            value += prior.hessian(x)
        return value

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(variables={self.variables}, "
            f"transform={self.transform}, offset={self.offset}, "
            f"priors={self.priors})"
        )
