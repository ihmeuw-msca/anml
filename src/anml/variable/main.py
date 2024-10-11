from operator import attrgetter
from typing import List, Optional, Tuple, Type, Union

import numpy as np
from anml.data.component import Component
from anml.data.validator import NoNans
from anml.prior.main import Prior
from anml.prior.utils import filter_priors, get_prior_type
from numpy.typing import NDArray
from pandas import DataFrame


class Variable:
    """Variable class that contains information of variable, including name and
    priors. It provides functions of create design matrix and gather prior
    information for likelihood building.

    Parameters
    ----------
    component
        You can pass in the name of the variable corresponding to the column
        name in the data frame. It will be automatically converted into an
        instance of :class:`Component` with :class:`NoNans` as the validator.
        Alternatively, you can also pass in an instance of :class:`Component`,
        with your own set of validators.
    priors
        A list of priors corresponding to the variable.

    """

    component = property(attrgetter("_component"))
    """Data compoent for the variable.

    Raises
    ------
    TypeError
        Raised when the input component are not a string nor an instance of
        :class:`Component`.

    """
    priors = property(attrgetter("_priors"))
    """A list of priors corresponding to the variable.

    Raises
    ------
    TypeError
        Raised when the input priors are not a list of given prior types.
        Different classes have different legal prior types that is stored in a
        protected class variable `_prior_types`.

    """
    _prior_types: Tuple[Type, ...] = (Prior,)
    """A Tuple of all allowed prior types.

    """

    def __init__(
        self, component: Union[str, Component], priors: Optional[List[Prior]] = None
    ):
        self.component = component
        self.priors = priors

    @component.setter
    def component(self, component: Union[str, Component]):
        if not isinstance(component, (str, Component)):
            raise TypeError(
                "Variable input component has to be a string or "
                "an instance of Component."
            )
        if isinstance(component, str):
            component = Component(component, validators=[NoNans()])
        self._component = component

    @priors.setter
    def priors(self, priors: Optional[List[Prior]]):
        priors = list(priors) if priors is not None else []
        if not all(isinstance(prior, self._prior_types) for prior in priors):
            raise TypeError(
                "Variable input priors must be a list of " "instances of Prior."
            )
        self._priors = priors

    @property
    def size(self) -> Optional[int]:
        """Size of the variable."""
        return 1

    def attach(self, df: DataFrame):
        """Attach the data to variable. It will attach data to the component.

        Parameters
        ----------
        df
            The data frame contains the corresponding data column.

        """
        self.component.attach(df)

    def get_design_mat(self, df: DataFrame) -> NDArray:
        """Get design matrix.

        Parameters
        ----------
        df
            The data frame contains the corresponding data column.

        Returns
        -------
        NDArray
            The design matrix as a numpy array.

        """
        self.attach(df)
        return self.component.value[:, np.newaxis]

    def get_direct_prior_params(self, prior_type: str) -> NDArray:
        """Get the direct prior parameters. The direct prior refers to the
        priors that do not have a linear map and direct act on the variable.
        We require that one variable can only have one direct prior for a given
        prior type. If there is no direct prior in the prior list, we will use
        the default prior parameters of the given prior type.

        Parameters
        ----------
        prior_type
            Given name of the prior type.

        Returns
        -------
        NDArray
            The prior parameters as an array.

        Raises
        ------
        ValueError
            Raised when have more than one direct priors.
        ValueError
            Raised when the size of the prior parameters doesn't match with the
            size of the variable.

        """
        direct_priors = filter_priors(self.priors, prior_type, with_mat=False)
        prior_type = get_prior_type(prior_type)
        if len(direct_priors) == 0:
            return np.repeat(prior_type.default_params, self.size, axis=1)
        if len(direct_priors) >= 2:
            raise ValueError("Variable can only have one direct prior.")
        prior = direct_priors[0]
        if prior.shape[1] != self.size:
            raise ValueError("Variable and prior size don't match.")
        return prior.params

    def get_linear_prior_params(self, prior_type: str) -> Tuple[NDArray, NDArray]:
        """Get the linear prior parameters. The linear prior refers to the
        priors that contain a linear map. If there is no linear prior in the
        prior list, we will return empty arrays that match the size of the
        variable.

        Parameters
        ----------
        prior_type
            Given name of the prior type.

        Returns
        -------
        Tuple[NDArray, NDArray]
            The prior parameters as an array. The first one is the distribution
            parameters and the second one is the linear map.

        Raises
        ------
        ValueError
            Raised when the size of the prior parameters doesn't match with the
            size of the variable.

        """
        linear_priors = filter_priors(self.priors, prior_type, with_mat=True)
        if len(linear_priors) == 0:
            return np.empty((2, 0)), np.empty((0, self.size))
        if not all(prior.shape[1] == self.size for prior in linear_priors):
            raise ValueError("Variable and prior size don't match.")
        params = np.hstack([prior.params for prior in linear_priors])
        mat = np.vstack([prior.mat for prior in linear_priors])
        return params, mat

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(component={self.component}, priors={self.priors})"
        )
