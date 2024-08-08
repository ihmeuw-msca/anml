from collections.abc import Iterable
from operator import attrgetter
from typing import Any, List, Optional

from anml.data.validator import Validator
from pandas import DataFrame


class Component:
    """Component class validates, accesses and stores data from a data frame.

    Parameters
    ----------
    key
        Key to access the column in a data frame.
    validators
        A list of validators to check the corresponding column value in a data
        frame. Default to `None`.
    default_value
        A default value used to create the column when the key doesn't exist in
        the data frame. If `default_value=None` and the data frame doesn't
        contain the key, a `KeyError` will be raised. Default to `None`.

    """

    key = property(attrgetter("_key"))
    """Key to access the column in a data frame.

    """
    value = property(attrgetter("_value"))
    """Stored value from a data frame. This property can only be modified
    through class functions.

    """
    validators = property(attrgetter("_validators"))
    """A list of validators to check the corresponding column value in a data
    frame.

    """

    def __init__(
        self,
        key: str,
        validators: Optional[List[Validator]] = None,
        default_value: Optional[Any] = None,
    ):
        self.key = key
        self.validators = validators
        self.default_value = default_value
        self._value = None

    @key.setter
    def key(self, key: str):
        if not isinstance(key, str):
            raise TypeError("Key of the component must be a string.")
        self._key = key

    @validators.setter
    def validators(self, validators: Optional[List[Validator]]):
        if validators is None:
            self._validators = []
        else:
            if (not isinstance(validators, Iterable)) or (
                not all(isinstance(validator, Validator) for validator in validators)
            ):
                raise TypeError("Validators must be a list of validator.")
            self._validators = list(validators)

    def attach(self, df: DataFrame):
        """Validate, fill and store value from a data frame.

        Parameters
        ----------
        df
            Given data frame.

        Raises
        ------
        KeyError
            Raised when data frame doesn't contain the key and there is no
            default value.

        """
        if self.key not in df:
            if self.default_value is None:
                raise KeyError(f"Dataframe doesn't contain column {self.key}.")
            df[self.key] = self.default_value
        value = df[self.key].to_numpy()
        for validator in self.validators:
            validator(self.key, value)
        self._value = value

    def clear(self):
        """Clear stored value."""
        self._value = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(key={self.key})"
