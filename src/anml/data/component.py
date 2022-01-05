from operator import attrgetter
from typing import Any, Callable, List, Optional

import pandas as pd
from anml.data.validator import validate_validator


class Component:
    """Component class validates, accesses and stores data from a data frame.
    """

    key = property(attrgetter("_key"))
    value = property(attrgetter("_value"))
    validators = property(attrgetter("_validators"))

    def __init__(self,
                 key: str,
                 validators: Optional[List[Callable]] = None,
                 default_value: Optional[Any] = None):
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
    def validators(self, validators: Optional[List[Callable]]):
        if validators is None:
            self._validators = []
        else:
            for validator in validators:
                validate_validator(validator)
            self._validators = validators

    def attach(self, df: pd.DataFrame):
        if self.key not in df:
            if self.default_value is None:
                raise KeyError(f"Dataframe doesn't contain column {self.key}.")
            df[self.key] = self.default_value
        value = df[self.key].values
        for validator in self.validators:
            validator(self.key, value)
        self._value = value

    def clear(self):
        self._value = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(key={self.key})"
