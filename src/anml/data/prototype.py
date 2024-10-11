from operator import attrgetter
from typing import Dict

from anml.data.component import Component
from pandas import DataFrame


class DataPrototype:
    """A set of components for easy validation and accessing the data frame.
    The class will automatically render the values in the `components` as the
    attributes of the class with the keys as the names of the attributes.  For
    an example please check the class :class:`anml.data.example.DataExample`.

    Parameters
    ----------
    components
        Components to validate and access data in a data frame.

    """

    components: Dict[str, Component] = property(attrgetter("_components"))
    """Components to validate and access data in a data frame.

    """

    def __init__(self, components: Dict[str, Component]):
        self.components = components
        for name, component in components.items():
            setattr(self, name, component)

    @components.setter
    def components(self, components: Dict[str, Component]):
        for name, component in components.items():
            if not isinstance(name, str):
                raise TypeError("Components key must be a string.")
            if not isinstance(component, Component):
                raise TypeError(
                    f"Components {name} value must be a instance " "of Component"
                )
        self._components = components

    def attach(self, df: DataFrame):
        """Attach data frame to every component.

        Parameters
        ----------
        df
            Given data frame.

        """
        for name in self.components.keys():
            getattr(self, name).attach(df)

    def clear(self):
        """Clear stored value for each component."""
        for name in self.components.keys():
            getattr(self, name).clear()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(components={self.components})"
