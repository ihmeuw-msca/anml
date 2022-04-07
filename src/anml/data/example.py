from anml.data.component import Component
from anml.data.prototype import DataPrototype
from anml.data.validator import NoNans, Positive


class DataExample(DataPrototype):
    """An example class for simple least-square problem. And for that purpose we
    only need observations and their standard deviations.

    Parameters
    ----------
    obs
        The observation column name in the data frame.
    obs_se
        The observation standard deviation column name in the data frame.

    """

    def __init__(self, obs: str, obs_se: str):
        obs = Component(obs, [NoNans()])
        obs_se = Component(obs_se, [NoNans(), Positive()], default_value=1.0)
        components = {
            "obs": obs,
            "obs_se": obs_se,
        }
        super().__init__(components)
