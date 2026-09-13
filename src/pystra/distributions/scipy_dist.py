"""Adapt frozen continuous SciPy distributions to the marginal interface."""

from scipy.stats._distn_infrastructure import rv_frozen
from scipy.stats import rv_continuous

from .distribution import Distribution
from ..errors import ModelError

__all__ = ["ScipyDist"]


class ScipyDist(Distribution):
    """Wrapper for a frozen continuous SciPy distribution.

    Discrete random variables are not supported.

    Parameters
    ----------
    name : str
        Name of the random variable, matching a limit-state argument.
    dist_obj : scipy.stats.rv_frozen
        Frozen continuous SciPy distribution to wrap.
    start_point : float, optional
        Starting point for the design-point search. Defaults to the mean.
    """

    _native_parameters = ()

    def _parameter_values(self):
        return {"dist_obj": self.dist_obj}

    @property
    def sensitivity_params(self) -> dict[str, float]:
        """No generic moment perturbation; replace the constructor inputs."""
        return {}

    def __init__(
        self, name: str, dist_obj: rv_frozen, start_point: float | None = None
    ) -> None:
        if not isinstance(dist_obj, rv_frozen):
            raise ModelError(
                f"ScipyDist {name} requires a frozen Scipy distribution object"
            )
        if not isinstance(dist_obj.dist, rv_continuous):
            raise ModelError(f"ScipyDist {name} requires a continuous distribution")

        super().__init__(
            name=name,
            dist_obj=dist_obj,
            start_point=start_point,
        )

        self.dist_type = "ScipyDist"
