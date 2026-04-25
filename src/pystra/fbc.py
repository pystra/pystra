# -*- coding: utf-8 -*-

from .distributions import Distribution, Maximum, MaxParent


class FBCProcess:
    """Ferry-Borges-Castanheta rectangular-wave load process.

    The Ferry-Borges-Castanheta (FBC) model represents a load process as a
    sequence of independent rectangular pulses of equal duration.  Within
    each basic interval the process is constant; between intervals a new
    value is drawn from the interval parent distribution.

    This class is intentionally a small distribution factory.  It does not
    run a reliability analysis and does not create load combinations by
    itself.  It only returns ordinary Pystra distribution objects that can be
    added to a :class:`~pystra.model.StochasticModel` or used in
    :class:`~pystra.loadcomb.LoadCombination` cases.

    Load combinations are a separate modelling choice.  For example,
    :meth:`pystra.loadcomb.LoadCombination.turkstra` uses FBC process objects
    to generate leading-action cases according to Turkstra's rule.

    Parameters
    ----------
    name : str
        Name of the load process.  This must match the name of ``parent`` and
        the corresponding argument in the limit-state function.
    parent : Distribution
        Distribution of the process value in one basic interval.
    basic_interval : float
        Duration of one rectangular-wave interval, in the same time units used
        for ``duration`` arguments passed to :meth:`maximum`.

    Notes
    -----
    For a duration :math:`T` and basic interval :math:`\\tau`, the maximum is
    represented by ``Maximum(parent, N=T/tau)``.  Durations shorter than one
    basic interval are treated as one interval.

    Examples
    --------
    >>> import pystra as ra
    >>> Q = ra.FBCProcess(
    ...     "Q", parent=ra.Gumbel("Q", 0.89, 0.2), basic_interval=1/52
    ... )
    >>> Q.point_in_time()
    >>> Q.maximum(duration=50)
    """

    def __init__(self, name, parent, basic_interval):
        if not isinstance(parent, Distribution):
            raise Exception("FBCProcess parent must be a Pystra Distribution")
        if basic_interval <= 0:
            raise Exception("FBCProcess basic_interval must be positive")
        if parent.getName() != name:
            raise Exception("FBCProcess name must match parent distribution name")

        self.name = name
        self.parent = parent
        self.basic_interval = basic_interval

    @classmethod
    def from_maximum(cls, name, maximum, maximum_duration, basic_interval):
        """Create a process from a known maximum distribution.

        This is useful when a code or model supplies, for example, an annual
        maximum distribution but the FBC process requires the distribution of
        one basic interval.  The interval parent is inferred using
        :class:`~pystra.distributions.parent.MaxParent`.

        Parameters
        ----------
        name : str
            Process and random-variable name.
        maximum : Distribution
            Distribution of the maximum over ``maximum_duration``.
        maximum_duration : float
            Duration represented by ``maximum``.
        basic_interval : float
            Basic interval of the rectangular-wave process.

        Returns
        -------
        FBCProcess
            Process whose parent distribution is inferred from ``maximum``.
        """
        if maximum_duration <= 0:
            raise Exception("FBCProcess maximum_duration must be positive")
        if basic_interval <= 0:
            raise Exception("FBCProcess basic_interval must be positive")
        if not isinstance(maximum, Distribution):
            raise Exception("FBCProcess maximum must be a Pystra Distribution")
        if maximum.getName() != name:
            raise Exception("FBCProcess name must match maximum distribution name")

        n = max(1.0, maximum_duration / basic_interval)
        parent = MaxParent(name, maximum, N=n)
        return cls(name, parent, basic_interval)

    def interval_count(self, duration=None, n=None):
        """Return the number of basic intervals in a duration.

        Either ``duration`` or ``n`` may be supplied, but not both.  Durations
        shorter than one basic interval are treated as one interval, which is
        the parent distribution.

        Parameters
        ----------
        duration : float, optional
            Duration over which the process is observed.
        n : float, optional
            Direct number of basic intervals.

        Returns
        -------
        float
            Number of basic intervals used in the maximum distribution.
        """
        if (duration is None) == (n is None):
            raise Exception("Specify exactly one of duration or n")
        if n is not None:
            if n < 1.0:
                raise Exception("FBCProcess n must be >= 1.0")
            return n
        if duration <= 0:
            raise Exception("FBCProcess duration must be positive")
        return max(1.0, duration / self.basic_interval)

    def point_in_time(self):
        """Return the interval parent distribution.

        Returns
        -------
        Distribution
            Distribution of the process value in one basic interval.
        """
        return self.parent

    def maximum(self, duration=None, n=None):
        """Return the distribution of the process maximum.

        Parameters
        ----------
        duration : float, optional
            Duration over which the maximum is taken.
        n : float, optional
            Direct number of basic intervals over which the maximum is taken.

        Returns
        -------
        Maximum
            Distribution of the maximum over the requested duration or number
            of intervals.
        """
        n_intervals = self.interval_count(duration=duration, n=n)
        return Maximum(self.name, self.parent, N=n_intervals)
