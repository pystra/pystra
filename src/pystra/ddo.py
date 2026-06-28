"""Design decision optimization and societal risk acceptance.

The objects in this module sit above the reliability methods.  They do not
change how FORM, SORM, or simulation analyses are run; instead
:class:`DDO` evaluates an objective subject to a selected acceptability
criterion.  The initial implementation
provides the :class:`LQI` criterion for minimum acceptable life-safety levels
using the life quality index (LQI) and societal willingness-to-pay (SWTP)
values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm

SWTP_COUNTRY_SOURCE = (
    "Rackwitz, JCSS LQI philosophy background document, Table 7. "
    "Values are G_delta_lbar in 10^6 PPPUS$."
)
SWTP_TARGET_SOURCE = "Fischer, Barnardo, and Faber (2012), LQI Symposium, Table 3."
SWTP_INDEX_SOURCE = (
    "World Bank World Development Indicators, NY.GDP.PCAP.PP.CD, "
    "GDP per capita, PPP (current international $). API last updated "
    "2026-04-08; values checked 2026-06-23."
)
SWTP_INDEX_INDICATOR = "NY.GDP.PCAP.PP.CD"


@dataclass(frozen=True)
class SWTPIndexRecord:
    """Index data used to update an anchored SWTP value.

    The built-in index records use GDP per capita in current international
    PPP dollars as a practical update of the LQI income term ``g``.
    """

    code: str
    base_year: int
    target_year: int
    base_index: float
    target_index: float
    factor: float
    indicator: str = SWTP_INDEX_INDICATOR
    source: str = SWTP_INDEX_SOURCE


@dataclass(frozen=True)
class SWTPRecord:
    """Country-level societal willingness-to-pay value.

    Parameters
    ----------
    code : str
        ISO-style country code.  The United Kingdom is stored as ``GB``;
        ``UK`` is accepted as an alias by :func:`get_swtp_record`.
    country : str
        Country name used in the source table.
    value_per_life : float
        Societal willingness to pay per statistical life, expressed in
        ``currency`` units.
    currency : str
        Currency basis for the value.
    price_year : int
        Price year of the source value.
    source : str
        Literature source for the value.
    notes : str
        Short note describing how the source value was obtained.
    """

    code: str
    country: str
    value_per_life: float
    currency: str = "PPPUSD"
    price_year: int = 1999
    source: str = SWTP_COUNTRY_SOURCE
    notes: str = (
        "Constant additive mortality-change value at all ages from "
        "predictive cohort tables."
    )

    @property
    def value_million(self) -> float:
        """Return the SWTP value in millions of ``currency`` units."""

        return self.value_per_life / 1_000_000.0


def _record(code: str, country: str, value_million: float) -> SWTPRecord:
    return SWTPRecord(code=code, country=country, value_per_life=value_million * 1e6)


SWTP_COUNTRY_VALUES = {
    "CA": _record("CA", "Canada", 1.8),
    "US": _record("US", "USA", 2.1),
    "AT": _record("AT", "Austria", 1.9),
    "BE": _record("BE", "Belgium", 2.4),
    "CZ": _record("CZ", "Czech Republic", 0.54),
    "DK": _record("DK", "Denmark", 1.7),
    "FI": _record("FI", "Finland", 1.3),
    "FR": _record("FR", "France", 1.9),
    "DE": _record("DE", "Germany", 1.9),
    "IT": _record("IT", "Italy", 1.8),
    "NL": _record("NL", "Netherlands", 2.8),
    "NO": _record("NO", "Norway", 1.8),
    "ES": _record("ES", "Spain", 1.3),
    "SE": _record("SE", "Sweden", 1.5),
    "CH": _record("CH", "Switzerland", 1.8),
    "GB": _record("GB", "United Kingdom", 1.7),
    "JP": _record("JP", "Japan", 1.3),
    "NZ": _record("NZ", "New Zealand", 1.3),
}


def _index(code: str, base_index: float, target_index: float) -> SWTPIndexRecord:
    return SWTPIndexRecord(
        code=code,
        base_year=1999,
        target_year=2024,
        base_index=base_index,
        target_index=target_index,
        factor=target_index / base_index,
    )


SWTP_GDP_PPP_INDEX_2024 = {
    "CA": _index("CA", 27841.35, 64610.38),
    "US": _index("US", 34515.38, 85809.90),
    "AT": _index("AT", 27500.08, 73911.44),
    "BE": _index("BE", 25440.84, 73514.48),
    "CZ": _index("CZ", 15493.98, 57285.42),
    "DK": _index("DK", 26642.22, 81878.23),
    "FI": _index("FI", 24761.82, 65378.38),
    "FR": _index("FR", 24203.88, 62556.92),
    "DE": _index("DE", 26517.93, 73551.93),
    "IT": _index("IT", 25654.68, 62014.27),
    "NL": _index("NL", 29316.26, 86173.63),
    "NO": _index("NO", 30573.73, 102037.53),
    "ES": _index("ES", 19938.18, 57965.29),
    "SE": _index("SE", 27496.44, 71844.91),
    "CH": _index("CH", 34742.63, 96497.69),
    "GB": _index("GB", 24493.50, 62009.49),
    "JP": _index("JP", 25735.97, 52039.17),
    "NZ": _index("NZ", 20579.37, 55551.14),
}

SWTP_ALIASES = {
    "UK": "GB",
    "UNITED KINGDOM": "GB",
    "U.K.": "GB",
    "USA": "US",
    "UNITED STATES": "US",
    "CZECH REP.": "CZ",
    "CZECH REPUBLIC": "CZ",
    "N. ZEALAND": "NZ",
    "NEW ZEALAND": "NZ",
}


def _normalise_country_code(code: str) -> str:
    key = str(code).strip().upper()
    return SWTP_ALIASES.get(key, key)


def _index_table(index_table: Optional[Mapping[str, SWTPIndexRecord]] = None):
    return SWTP_GDP_PPP_INDEX_2024 if index_table is None else index_table


def get_swtp_index_record(
    code: str, index_table: Optional[Mapping[str, SWTPIndexRecord]] = None
) -> SWTPIndexRecord:
    """Return the SWTP index record for a country code."""

    key = _normalise_country_code(code)
    table = _index_table(index_table)
    try:
        return table[key]
    except KeyError as exc:
        available = ", ".join(sorted(table))
        raise KeyError(
            f"Unknown SWTP index country code {code!r}. Available: {available}"
        ) from exc


def get_swtp_record(
    code: str,
    indexed: bool = False,
    index_table: Optional[Mapping[str, SWTPIndexRecord]] = None,
) -> SWTPRecord:
    """Return a country-level SWTP record.

    Parameters
    ----------
    code : str
        Country code or supported country-name alias.
    indexed : bool, optional
        If ``True``, return the Rackwitz value indexed with the built-in
        World Bank GDP per capita PPP factors.
    index_table : mapping, optional
        Alternate country-code mapping of :class:`SWTPIndexRecord` objects.

    Returns
    -------
    SWTPRecord
        Source-backed SWTP value.
    """

    if indexed:
        return index_swtp_record(code, index_table=index_table)

    key = _normalise_country_code(code)
    try:
        return SWTP_COUNTRY_VALUES[key]
    except KeyError as exc:
        available = ", ".join(sorted(SWTP_COUNTRY_VALUES))
        raise KeyError(
            f"Unknown SWTP country code {code!r}. Available: {available}"
        ) from exc


def index_swtp_record(
    code: str, index_table: Optional[Mapping[str, SWTPIndexRecord]] = None
) -> SWTPRecord:
    """Return a Rackwitz SWTP record indexed to a newer target year."""

    base = get_swtp_record(code)
    index = get_swtp_index_record(code, index_table=index_table)
    return SWTPRecord(
        code=base.code,
        country=base.country,
        value_per_life=base.value_per_life * index.factor,
        currency="current international PPPUSD",
        price_year=index.target_year,
        source=f"{base.source}; indexed using {index.source}",
        notes=(
            f"Anchored to {base.price_year} Rackwitz value and indexed by "
            f"{index.indicator} from {index.base_year} to {index.target_year}; "
            f"factor={index.factor:.4f}."
        ),
    )


def get_swtp(
    code: str,
    indexed: bool = False,
    index_table: Optional[Mapping[str, SWTPIndexRecord]] = None,
) -> float:
    """Return the SWTP value per statistical life for a country code."""

    return get_swtp_record(
        code, indexed=indexed, index_table=index_table
    ).value_per_life


def swtp_table(
    indexed: bool = False,
    index_table: Optional[Mapping[str, SWTPIndexRecord]] = None,
) -> pd.DataFrame:
    """Return the built-in country SWTP table as a dataframe."""

    if not indexed:
        records = [asdict(record) for record in SWTP_COUNTRY_VALUES.values()]
        return pd.DataFrame.from_records(records).set_index("code")

    records = []
    for code, base in SWTP_COUNTRY_VALUES.items():
        index = get_swtp_index_record(code, index_table=index_table)
        record = asdict(index_swtp_record(code, index_table=index_table))
        record["anchor_value_per_life"] = base.value_per_life
        record["anchor_price_year"] = base.price_year
        record["index_factor"] = index.factor
        record["index_base_value"] = index.base_index
        record["index_target_value"] = index.target_index
        record["index_indicator"] = index.indicator
        records.append(record)
    return pd.DataFrame.from_records(records).set_index("code")


@dataclass(frozen=True)
class SWTP:
    """Societal willingness to pay per statistical life."""

    value_per_life: float
    currency: str = "PPPUSD"
    price_year: Optional[int] = None
    source: Optional[str] = None

    def __post_init__(self):
        if self.value_per_life <= 0:
            raise ValueError("SWTP value_per_life must be positive")

    @classmethod
    def from_country(cls, code: str, indexed: Optional[bool] = None) -> "SWTP":
        """Create an SWTP value from the built-in country table.

        ``indexed`` must be supplied explicitly.  Use ``indexed=False`` for
        the Rackwitz 1999 anchor and ``indexed=True`` for the built-in indexed
        current-PPP view.
        """

        record = get_swtp_record(code, indexed=_require_explicit_indexed(indexed))
        return cls(
            value_per_life=record.value_per_life,
            currency=record.currency,
            price_year=record.price_year,
            source=record.source,
        )

    @classmethod
    def from_lqi(
        cls,
        gross_domestic_product_per_capita: float,
        work_leisure_parameter: float,
        demographic_constant: float,
        currency: str = "currency units",
        price_year: Optional[int] = None,
        source: Optional[str] = "LQI relation SWTP = g / q * G",
    ) -> "SWTP":
        """Create an SWTP value from the LQI relation.

        Parameters
        ----------
        gross_domestic_product_per_capita : float
            Gross domestic product per person, denoted ``g`` in the LQI
            literature.
        work_leisure_parameter : float
            The dimensionless LQI work--leisure (income-elasticity) parameter
            ``q``, typically about 0.1--0.2 (e.g. 0.175 in Schubert and Faber,
            2009).  This is *not* an annual mortality rate.
        demographic_constant : float
            Demographic life-time constant multiplying ``g / q``.
        currency : str, optional
            Currency of ``gross_domestic_product_per_capita``.
        price_year : int, optional
            Price year for the resulting value.
        source : str, optional
            Source note carried with the value.
        """

        if work_leisure_parameter <= 0:
            raise ValueError("work_leisure_parameter must be positive")
        if demographic_constant <= 0:
            raise ValueError("demographic_constant must be positive")
        value = (
            gross_domestic_product_per_capita
            / work_leisure_parameter
            * demographic_constant
        )
        return cls(
            value_per_life=value,
            currency=currency,
            price_year=price_year,
            source=source,
        )

    def for_lives(self, expected_fatalities: float) -> float:
        """Return the SWTP-equivalent consequence for expected fatalities."""

        if expected_fatalities < 0:
            raise ValueError("expected_fatalities must be non-negative")
        return self.value_per_life * expected_fatalities


@dataclass(frozen=True)
class FatalityConsequence:
    """Expected fatalities conditional on structural failure."""

    people_exposed: float
    probability_death_given_failure: float = 1.0

    def __post_init__(self):
        if self.people_exposed < 0:
            raise ValueError("people_exposed must be non-negative")
        if not 0 <= self.probability_death_given_failure <= 1:
            raise ValueError("probability_death_given_failure must be in [0, 1]")

    @property
    def expected_fatalities_given_failure(self) -> float:
        """Return expected fatalities conditional on failure."""

        return self.people_exposed * self.probability_death_given_failure


@dataclass(frozen=True)
class TargetReliability:
    """Target failure probability and reliability index from a calibration.

    A single result type for every target-reliability route in this module: the
    rounded LQI table lookup (:meth:`LQI.lookup_target`), the LQI marginal
    optimization (:meth:`LQI.derive_target`), and the Rackwitz/Steenbergen
    code-calibration model (:meth:`RackwitzTargetModel.calibrate`).  Only ``pf``,
    ``beta`` and ``method`` are always populated; fields that do not apply to a
    given route are left as ``None``.

    Parameters
    ----------
    pf : float
        Annual failure probability or rate at the target.
    beta : float
        Reliability index corresponding to ``pf``.
    method : str
        Route that produced the target, e.g. ``"lqi-table"``,
        ``"LQI marginal"`` or ``"Rackwitz/Steenbergen"``.
    k1 : float, optional
        LQI safety cost ratio, when the target comes from the LQI route.
    cost_class : str, optional
        Discrete cost-class label from the rounded LQI table.
    variability : str, optional
        Variability class used for the rounded LQI table.
    design : float, optional
        Optimizing design parameter for calculated targets (the mean
        resistance-to-load ratio in the Rackwitz examples).
    objective : float, optional
        Objective value at the optimizing design.
    converged : bool, optional
        Whether the underlying optimization converged.
    message : str, optional
        Solver message for calculated targets.
    source : str, optional
        Literature source carried with a looked-up target.
    metadata : mapping, optional
        Additional model parameters or labels.
    """

    pf: float
    beta: float
    method: str
    k1: Optional[float] = None
    cost_class: Optional[str] = None
    variability: Optional[str] = None
    design: Optional[float] = None
    objective: Optional[float] = None
    converged: Optional[bool] = None
    message: str = ""
    source: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.pf <= 1.0:
            raise ValueError("pf must be in [0, 1]")

    def to_dict(self) -> dict:
        """Return the populated scalar quantities as a dictionary."""

        data = {"method": self.method, "pf": self.pf, "beta": self.beta}
        for name in (
            "design",
            "objective",
            "k1",
            "cost_class",
            "variability",
            "converged",
            "source",
        ):
            value = getattr(self, name)
            if value is not None:
                data[name] = value
        data.update(self.metadata or {})
        return data

    def for_period(
        self, years: float, dependence_interval: float = 1.0
    ) -> "TargetReliability":
        """Return the target converted to a reference period of ``years``.

        The annual failure probability ``pf`` is compounded over ``years``,
        assuming the governing maximum renews every ``dependence_interval``
        years.  ``dependence_interval=1`` treats the annual maxima as
        independent (the most onerous case), while
        ``dependence_interval=years`` treats the period as fully dependent (a
        single renewal) and leaves the annual ``pf`` unchanged.

        The conversion uses ``pf`` directly rather than re-deriving an annual
        probability from ``beta``; for rounded table targets the two are not an
        exact pair, and compounding ``pf`` is the correct quantity.

        Parameters
        ----------
        years : float
            Reference period in years.
        dependence_interval : float, optional
            Renewal interval of the governing maximum, in years; must lie in
            ``(0, years]``.

        Returns
        -------
        TargetReliability
            A new target at the reference period, with the original ``method``
            and class labels retained and ``reference_period_years`` /
            ``dependence_interval`` recorded in ``metadata``.
        """

        if years <= 0:
            raise ValueError("years must be positive")
        if not 0 < dependence_interval <= years:
            raise ValueError("dependence_interval must be in (0, years]")

        annual_pf = float(self.pf)
        periods = years / dependence_interval
        period_pf = 1.0 - (1.0 - annual_pf) ** periods
        metadata = dict(self.metadata or {})
        metadata["reference_period_years"] = years
        metadata["dependence_interval"] = dependence_interval
        return replace(
            self,
            pf=period_pf,
            beta=_beta_from_failure_probability(period_pf),
            metadata=metadata,
        )


@dataclass(frozen=True)
class RiskResult:
    """Expected annual risk quantities for a component or scenario model.

    Parameters
    ----------
    annual_failure_rate : float
        Annual probability or rate of the represented failure event.  For
        scenario studies this may be the sum of joint failure-state rates.
    expected_fatalities : float
        Expected fatalities per year.
    expected_economic_loss : float
        Expected economic loss per year, excluding SWTP life-safety valuation.
    scenarios : pandas.DataFrame, optional
        Scenario table used to derive the expected values.  This is useful for
        joint failure states and nonlinear consequence models.
    metadata : mapping, optional
        User-supplied context such as model name, design point, or units.
    """

    annual_failure_rate: float
    expected_fatalities: float = 0.0
    expected_economic_loss: float = 0.0
    scenarios: Optional[pd.DataFrame] = None
    metadata: Optional[Mapping[str, Any]] = None

    def __post_init__(self):
        if self.annual_failure_rate < 0:
            raise ValueError("annual_failure_rate must be non-negative")
        if self.expected_fatalities < 0:
            raise ValueError("expected_fatalities must be non-negative")
        if self.expected_economic_loss < 0:
            raise ValueError("expected_economic_loss must be non-negative")
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

    @classmethod
    def from_scenarios(
        cls,
        scenarios: Any,
        weight_col: str = "probability",
        fatalities_col: str = "fatalities",
        economic_loss_col: str = "economic_loss",
        failure_col: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "RiskResult":
        """Aggregate a scenario table into annual risk quantities.

        The scenario weights may be annual probabilities or annual rates.  No
        independence assumptions are made, so rows may represent joint
        failure states produced by an upstream reliability model.
        If ``failure_col`` is omitted, every row is treated as a failure/loss
        scenario for the purpose of ``annual_failure_rate``.
        """

        df = pd.DataFrame(scenarios).copy()
        for column in (weight_col, fatalities_col, economic_loss_col):
            if column not in df:
                raise KeyError(f"Scenario table is missing column {column!r}")
        if failure_col is not None and failure_col not in df:
            raise KeyError(f"Scenario table is missing column {failure_col!r}")

        weights = df[weight_col].astype(float)
        fatalities = df[fatalities_col].astype(float)
        economic_loss = df[economic_loss_col].astype(float)

        if (weights < 0).any():
            raise ValueError("scenario weights must be non-negative")
        if (fatalities < 0).any():
            raise ValueError("scenario fatalities must be non-negative")
        if (economic_loss < 0).any():
            raise ValueError("scenario economic losses must be non-negative")

        if failure_col is None:
            annual_failure_rate = float(weights.sum())
        else:
            annual_failure_rate = float((weights * df[failure_col].astype(bool)).sum())

        df["expected_fatalities_contribution"] = weights * fatalities
        df["expected_economic_loss_contribution"] = weights * economic_loss

        return cls(
            annual_failure_rate=annual_failure_rate,
            expected_fatalities=float(df["expected_fatalities_contribution"].sum()),
            expected_economic_loss=float(
                df["expected_economic_loss_contribution"].sum()
            ),
            scenarios=df,
            metadata=metadata,
        )

    @property
    def beta(self) -> Optional[float]:
        """Return the generalized reliability index when rate is probability-like."""

        if self.annual_failure_rate == 0:
            return float("inf")
        if 0 < self.annual_failure_rate < 1:
            return -float(norm.ppf(self.annual_failure_rate))
        return None

    def getFailure(self) -> float:
        """Return ``annual_failure_rate`` for reliability-result compatibility."""

        return self.annual_failure_rate

    def getBeta(self) -> Optional[float]:
        """Return ``beta`` for reliability-result compatibility."""

        return self.beta

    def life_safety_cost(self, swtp: Union[float, SWTP]) -> float:
        """Return SWTP-valued expected annual life-safety cost."""

        return _swtp_value(swtp) * self.expected_fatalities

    def total_risk_cost(
        self,
        swtp: Optional[Union[float, SWTP]] = None,
        include_life_safety: bool = True,
    ) -> float:
        """Return expected annual economic plus optional life-safety risk cost."""

        total = self.expected_economic_loss
        if include_life_safety and swtp is not None:
            total += self.life_safety_cost(swtp)
        return float(total)

    def to_dict(self) -> dict:
        """Return scalar risk quantities as a dictionary."""

        data = {
            "annual_failure_rate": self.annual_failure_rate,
            "beta": self.beta,
            "expected_fatalities": self.expected_fatalities,
            "expected_economic_loss": self.expected_economic_loss,
        }
        data.update(self.metadata or {})
        return data


@dataclass
class ScenarioRiskModel:
    """Scenario-table risk model for aggregated risk studies.

    ``scenarios`` may be a dataframe-like object or a callable returning one.
    Callable scenarios are evaluated with the design value when supplied.
    """

    scenarios: Union[Any, Callable[..., Any]]
    weight_col: str = "probability"
    fatalities_col: str = "fatalities"
    economic_loss_col: str = "economic_loss"
    failure_col: Optional[str] = None
    metadata: Optional[Mapping[str, Any]] = None

    def evaluate(self, design: Any = None) -> RiskResult:
        """Evaluate the scenario model and return a :class:`RiskResult`."""

        if callable(self.scenarios):
            scenario_data = (
                self.scenarios(design) if design is not None else self.scenarios()
            )
        else:
            scenario_data = self.scenarios

        metadata = dict(self.metadata or {})
        if design is not None:
            metadata.setdefault("design", design)

        return RiskResult.from_scenarios(
            scenario_data,
            weight_col=self.weight_col,
            fatalities_col=self.fatalities_col,
            economic_loss_col=self.economic_loss_col,
            failure_col=self.failure_col,
            metadata=metadata,
        )


_TARGET_TABLE = (
    ("large", 1e-3, 1e-2, 1e-3, 3.1),
    ("medium", 1e-4, 1e-3, 1e-4, 3.7),
    ("small", 1e-5, 1e-4, 1e-5, 4.2),
)
_VARIABILITY_FACTORS = {"medium": 1.0, "high": 5.0, "low": 0.5}


def lqi_k1(
    safety_cost_rate: float,
    swtp: Union[float, SWTP],
    expected_fatalities_given_failure: float,
) -> float:
    """Return the LQI safety cost ratio ``K1``.

    The Fischer, Barnardo, and Faber table uses a ratio of the marginal
    safety cost to the SWTP-weighted expected fatalities.  In their bridge
    notation the numerator is typically ``C1 * (gamma_s + omega)``.
    """

    value_per_life = swtp.value_per_life if isinstance(swtp, SWTP) else float(swtp)
    if safety_cost_rate <= 0:
        raise ValueError("safety_cost_rate must be positive")
    if value_per_life <= 0:
        raise ValueError("swtp must be positive")
    if expected_fatalities_given_failure <= 0:
        raise ValueError("expected_fatalities_given_failure must be positive")
    return safety_cost_rate / (value_per_life * expected_fatalities_given_failure)


def _swtp_value(swtp: Union[float, SWTP]) -> float:
    return swtp.value_per_life if isinstance(swtp, SWTP) else float(swtp)


def _as_scalar_or_array(value):
    array = np.asarray(value)
    return float(array) if array.ndim == 0 else array


def _beta_from_failure_probability(pf: float) -> float:
    if pf < 0 or pf > 1:
        raise ValueError("failure probability must be in [0, 1]")
    if pf == 0:
        return float("inf")
    if pf == 1:
        return float("-inf")
    return -float(norm.ppf(pf))


def lognormal_ratio_failure_probability(
    design: Union[float, np.ndarray],
    resistance_cov: float,
    load_cov: float,
) -> Union[float, np.ndarray]:
    """Return ``P(R - S <= 0)`` for independent lognormal resistance and load.

    ``design`` is the ratio ``E[R] / E[S]``.  The expression is the closed-form
    reliability model used in the Rackwitz/JCSS target-reliability examples.
    """

    if resistance_cov < 0:
        raise ValueError("resistance_cov must be non-negative")
    if load_cov < 0:
        raise ValueError("load_cov must be non-negative")
    if resistance_cov == 0 and load_cov == 0:
        raise ValueError("at least one coefficient of variation must be positive")

    design_ratio = np.asarray(design, dtype=float)
    if np.any(design_ratio <= 0):
        raise ValueError("design must be positive")

    numerator = np.log(
        design_ratio * np.sqrt((1.0 + load_cov**2) / (1.0 + resistance_cov**2))
    )
    denominator = np.sqrt(np.log((1.0 + resistance_cov**2) * (1.0 + load_cov**2)))
    return _as_scalar_or_array(norm.cdf(-numerator / denominator))


@dataclass
class TargetReliabilityCalibration:
    """One-dimensional calibration for deriving a target reliability.

    The calibration maximizes a user-supplied objective over a scalar design
    variable and then evaluates the associated failure probability.  It is the
    generic calculation layer behind the built-in Rackwitz and LQI marginal
    target helpers.
    """

    objective: Callable[[float], float]
    failure_probability: Callable[[float], float]
    bounds: tuple[float, float] = (1.0, 15.0)
    variable: str = "design"
    metadata: Optional[Mapping[str, Any]] = None

    def run(self) -> TargetReliability:
        """Run the bounded scalar optimization."""

        lower, upper = map(float, self.bounds)
        if lower >= upper:
            raise ValueError("bounds must be an increasing (lower, upper) pair")

        solution = minimize_scalar(
            lambda value: -float(self.objective(value)),
            bounds=(lower, upper),
            method="bounded",
        )
        design = float(solution.x)
        objective = float(self.objective(design))
        pf = float(self.failure_probability(design))
        beta = _beta_from_failure_probability(pf)

        metadata = dict(self.metadata or {})
        metadata.setdefault("variable", self.variable)

        return TargetReliability(
            pf=pf,
            beta=beta,
            method=str(metadata.get("method", "calibration")),
            design=design,
            objective=objective,
            converged=bool(solution.success),
            message=str(solution.message),
            metadata=metadata,
        )


@dataclass(frozen=True)
class RackwitzTargetModel:
    """Rackwitz/Steenbergen target-reliability model for code calibration.

    The model follows the normalized life-cycle objective used by Rackwitz and
    restated by Steenbergen, Rózsás, and Vrouwenvelder.  Every cost is expressed
    as a fraction of the base construction cost ``C0`` (``base_cost``), and the
    design variable is the mean resistance-to-load ratio ``p = E[R] / E[S]``.
    The construction cost is ``C(p) = C0 + C1 p = base_cost * (1 +
    safety_cost_ratio * p)``, so the cost inputs below are ratios to ``C0``.

    Parameters
    ----------
    safety_cost_ratio : float
        Marginal safety cost ``C1 / C0`` -- the extra construction cost per unit
        of ``p`` as a fraction of the base cost.  Larger values make safety
        relatively more expensive and lower the optimal target.
    failure_cost_ratio : float
        Failure (ULS) consequence cost ``H / C0``.
    resistance_cov, load_cov : float, optional
        Coefficients of variation of resistance and load in the closed-form
        lognormal ``P_f(p)`` model.
    base_cost : float, optional
        Base construction cost ``C0``.  Because every other cost is a ratio to
        it, its value does not change the calibrated target; defaults to 1.
    interest_rate : float, optional
        Discount/interest rate ``gamma``.
    obsolescence_rate : float, optional
        Obsolescence rate ``omega``.
    load_occurrence_rate : float, optional
        Load occurrence rate ``lambda`` (renewals per year).
    serviceability_cost_ratio : float, optional
        Serviceability (SLS) cost ``U / C0``.
    demolition_cost_ratio : float, optional
        Demolition/obsolescence cost ``A / C0``.
    serviceability_resistance_ratio : float, optional
        Ratio of the ULS to SLS resistance thresholds; the SLS check uses
        ``p / serviceability_resistance_ratio``.
    benefit_rate : float, optional
        Constant annual benefit ``b / C0``.  It is independent of ``p`` and so
        does not affect the optimum; defaults to 0.

    Notes
    -----
    To recalibrate a whole table for your own classes, pass ``safety_costs``
    (the ``C1 / C0`` values) and ``failure_costs`` (the ``H / C0`` values) to
    :meth:`table`.
    """

    safety_cost_ratio: float
    failure_cost_ratio: float
    resistance_cov: float = 0.3
    load_cov: float = 0.3
    base_cost: float = 1.0
    interest_rate: float = 0.035
    obsolescence_rate: float = 0.02
    load_occurrence_rate: float = 1.0
    serviceability_cost_ratio: float = 0.3
    demolition_cost_ratio: float = 0.2
    serviceability_resistance_ratio: float = 1.5
    benefit_rate: float = 0.0

    def __post_init__(self):
        if self.safety_cost_ratio <= 0:
            raise ValueError("safety_cost_ratio must be positive")
        if self.failure_cost_ratio < 0:
            raise ValueError("failure_cost_ratio must be non-negative")
        if self.base_cost <= 0:
            raise ValueError("base_cost must be positive")
        if self.interest_rate <= 0:
            raise ValueError("interest_rate must be positive")
        if self.obsolescence_rate < 0:
            raise ValueError("obsolescence_rate must be non-negative")
        if self.load_occurrence_rate < 0:
            raise ValueError("load_occurrence_rate must be non-negative")
        if self.serviceability_cost_ratio < 0:
            raise ValueError("serviceability_cost_ratio must be non-negative")
        if self.demolition_cost_ratio < 0:
            raise ValueError("demolition_cost_ratio must be non-negative")
        if self.serviceability_resistance_ratio <= 0:
            raise ValueError("serviceability_resistance_ratio must be positive")

    @property
    def metadata(self) -> dict:
        """Return source parameters used by the normalized model."""

        return {
            "method": "Rackwitz/Steenbergen",
            "safety_cost_ratio": self.safety_cost_ratio,
            "failure_cost_ratio": self.failure_cost_ratio,
            "resistance_cov": self.resistance_cov,
            "load_cov": self.load_cov,
            "base_cost": self.base_cost,
            "interest_rate": self.interest_rate,
            "obsolescence_rate": self.obsolescence_rate,
            "load_occurrence_rate": self.load_occurrence_rate,
            "serviceability_cost_ratio": self.serviceability_cost_ratio,
            "demolition_cost_ratio": self.demolition_cost_ratio,
            "serviceability_resistance_ratio": self.serviceability_resistance_ratio,
            "benefit_rate": self.benefit_rate,
        }

    def construction_cost(self, design: float) -> float:
        """Return ``C(p) = C0 + C1 p``."""

        if design <= 0:
            raise ValueError("design must be positive")
        return self.base_cost * (1.0 + self.safety_cost_ratio * float(design))

    def failure_probability(self, design: float) -> float:
        """Return the ULS failure probability for a design value."""

        return float(
            lognormal_ratio_failure_probability(
                design, self.resistance_cov, self.load_cov
            )
        )

    def serviceability_probability(self, design: float) -> float:
        """Return the SLS failure probability for a design value."""

        return float(
            lognormal_ratio_failure_probability(
                float(design) / self.serviceability_resistance_ratio,
                self.resistance_cov,
                self.load_cov,
            )
        )

    def objective(self, design: float) -> float:
        """Return the normalized Rackwitz/Steenbergen life-cycle objective."""

        construction = self.construction_cost(design)
        serviceability = self.base_cost * self.serviceability_cost_ratio
        demolition = self.base_cost * self.demolition_cost_ratio
        failure_cost = self.base_cost * self.failure_cost_ratio

        gamma = self.interest_rate
        benefit = self.base_cost * self.benefit_rate / gamma
        serviceability_loss = (
            serviceability
            * self.load_occurrence_rate
            / gamma
            * self.serviceability_probability(design)
        )
        obsolescence_loss = (construction + demolition) * self.obsolescence_rate / gamma
        failure_loss = (
            (construction + failure_cost)
            * self.load_occurrence_rate
            / gamma
            * self.failure_probability(design)
        )
        return float(
            benefit
            - construction
            - serviceability_loss
            - obsolescence_loss
            - failure_loss
        )

    def calibration(
        self, bounds: tuple[float, float] = (1.0, 15.0)
    ) -> TargetReliabilityCalibration:
        """Return a generic calibration object for this model."""

        return TargetReliabilityCalibration(
            objective=self.objective,
            failure_probability=self.failure_probability,
            bounds=bounds,
            variable="p",
            metadata=self.metadata,
        )

    def calibrate(self, bounds: tuple[float, float] = (1.0, 15.0)) -> TargetReliability:
        """Return the target reliability implied by this model."""

        return self.calibration(bounds=bounds).run()

    @classmethod
    def table(
        cls,
        safety_costs: Optional[Mapping[str, float]] = None,
        failure_costs: Optional[Mapping[str, float]] = None,
        bounds: tuple[float, float] = (1.0, 15.0),
        **kwargs,
    ) -> pd.DataFrame:
        """Calculate a Rackwitz-style target-reliability table.

        Parameters
        ----------
        safety_costs : mapping, optional
            Label -> ``C1 / C0`` (marginal safety cost) for each relative
            safety-cost class.  Defaults to representative ``large``/``normal``/
            ``small`` values.
        failure_costs : mapping, optional
            Label -> ``H / C0`` (failure consequence cost) for each consequence
            class.  Defaults to representative ``minor``/``moderate``/``large``
            values.
        bounds : tuple of float, optional
            Search interval for the optimizing ``p``.
        **kwargs
            Forwarded to :class:`RackwitzTargetModel` (e.g. ``resistance_cov``,
            ``interest_rate``) so every model setting can be varied too.

        Notes
        -----
        The defaults are representative values from the broad classes used in
        the literature, not a retyping of any rounded target table.  Supply your
        own ``safety_costs`` and ``failure_costs`` to recalibrate for different
        cost and consequence assumptions.
        """

        if safety_costs is None:
            safety_costs = {"large": 0.3, "normal": 0.03, "small": 0.003}
        if failure_costs is None:
            failure_costs = {"minor": 0.5, "moderate": 2.5, "large": 6.5}

        rows = []
        for safety_label, safety_cost_ratio in safety_costs.items():
            for failure_label, failure_cost_ratio in failure_costs.items():
                model = cls(
                    safety_cost_ratio=safety_cost_ratio,
                    failure_cost_ratio=failure_cost_ratio,
                    **kwargs,
                )
                result = model.calibrate(bounds=bounds)
                row = result.to_dict()
                row.update(
                    {
                        "relative_safety_cost": safety_label,
                        "failure_consequence": failure_label,
                    }
                )
                rows.append(row)

        columns = [
            "relative_safety_cost",
            "failure_consequence",
            "safety_cost_ratio",
            "failure_cost_ratio",
            "design",
            "pf",
            "beta",
            "objective",
            "resistance_cov",
            "load_cov",
            "base_cost",
            "interest_rate",
            "obsolescence_rate",
            "load_occurrence_rate",
            "serviceability_cost_ratio",
            "demolition_cost_ratio",
            "serviceability_resistance_ratio",
            "benefit_rate",
            "converged",
        ]
        return pd.DataFrame(rows, columns=columns)


def derive_lqi_target(
    k1: float,
    resistance_cov: float = 0.4,
    load_cov: float = 0.4,
    bounds: tuple[float, float] = (1.0, 20.0),
) -> TargetReliability:
    """Calculate an LQI target from the marginal optimization condition.

    This derives a target reliability by minimizing the non-dimensional
    expression ``K1 * p + P_f(p)`` for the same lognormal resistance-demand
    model used in the Rackwitz examples.  The table returned by
    :func:`lqi_target_reliability` remains the rounded source-table lookup.
    """

    if k1 <= 0:
        raise ValueError("k1 must be positive")

    def failure_probability(design):
        return lognormal_ratio_failure_probability(design, resistance_cov, load_cov)

    calibration = TargetReliabilityCalibration(
        objective=lambda design: -(k1 * design + failure_probability(design)),
        failure_probability=failure_probability,
        bounds=bounds,
        variable="p",
        metadata={
            "method": "LQI marginal",
            "k1": k1,
            "resistance_cov": resistance_cov,
            "load_cov": load_cov,
        },
    )
    return calibration.run()


def rackwitz_table(**kwargs) -> pd.DataFrame:
    """Calculate the default Rackwitz/Steenbergen target-reliability table."""

    return RackwitzTargetModel.table(**kwargs)


def jcss_lqi_risk_cost(
    safety_cost: Union[float, np.ndarray],
    failure_rate: Union[float, np.ndarray],
    swtp: Union[float, SWTP],
    expected_fatalities_given_failure: float,
):
    """Return the JCSS LQI life-safety risk cost.

    This implements the canonical LQI optimization term
    ``S(p) = C(p) + G_x k N_F h(p)`` from the JCSS background documents,
    where ``h(p)`` is a failure rate or annual failure probability.
    """

    value_per_life = _swtp_value(swtp)
    if value_per_life <= 0:
        raise ValueError("swtp must be positive")
    if expected_fatalities_given_failure < 0:
        raise ValueError("expected_fatalities_given_failure must be non-negative")
    result = np.asarray(
        safety_cost
    ) + value_per_life * expected_fatalities_given_failure * np.asarray(failure_rate)
    return _as_scalar_or_array(result)


def jcss_lqi_risk_cost_from_result(
    safety_cost: float,
    risk: RiskResult,
    swtp: Union[float, SWTP],
    include_economic_loss: bool = False,
) -> float:
    """Return JCSS LQI risk cost from an aggregated risk result.

    This form is intended for scenario studies where the upstream model already
    provides expected annual fatalities.  Set
    ``include_economic_loss=True`` when the objective should include nonlinear
    economic consequences in addition to the SWTP life-safety term.
    """

    if safety_cost < 0:
        raise ValueError("safety_cost must be non-negative")
    risk_cost = risk.life_safety_cost(swtp)
    if include_economic_loss:
        risk_cost += risk.expected_economic_loss
    return float(safety_cost + risk_cost)


def jcss_lqi_acceptability_margin(
    safety_cost_derivative: float,
    failure_rate_derivative: float,
    swtp: Union[float, SWTP],
    expected_fatalities_given_failure: float,
) -> float:
    """Return the JCSS marginal LQI acceptability margin.

    For a design parameter that increases safety, the JCSS condition is
    ``dC/dp >= -G_x k N_F dh/dp``.  The returned margin is therefore
    ``dC/dp + G_x k N_F dh/dp``; non-negative values satisfy the criterion.
    """

    value_per_life = _swtp_value(swtp)
    if value_per_life <= 0:
        raise ValueError("swtp must be positive")
    if expected_fatalities_given_failure < 0:
        raise ValueError("expected_fatalities_given_failure must be non-negative")
    return float(
        safety_cost_derivative
        + value_per_life * expected_fatalities_given_failure * failure_rate_derivative
    )


def finite_difference_derivative(
    function: Callable[[float], float],
    design: float,
    step: Optional[float] = None,
) -> float:
    """Return a central finite-difference derivative for a scalar design."""

    x = float(design)
    h = max(abs(x) * 1e-6, 1e-6) if step is None else float(step)
    if h <= 0:
        raise ValueError("step must be positive")
    return float((function(x + h) - function(x - h)) / (2.0 * h))


def jcss_lqi_acceptability(
    safety_cost: Callable[[float], float],
    failure_rate: Callable[[float], float],
    design: float,
    swtp: Union[float, SWTP],
    expected_fatalities_given_failure: float,
    step: Optional[float] = None,
) -> float:
    """Return the finite-difference JCSS LQI acceptability margin."""

    dcost = finite_difference_derivative(safety_cost, design, step=step)
    drate = finite_difference_derivative(failure_rate, design, step=step)
    return jcss_lqi_acceptability_margin(
        dcost, drate, swtp, expected_fatalities_given_failure
    )


def jcss_lqi_is_acceptable(
    safety_cost: Callable[[float], float],
    failure_rate: Callable[[float], float],
    design: float,
    swtp: Union[float, SWTP],
    expected_fatalities_given_failure: float,
    step: Optional[float] = None,
) -> bool:
    """Return ``True`` when the JCSS marginal LQI condition is satisfied."""

    margin = jcss_lqi_acceptability(
        safety_cost,
        failure_rate,
        design,
        swtp,
        expected_fatalities_given_failure,
        step=step,
    )
    return margin >= 0.0


def jcss_systematic_reconstruction_objective(
    benefit_rate: float,
    safety_cost: float,
    failure_rate: float,
    failure_consequence: float,
    discount_rate: float,
) -> float:
    """Return the JCSS infinite-horizon reconstruction objective.

    The expression follows the JCSS resistance-demand example:
    ``Z(p) = b / gamma - C(p) - (C(p) + H) h(p) / gamma``.
    """

    if discount_rate <= 0:
        raise ValueError("discount_rate must be positive")
    if not 0 <= failure_rate:
        raise ValueError("failure_rate must be non-negative")
    if safety_cost < 0:
        raise ValueError("safety_cost must be non-negative")
    if failure_consequence < 0:
        raise ValueError("failure_consequence must be non-negative")
    return float(
        benefit_rate / discount_rate
        - safety_cost
        - (safety_cost + failure_consequence) * failure_rate / discount_rate
    )


def lqi_target_reliability(k1: float, variability: str = "medium") -> TargetReliability:
    """Return an LQI target reliability for a safety cost ratio.

    Parameters
    ----------
    k1 : float
        Safety cost ratio.  Medium-variability classes follow the ranges in
        Fischer, Barnardo, and Faber (2012), Table 3.
    variability : {"low", "medium", "high"}, optional
        Approximate consequence/resistance variability adjustment.  The
        source table reports that high variability gives failure
        probabilities about five times larger and low variability about two
        times smaller than the medium case.
    """

    if k1 <= 0:
        raise ValueError("k1 must be positive")

    key = str(variability).strip().lower()
    if key not in _VARIABILITY_FACTORS:
        valid = ", ".join(sorted(_VARIABILITY_FACTORS))
        raise ValueError(f"variability must be one of: {valid}")
    factor = _VARIABILITY_FACTORS[key]

    cost_class = "extrapolated"
    pf = k1 / 5.0
    beta = -float(norm.ppf(pf))

    for name, lower, upper, table_pf, table_beta in _TARGET_TABLE:
        if lower <= k1 <= upper:
            cost_class = name
            pf = table_pf
            beta = table_beta
            break

    if factor != 1.0:
        pf = pf * factor
        pf = float(np.clip(pf, np.finfo(float).tiny, 1.0 - np.finfo(float).eps))
        beta = -float(norm.ppf(pf))

    return TargetReliability(
        pf=float(pf),
        beta=float(beta),
        method="lqi-table",
        k1=k1,
        cost_class=cost_class,
        variability=key,
        source=SWTP_TARGET_SOURCE,
    )


def present_value_factor(interest_rate: float, service_life: float) -> float:
    """Return the JCSS-style present value factor for a constant annual flow."""

    if service_life <= 0:
        raise ValueError("service_life must be positive")
    if interest_rate < 0:
        raise ValueError("interest_rate must be non-negative")
    if interest_rate == 0:
        return float(service_life)
    return float(
        (1.0 - (1.0 + interest_rate) ** (-service_life)) / np.log1p(interest_rate)
    )


def annualized_safety_cost(
    initial_cost: float,
    failure_probability: float,
    service_life: float,
    interest_rate: float,
    replacement_cost: Optional[float] = None,
) -> float:
    """Return the annualized safety cost used by the JCSS LQI example."""

    if initial_cost < 0:
        raise ValueError("initial_cost must be non-negative")
    if not 0 <= failure_probability <= 1:
        raise ValueError("failure_probability must be in [0, 1]")
    if service_life <= 0:
        raise ValueError("service_life must be positive")
    if interest_rate < 0:
        raise ValueError("interest_rate must be non-negative")
    replacement = initial_cost if replacement_cost is None else replacement_cost
    if replacement < 0:
        raise ValueError("replacement_cost must be non-negative")

    if interest_rate == 0:
        adjustment = -1.0
    else:
        adjustment = (1.0 - (1.0 + interest_rate)) / np.log1p(interest_rate)
    return float(
        (initial_cost + replacement * adjustment * failure_probability) / service_life
    )


class DDOObjective:
    """Base interface for DDO objectives."""

    name = "objective"
    objective_column = "objective"

    def evaluate(self, results: pd.DataFrame, design: str) -> pd.DataFrame:
        """Return decision results with objective-specific columns."""

        raise NotImplementedError


@dataclass
class CostBenefitModel(DDOObjective):
    """Cost-benefit objective for a reliability design study."""

    benefit_rate: float
    interest_rate: float
    service_life: float
    construction_cost: Union[float, Callable[[Any], float]]
    failure_cost: Union[float, Callable[[Any], float]]

    def _value(self, item: Union[float, Callable[[Any], float]], design: Any) -> float:
        value = item(design) if callable(item) else item
        return float(value)

    def present_value_factor(self) -> float:
        """Return the present value factor for this model."""

        return present_value_factor(self.interest_rate, self.service_life)

    def construction(self, design: Any) -> float:
        """Return the construction or safety cost for a design value."""

        return self._value(self.construction_cost, design)

    def consequence(self, design: Any) -> float:
        """Return the failure consequence cost for a design value."""

        return self._value(self.failure_cost, design)

    def objective(self, design: Any, failure_probability: float) -> float:
        """Return the net present value objective for a design point."""

        if not 0 <= failure_probability <= 1:
            raise ValueError("failure_probability must be in [0, 1]")
        pvf = self.present_value_factor()
        construction = self.construction(design)
        failure_consequence = self.consequence(design)
        return float(
            self.benefit_rate * pvf
            - construction
            - failure_probability * failure_consequence * pvf
        )

    def annualized_safety_cost(
        self,
        design: Any,
        failure_probability: float,
        replacement_cost: Optional[float] = None,
    ) -> float:
        """Return the JCSS annualized safety cost for a design point."""

        construction = self.construction(design)
        replacement = construction if replacement_cost is None else replacement_cost
        return annualized_safety_cost(
            construction,
            failure_probability,
            self.service_life,
            self.interest_rate,
            replacement,
        )

    def evaluate(self, results: pd.DataFrame, design: str) -> pd.DataFrame:
        """Return results with cost-benefit objective columns."""

        df = results.copy()
        if design not in df:
            raise KeyError(f"Design column {design!r} is missing")
        if "pf" not in df:
            raise KeyError("CostBenefitModel requires a 'pf' column")

        design_values = df[design].to_numpy()
        df[self.objective_column] = [
            self.objective(design_value, pf)
            for design_value, pf in zip(design_values, df["pf"])
        ]
        df["annualized_safety_cost"] = [
            self.annualized_safety_cost(design_value, pf)
            for design_value, pf in zip(design_values, df["pf"])
        ]
        return df


def _default_decision_plot_quantities(data: pd.DataFrame) -> list[str]:
    candidates = [
        "pf",
        "annual_failure_rate",
        "annualized_safety_cost",
        "acceptability_margin",
        "lqi_marginal_term",
        "screening_margin",
        "objective",
    ]
    return [column for column in candidates if column in data]


def _coerce_reference_designs(reference_designs) -> list[dict[str, Any]]:
    if reference_designs is None:
        return []

    if isinstance(reference_designs, Mapping):
        if "design" in reference_designs:
            return [dict(reference_designs)]
        return [
            {"label": str(label), "design": design}
            for label, design in reference_designs.items()
        ]

    if np.isscalar(reference_designs):
        return [{"design": float(reference_designs)}]

    references = []
    for item in reference_designs:
        if isinstance(item, Mapping):
            references.append(dict(item))
        else:
            references.append({"design": float(item)})
    return references


def plot_summary(
    data: Any,
    design: str,
    quantities: Optional[Sequence[str]] = None,
    labels: Optional[Mapping[str, str]] = None,
    yscales: Optional[Mapping[str, str]] = None,
    reference_designs: Optional[Any] = None,
    target_failure_probability: Optional[float] = None,
    target_label: str = "LQI target",
    invert_yaxis: Optional[Iterable[str]] = None,
    panel_labels: bool = False,
    axes: Optional[Sequence[Any]] = None,
    figsize: Optional[tuple[float, float]] = None,
    line_kwargs: Optional[Mapping[str, Any]] = None,
):
    """Plot a one-dimensional design decision optimization summary.

    The helper is intentionally generic: it expects a dataframe-like object
    with a design column and one or more result columns.  It is suitable for
    continuous one-dimensional design sweeps.  More complex discrete
    alternatives or scenario studies can still use the same risk quantities,
    but usually need a problem-specific visualization.

    Parameters
    ----------
    data : dataframe-like
        Table containing the design values and result quantities to plot.
    design : str
        Name of the design-variable column.
    quantities : sequence of str, optional
        Result columns to plot.  When omitted, common decision columns such as
        ``pf``, ``annualized_safety_cost``, ``screening_margin``, and ``objective``
        are used when present.
    labels : mapping, optional
        Axis label overrides keyed by column name.  The design column may also
        be included to set the shared x-axis label.
    yscales : mapping, optional
        Matplotlib y-scale overrides keyed by quantity column name.  Failure
        probability columns use ``"log"`` by default.
    reference_designs : sequence or mapping, optional
        Designs to mark with vertical lines and point markers.  Each item may
        be a scalar design value or a mapping with ``design``, ``label``,
        ``color``, and ``marker`` keys.  A mapping without a ``design`` key is
        interpreted as ``{label: design}``.
    target_failure_probability : float, optional
        Drawn as a horizontal line on failure-probability panels.
    target_label : str, optional
        Legend label for ``target_failure_probability``.
    invert_yaxis : iterable of str, optional
        Quantity columns whose y-axis should be inverted.
    panel_labels : bool, optional
        If ``True``, label panels ``A)``, ``B)``, ...
    axes : sequence, optional
        Existing matplotlib axes.  Its length must match the number of
        quantities.
    figsize : tuple, optional
        Figure size used when ``axes`` is not supplied.
    line_kwargs : mapping, optional
        Keyword arguments passed to the main line plots.

    Returns
    -------
    tuple
        ``(fig, axes)`` from matplotlib.
    """

    df = pd.DataFrame(data).copy()
    if design not in df:
        raise KeyError(f"Design column {design!r} is missing")
    if df.empty:
        raise ValueError("data must contain at least one design row")

    plot_quantities = (
        list(quantities)
        if quantities is not None
        else _default_decision_plot_quantities(df)
    )
    if not plot_quantities:
        raise ValueError("No decision quantities available to plot")

    missing = [column for column in plot_quantities if column not in df]
    if missing:
        raise KeyError(f"Plot quantities are missing: {', '.join(missing)}")

    labels = {} if labels is None else dict(labels)
    yscales = {} if yscales is None else dict(yscales)
    invert = set() if invert_yaxis is None else set(invert_yaxis)
    references = _coerce_reference_designs(reference_designs)

    df = df.sort_values(design)
    x = df[design].astype(float).to_numpy()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError("plot_summary requires matplotlib") from exc

    if axes is None:
        if figsize is None:
            figsize = (7.5, max(2.5, 2.0 * len(plot_quantities)))
        fig, axes = plt.subplots(len(plot_quantities), 1, figsize=figsize, sharex=True)
        axes = np.atleast_1d(axes)
    else:
        axes = np.atleast_1d(axes)
        if len(axes) != len(plot_quantities):
            raise ValueError("axes length must match quantities length")
        fig = axes[0].figure

    main_line = {"color": "#27349a", "marker": "o", "linewidth": 1.8}
    if line_kwargs is not None:
        main_line.update(line_kwargs)

    reference_defaults = [
        {"color": "#2f3aa6", "marker": "o"},
        {"color": "#d62728", "marker": "s"},
        {"color": "#2ca02c", "marker": "^"},
        {"color": "#9467bd", "marker": "D"},
    ]

    for index, (axis, quantity) in enumerate(zip(axes, plot_quantities)):
        y = df[quantity].astype(float).to_numpy()
        axis.plot(x, y, **main_line)
        axis.set_ylabel(labels.get(quantity, quantity))
        axis.grid(True, alpha=0.3)

        yscale = yscales.get(quantity)
        if yscale is None and quantity in {"pf", "annual_failure_rate"}:
            yscale = "log"
        if yscale is not None:
            axis.set_yscale(yscale)
        if quantity in invert:
            axis.invert_yaxis()

        if panel_labels:
            axis.text(
                0.02,
                0.86,
                f"{chr(ord('A') + index)})",
                transform=axis.transAxes,
                fontsize="large",
                fontstyle="italic",
            )

        if quantity in {
            "screening_margin",
            "acceptability_margin",
            "lqi_marginal_term",
        }:
            axis.axhline(0.0, color="0.35", linewidth=0.9)

        if target_failure_probability is not None and quantity in {
            "pf",
            "annual_failure_rate",
        }:
            axis.axhline(
                target_failure_probability,
                color="#d62728",
                linestyle="--",
                linewidth=1.0,
                label=target_label,
            )

        for reference_index, reference in enumerate(references):
            if "design" not in reference:
                raise KeyError("Each reference design mapping requires a 'design' key")
            style = reference_defaults[reference_index % len(reference_defaults)]
            color = reference.get("color", style["color"])
            marker = reference.get("marker", style["marker"])
            label = reference.get("label")
            design_value = float(reference["design"])
            line_label = label if index == 0 and label else None
            axis.axvline(
                design_value,
                color=color,
                linewidth=1.0,
                alpha=0.75,
                label=line_label,
            )
            if x[0] <= design_value <= x[-1]:
                y_value = float(np.interp(design_value, x, y))
                axis.plot(
                    design_value,
                    y_value,
                    marker=marker,
                    markersize=8,
                    markerfacecolor="none",
                    markeredgecolor=color,
                    markeredgewidth=1.6,
                    linestyle="none",
                )

        handles, legend_labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, legend_labels, fontsize="small", loc="best")

    axes[-1].set_xlabel(labels.get(design, design))
    fig.tight_layout()
    return fig, axes


def _coerce_analysis_result(result: Any) -> Mapping[str, float]:
    if isinstance(result, Mapping):
        pf = (
            result.get("pf")
            if "pf" in result
            else result.get("failure_probability", result.get("failure"))
        )
        beta = result.get("beta", result.get("reliability_index"))
    elif hasattr(result, "getFailure") or hasattr(result, "getBeta"):
        pf = result.getFailure() if hasattr(result, "getFailure") else None
        beta = result.getBeta() if hasattr(result, "getBeta") else None
    elif isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        pf = result[0] if len(result) > 0 else None
        beta = result[1] if len(result) > 1 else None
    else:
        pf = result
        beta = None

    if pf is None and beta is None:
        raise ValueError("analysis result must provide pf and/or beta")
    if pf is None:
        pf = float(norm.cdf(-float(beta)))
    if beta is None:
        beta = -float(norm.ppf(float(pf)))
    return {"pf": float(pf), "beta": float(beta)}


@dataclass
class DesignStudy:
    """Evaluate reliability results over a one-dimensional design range."""

    variable: str
    values: Iterable[Any]
    analysis: Callable[[Any], Any]

    def evaluate(self, include_analysis: bool = False) -> pd.DataFrame:
        """Run the analysis callback for each design value."""

        rows = []
        for value in self.values:
            result = self.analysis(value)
            data = dict(_coerce_analysis_result(result))
            data[self.variable] = value
            if include_analysis:
                data["analysis"] = result
            rows.append(data)
        columns = [self.variable, "pf", "beta"]
        if include_analysis:
            columns.append("analysis")
        return pd.DataFrame(rows, columns=columns)


@dataclass
class RiskStudy:
    """Evaluate a risk model over design alternatives."""

    variable: str
    values: Iterable[Any]
    model: Union[ScenarioRiskModel, Callable[[Any], Union[RiskResult, Any]]]

    def _evaluate_model(self, value: Any) -> RiskResult:
        if isinstance(self.model, ScenarioRiskModel):
            return self.model.evaluate(value)

        result = self.model(value)
        if isinstance(result, RiskResult):
            return result
        return RiskResult.from_scenarios(result, metadata={self.variable: value})

    def evaluate(self, swtp: Optional[Union[float, SWTP]] = None) -> pd.DataFrame:
        """Return risk quantities for each design value.

        The annual failure rate is also exposed as a ``pf`` column so that the
        same :class:`DDO` orchestration, objectives, and criteria used with a
        :class:`DesignStudy` accept a :class:`RiskStudy` unchanged.
        """

        rows = []
        for value in self.values:
            risk = self._evaluate_model(value)
            row = risk.to_dict()
            row["pf"] = risk.annual_failure_rate
            row[self.variable] = value
            if swtp is not None:
                row["life_safety_cost"] = risk.life_safety_cost(swtp)
                row["total_risk_cost"] = risk.total_risk_cost(swtp)
            rows.append(row)

        columns = [
            self.variable,
            "annual_failure_rate",
            "pf",
            "beta",
            "expected_fatalities",
            "expected_economic_loss",
        ]
        if swtp is not None:
            columns.extend(["life_safety_cost", "total_risk_cost"])
        extra_columns = [
            column for column in pd.DataFrame(rows).columns if column not in columns
        ]
        return pd.DataFrame(rows, columns=columns + extra_columns)


class DDOCriterion:
    """Base interface for DDO acceptability criteria."""

    name = "criterion"
    feasibility_column: Optional[str] = None

    def evaluate(self, results: pd.DataFrame) -> pd.DataFrame:
        """Return decision results with criterion-specific columns."""

        raise NotImplementedError

    def feasible(self, results: pd.DataFrame) -> pd.Series:
        """Return a boolean feasibility mask for evaluated results."""

        if self.feasibility_column is None:
            raise ValueError(f"{self.__class__.__name__} does not define feasibility")
        if self.feasibility_column not in results:
            raise KeyError(f"Feasibility column {self.feasibility_column!r} is missing")
        return results[self.feasibility_column].astype(bool)


def _lqi_consequence(
    expected_fatalities_given_failure: Optional[float],
    consequence: Optional[FatalityConsequence] = None,
) -> FatalityConsequence:
    if consequence is not None and expected_fatalities_given_failure is not None:
        raise ValueError(
            "Specify either expected_fatalities_given_failure or consequence, not both"
        )
    if consequence is not None:
        return consequence
    if expected_fatalities_given_failure is None:
        raise ValueError(
            "LQI construction requires expected_fatalities_given_failure or consequence"
        )
    return FatalityConsequence(people_exposed=expected_fatalities_given_failure)


def _require_explicit_indexed(indexed: Optional[bool]) -> bool:
    if indexed is None:
        raise ValueError("indexed=True or indexed=False must be supplied explicitly")
    return bool(indexed)


@dataclass(frozen=True)
class LQI(DDOCriterion):
    """Minimum acceptable life-safety criterion using LQI/SWTP.

    The criterion adds consequence valuation and LQI acceptability columns to
    the reliability and objective results produced by :class:`DDO`.  It does
    not select the economic optimum by itself.
    """

    swtp: Optional[SWTP] = None
    consequence: Optional[FatalityConsequence] = None
    target: Optional[TargetReliability] = None

    name = "lqi"
    feasibility_column = "lqi_acceptable"

    @staticmethod
    def _as_swtp(swtp: Union[float, SWTP]) -> SWTP:
        return swtp if isinstance(swtp, SWTP) else SWTP(float(swtp))

    @staticmethod
    def lookup_target(k1: float, variability: str = "medium") -> TargetReliability:
        """Return the rounded source-table LQI target for ``k1``."""

        return lqi_target_reliability(k1, variability=variability)

    @staticmethod
    def derive_target(
        k1: float,
        resistance_cov: float = 0.4,
        load_cov: float = 0.4,
        bounds: tuple[float, float] = (1.0, 20.0),
    ) -> TargetReliability:
        """Calculate an LQI target from the marginal optimization condition."""

        return derive_lqi_target(
            k1,
            resistance_cov=resistance_cov,
            load_cov=load_cov,
            bounds=bounds,
        )

    @classmethod
    def from_swtp(
        cls,
        swtp: Union[float, SWTP],
        *,
        expected_fatalities_given_failure: Optional[float] = None,
        consequence: Optional[FatalityConsequence] = None,
        marginal_safety_cost: float,
        variability: str = "medium",
    ) -> "LQI":
        """Create an LQI criterion from an SWTP value.

        Parameters
        ----------
        swtp : float or SWTP
            Societal willingness to pay per statistical life.
        expected_fatalities_given_failure : float, optional
            Expected fatalities conditional on failure.
        consequence : FatalityConsequence, optional
            Consequence model.  Use this instead of
            ``expected_fatalities_given_failure`` when exposure and fatality
            probability should remain explicit.
        marginal_safety_cost : float
            Marginal annual safety cost used in the LQI target-reliability
            table.
        variability : {"low", "medium", "high"}, optional
            Variability class for the LQI target-reliability table.
        """

        swtp_value = cls._as_swtp(swtp)
        consequence = _lqi_consequence(expected_fatalities_given_failure, consequence)
        target = lqi_target_reliability(
            lqi_k1(
                safety_cost_rate=marginal_safety_cost,
                swtp=swtp_value,
                expected_fatalities_given_failure=(
                    consequence.expected_fatalities_given_failure
                ),
            ),
            variability=variability,
        )
        return cls(swtp=swtp_value, consequence=consequence, target=target)

    @classmethod
    def from_country(
        cls,
        code: str,
        *,
        expected_fatalities_given_failure: Optional[float] = None,
        consequence: Optional[FatalityConsequence] = None,
        marginal_safety_cost: float,
        indexed: Optional[bool] = None,
        variability: str = "medium",
    ) -> "LQI":
        """Create an LQI criterion from a built-in country SWTP value."""

        return cls.from_swtp(
            SWTP.from_country(code, indexed=_require_explicit_indexed(indexed)),
            expected_fatalities_given_failure=expected_fatalities_given_failure,
            consequence=consequence,
            marginal_safety_cost=marginal_safety_cost,
            variability=variability,
        )

    @classmethod
    def from_lqi(
        cls,
        *,
        gross_domestic_product_per_capita: float,
        work_leisure_parameter: float,
        demographic_constant: float,
        expected_fatalities_given_failure: Optional[float] = None,
        consequence: Optional[FatalityConsequence] = None,
        marginal_safety_cost: float,
        variability: str = "medium",
        currency: str = "currency units",
        price_year: Optional[int] = None,
        source: Optional[str] = "LQI relation SWTP = g / q * G",
    ) -> "LQI":
        """Create an LQI criterion from the LQI SWTP relation.

        ``work_leisure_parameter`` is the dimensionless LQI parameter ``q``
        (~0.1--0.2), not an annual mortality rate.
        """

        return cls.from_swtp(
            SWTP.from_lqi(
                gross_domestic_product_per_capita=gross_domestic_product_per_capita,
                work_leisure_parameter=work_leisure_parameter,
                demographic_constant=demographic_constant,
                currency=currency,
                price_year=price_year,
                source=source,
            ),
            expected_fatalities_given_failure=expected_fatalities_given_failure,
            consequence=consequence,
            marginal_safety_cost=marginal_safety_cost,
            variability=variability,
        )

    @property
    def expected_fatalities_given_failure(self) -> Optional[float]:
        """Return expected fatalities conditional on failure when available."""

        if self.consequence is None:
            return None
        return self.consequence.expected_fatalities_given_failure

    @property
    def k1(self) -> Optional[float]:
        """Return the LQI safety cost ratio when a target is available."""

        if self.target is None:
            return None
        return self.target.k1

    def _swtp_and_expected_fatalities(self) -> tuple[SWTP, float]:
        if self.swtp is None:
            raise ValueError("LQI criterion requires an SWTP value")
        if self.consequence is None:
            raise ValueError("LQI criterion requires a consequence model")
        return self.swtp, self.consequence.expected_fatalities_given_failure

    def risk_cost(
        self,
        safety_cost: Union[float, np.ndarray],
        failure_rate: Union[float, np.ndarray],
    ):
        """Return the JCSS LQI life-safety risk-cost term."""

        swtp, expected = self._swtp_and_expected_fatalities()
        return jcss_lqi_risk_cost(safety_cost, failure_rate, swtp, expected)

    def risk_cost_from_result(
        self,
        safety_cost: float,
        risk: RiskResult,
        include_economic_loss: bool = False,
    ) -> float:
        """Return LQI risk cost from an aggregated risk result."""

        swtp, _ = self._swtp_and_expected_fatalities()
        return jcss_lqi_risk_cost_from_result(
            safety_cost,
            risk,
            swtp,
            include_economic_loss=include_economic_loss,
        )

    def acceptability_margin(
        self,
        safety_cost_derivative: float,
        failure_rate_derivative: float,
    ) -> float:
        """Return the JCSS marginal LQI acceptability margin."""

        swtp, expected = self._swtp_and_expected_fatalities()
        return jcss_lqi_acceptability_margin(
            safety_cost_derivative,
            failure_rate_derivative,
            swtp,
            expected,
        )

    def acceptability_margin_at(
        self,
        safety_cost: Callable[[float], float],
        failure_rate: Callable[[float], float],
        design: float,
        step: Optional[float] = None,
    ) -> float:
        """Return the finite-difference LQI acceptability margin at a design.

        This is :meth:`acceptability_margin` evaluated from cost and failure-rate
        callables, differentiating each by central finite differences.
        """

        dcost = finite_difference_derivative(safety_cost, design, step=step)
        drate = finite_difference_derivative(failure_rate, design, step=step)
        return self.acceptability_margin(dcost, drate)

    def acceptability_boundary(
        self,
        safety_cost: Callable[[float], float],
        failure_rate: Callable[[float], float],
        bounds: tuple[float, float],
        step: Optional[float] = None,
    ) -> float:
        """Return the design where the LQI acceptability margin changes sign.

        The marginal acceptability margin ``dC/dp + SWTP * N_F * dh/dp`` is
        negative where society would still pay to reduce risk and non-negative
        once the marginal cost of safety meets or exceeds the SWTP-valued risk
        reduction.  The acceptance boundary is the design at which the margin is
        zero, i.e. the social optimum of the LQI criterion.  Root finding uses
        Brent's method over ``bounds`` and requires the margin to change sign
        across the interval.

        Parameters
        ----------
        safety_cost : callable
            Safety or construction cost as a function of the design value.
        failure_rate : callable
            Annual failure probability or rate as a function of the design.
        bounds : tuple of float
            Increasing ``(lower, upper)`` search interval bracketing the
            boundary.
        step : float, optional
            Finite-difference step for the marginal derivatives.

        Returns
        -------
        float
            Design value at which the marginal LQI margin is zero.
        """

        lower, upper = map(float, bounds)
        if lower >= upper:
            raise ValueError("bounds must be an increasing (lower, upper) pair")

        def margin(design: float) -> float:
            return self.acceptability_margin_at(
                safety_cost, failure_rate, design, step=step
            )

        lower_margin = margin(lower)
        if lower_margin == 0.0:
            return lower
        upper_margin = margin(upper)
        if upper_margin == 0.0:
            return upper
        if (lower_margin > 0.0) == (upper_margin > 0.0):
            raise ValueError(
                "marginal LQI margin does not change sign over bounds; "
                "no acceptability boundary in the given interval"
            )
        return float(brentq(margin, lower, upper))

    def evaluate(self, results: pd.DataFrame) -> pd.DataFrame:
        """Return decision results with LQI/SWTP columns."""

        df = results.copy()

        if self.swtp is not None and self.consequence is not None:
            expected = self.consequence.expected_fatalities_given_failure
            df["expected_fatalities_given_failure"] = expected
            df["swtp_consequence"] = self.swtp.for_lives(expected)

        if self.target is not None:
            df["target_pf"] = self.target.pf
            df["target_beta"] = self.target.beta
            df["lqi_acceptable"] = df["pf"] <= self.target.pf
            # Screening slack: positive when pf is below the target.  Distinct
            # from the marginal-cost acceptability margin (acceptability_margin).
            df["screening_margin"] = self.target.pf - df["pf"]

        return df


@dataclass
class DDO:
    """Evaluate a decision context with an objective and acceptability criterion.

    Construct directly from the three pieces::

        ddo = DDO(
            study=study,
            objective=CostBenefitModel(...),
            criterion=LQI.from_country(...),
        )

    Construction is keyword-only by design: ``study``, ``objective`` and
    ``criterion`` are easy to transpose positionally, which would silently
    misassign them.  :meth:`run` evaluates every alternative and caches the
    table; :meth:`optimize` returns the best feasible alternative and
    :meth:`economic_optimum` the unconstrained economic best.
    """

    study: Union[DesignStudy, "RiskStudy"]
    criterion: DDOCriterion
    objective: Optional[DDOObjective] = None
    results: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    def __init__(
        self,
        *,
        study: Union[DesignStudy, "RiskStudy"],
        criterion: DDOCriterion,
        objective: Optional[DDOObjective] = None,
    ):
        self.study = study
        self.criterion = criterion
        self.objective = objective
        self.results = None

    def _evaluate(self) -> pd.DataFrame:
        df = self.study.evaluate()
        if self.objective is not None:
            df = self.objective.evaluate(df, design=self.study.variable)
        return self.criterion.evaluate(df)

    def run(self) -> pd.DataFrame:
        """Evaluate every alternative and cache the decision table."""

        self.results = self._evaluate()
        return self.results

    def _results_or_run(self) -> pd.DataFrame:
        return self.results if self.results is not None else self.run()

    def _objective_column(self) -> str:
        if self.objective is not None:
            return self.objective.objective_column
        return "objective"

    def economic_optimum(self) -> pd.Series:
        """Return the alternative with the largest objective, ignoring feasibility."""

        df = self._results_or_run()
        objective_column = self._objective_column()
        if objective_column not in df:
            raise ValueError("Objective results are not available")
        return df.loc[df[objective_column].idxmax()]

    def feasible_results(self) -> pd.DataFrame:
        """Return evaluated alternatives satisfying the criterion."""

        df = self._results_or_run()
        return df.loc[self.criterion.feasible(df)]

    def optimize(self) -> pd.Series:
        """Return the feasible alternative with the largest objective value."""

        feasible = self.feasible_results()
        if feasible.empty:
            raise ValueError("No feasible alternatives satisfy the criterion")
        objective_column = self._objective_column()
        if objective_column not in feasible:
            raise ValueError("Objective results are not available")
        return feasible.loc[feasible[objective_column].idxmax()]

    def summary(self) -> pd.DataFrame:
        """Return the key decision points as a small labelled table.

        One row for the unconstrained economic optimum and, when the criterion
        admits one, a second for the best feasible alternative.  Columns are the
        design variable, ``pf``, ``beta`` (when present), the objective, and the
        criterion's feasibility flag.  This is the table a designer reads off a
        study, without rebuilding it row by row.
        """

        df = self._results_or_run()
        objective_column = self._objective_column()
        if objective_column not in df:
            raise ValueError("DDO.summary requires an objective")

        points = [("economic optimum", self.economic_optimum())]
        feasible = self.feasible_results()
        if not feasible.empty:
            points.append(
                ("best feasible", feasible.loc[feasible[objective_column].idxmax()])
            )

        feasibility_column = getattr(self.criterion, "feasibility_column", None)
        candidate_columns = [
            self.study.variable,
            "pf",
            "beta",
            objective_column,
            feasibility_column,
        ]
        columns = [
            column
            for column in candidate_columns
            if column is not None and column in df
        ]
        rows = [
            {"point": label, **{c: row[c] for c in columns}} for label, row in points
        ]
        return pd.DataFrame(rows, columns=["point"] + columns)

    def plot(
        self,
        design: Optional[str] = None,
        quantities: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        """Plot results from the most recent run."""

        df = self._results_or_run()
        design_column = self.study.variable if design is None else design
        return plot_summary(df, design=design_column, quantities=quantities, **kwargs)


# Public surface.  The canonical workflow is object-based: build an ``SWTP``
# and an ``LQI`` criterion, a ``CostBenefitModel`` objective and a
# ``DesignStudy`` (or ``RiskStudy``), combine them in a ``DDO``, and read the
# decision off ``DDO.run``/``optimize``.  The free ``jcss_lqi_*`` /
# ``lqi_target_reliability`` / ``derive_lqi_target`` / ``rackwitz_table``
# functions, the SWTP table helpers, and ``TargetReliabilityCalibration`` remain
# importable from ``pystra.ddo`` as a low-level functional layer, but are kept
# out of ``__all__`` so the object API is the obvious entry point.
__all__ = [
    # Societal value of life
    "SWTP",
    "FatalityConsequence",
    # Acceptability criterion and its result type
    "LQI",
    "TargetReliability",
    # Objective
    "CostBenefitModel",
    # Studies
    "DesignStudy",
    "RiskStudy",
    "ScenarioRiskModel",
    "RiskResult",
    # Orchestration
    "DDO",
    "DDOObjective",
    "DDOCriterion",
    # Code-calibration target model
    "RackwitzTargetModel",
]
