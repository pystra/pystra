"""Societal willingness to pay and the source/index records behind it."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import pandas as pd

__all__ = [
    "SWTP_COUNTRY_SOURCE",
    "SWTP_TARGET_SOURCE",
    "SWTP_INDEX_SOURCE",
    "SWTP_INDEX_INDICATOR",
    "SWTPIndexRecord",
    "SWTPRecord",
    "SWTP_COUNTRY_VALUES",
    "SWTP_GDP_PPP_INDEX_2024",
    "SWTP_ALIASES",
    "get_swtp_index_record",
    "get_swtp_record",
    "index_swtp_record",
    "get_swtp",
    "swtp_table",
    "SWTP",
]


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


def _index_table(index_table: Mapping[str, SWTPIndexRecord] | None = None):
    return SWTP_GDP_PPP_INDEX_2024 if index_table is None else index_table


def get_swtp_index_record(
    code: str, index_table: Mapping[str, SWTPIndexRecord] | None = None
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
    indexed: bool | None = None,
    index_table: Mapping[str, SWTPIndexRecord] | None = None,
) -> SWTPRecord:
    """Return a country-level SWTP record.

    Parameters
    ----------
    code : str
        Country code or supported country-name alias.
    indexed : bool
        Must be supplied explicitly.  ``False`` returns the 1999 Rackwitz
        anchor; ``True`` returns the value indexed with the built-in World Bank
        GDP per capita PPP factors.  There is no default so the anchor is never
        returned silently.
    index_table : mapping, optional
        Alternate country-code mapping of :class:`SWTPIndexRecord` objects.

    Returns
    -------
    SWTPRecord
        Source-backed SWTP value.
    """

    if _require_explicit_indexed(indexed):
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
    code: str, index_table: Mapping[str, SWTPIndexRecord] | None = None
) -> SWTPRecord:
    """Return a Rackwitz SWTP record indexed to a newer target year."""

    base = get_swtp_record(code, indexed=False)
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
    indexed: bool | None = None,
    index_table: Mapping[str, SWTPIndexRecord] | None = None,
) -> float:
    """Return the SWTP value per statistical life for a country code.

    ``indexed`` must be supplied explicitly (``False`` for the 1999 anchor,
    ``True`` for the indexed value); the anchor is never returned silently.
    """

    return get_swtp_record(
        code, indexed=indexed, index_table=index_table
    ).value_per_life


def swtp_table(
    indexed: bool | None = None,
    index_table: Mapping[str, SWTPIndexRecord] | None = None,
) -> pd.DataFrame:
    """Return the built-in country SWTP table as a dataframe.

    ``indexed`` must be supplied explicitly (``False`` for the 1999 Rackwitz
    anchor table, ``True`` for the indexed current-PPP view); the anchor table
    is never returned silently.
    """

    if not _require_explicit_indexed(indexed):
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
    price_year: int | None = None
    source: str | None = None

    def __post_init__(self):
        if self.value_per_life <= 0:
            raise ValueError("SWTP value_per_life must be positive")

    @classmethod
    def from_country(cls, code: str, indexed: bool | None = None) -> "SWTP":
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
        price_year: int | None = None,
        source: str | None = "LQI relation SWTP = g / q * G",
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


def _swtp_value(swtp: float | SWTP) -> float:
    return swtp.value_per_life if isinstance(swtp, SWTP) else float(swtp)


def _require_explicit_indexed(indexed: bool | None) -> bool:
    if indexed is None:
        raise ValueError("indexed=True or indexed=False must be supplied explicitly")
    return bool(indexed)
