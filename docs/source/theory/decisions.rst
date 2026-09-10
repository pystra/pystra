Design decisions and societal risk
**********************************

Design Decision optimization and Societal Risk Acceptance
=========================================================


Design decision optimization with societal risk acceptance is normally applied
after a reliability analysis has estimated :math:`p_f` or :math:`\beta`.  It
does not require a different FORM, SORM, or simulation model.  Instead, an
economic objective is evaluated subject to a societal acceptability criterion.
This follows the risk-based decision framing used in the JCSS risk assessment
guidance [JCSS2008RiskAssessment]_ [KroonMaes2008RiskFramework]_ and the
risk-informed decision principles codified in ISO 2394 [ISO2394]_.
Pystra's initial DDO criterion uses the life quality index (LQI) to define a
minimum acceptable life-safety level with societal willingness to pay (SWTP) as
the life-safety valuation, following the LQI method and the JCSS
risk-assessment background documents and examples [Nathwani1997LQI]_
[Nathwani2009LifeQuality]_ [Rackwitz2002LQI]_ [Rackwitz2008LQI]_
[Streicher2008LQI]_ [Schubert2009LQI]_ [VanCoile2019ALARP]_.

For life-safety problems, the LQI literature expresses the societal willingness
to pay (SWTP) to save one statistical life as a function of the gross domestic
product available for risk reduction, the LQI work--leisure parameter, and a
demographic life-time constant; mortality enters through the demographic
constant rather than through that parameter.  In the notation used by Rackwitz,
this is of the form

.. math::
   :label: eq:lqi_swtp

   \mathrm{SWTP}_x = \frac{g}{q} C_x

where :math:`g` is the income or GDP measure available for risk reduction,
:math:`q` is the dimensionless LQI work--leisure (income-elasticity) parameter
(typically about 0.1--0.2, e.g. 0.175 in Schubert and Faber, 2009; it is *not*
an annual mortality rate), and :math:`C_x` depends on the mortality reduction
scheme, discounting, and the predictive cohort life table
[Rackwitz2004Discounting]_.  In :meth:`~pystra.decision.ddo.SWTP.from_lqi` this parameter
is named ``work_leisure_parameter``.  This SWTP interpretation is developed in the LQI literature
[PandeyNathwani2004LQI]_ [PandeyNathwaniLind2006LQI]_ and used by
Rackwitz for structural reliability optimization and acceptability
[Rackwitz2002LQI]_.  The ``ra.SWTP.from_lqi`` helper
(:meth:`~pystra.decision.ddo.SWTP.from_lqi`) implements the relationship for
user-supplied demographic values.

Pystra also includes a small source-backed country table from Rackwitz's JCSS
background document [Rackwitz2008LQI]_.  The anchor values are the
:math:`G_{\Delta \bar{l}}` column in Table 7, in millions of 1999 PPPUS$:

.. list-table:: Built-in SWTP country values
   :header-rows: 1

   * - Code
     - Country
     - SWTP [10\ :sup:`6` PPPUS$]
   * - CA
     - Canada
     - 1.8
   * - US
     - USA
     - 2.1
   * - AT
     - Austria
     - 1.9
   * - BE
     - Belgium
     - 2.4
   * - CZ
     - Czech Republic
     - 0.54
   * - DK
     - Denmark
     - 1.7
   * - FI
     - Finland
     - 1.3
   * - FR
     - France
     - 1.9
   * - DE
     - Germany
     - 1.9
   * - IT
     - Italy
     - 1.8
   * - NL
     - Netherlands
     - 2.8
   * - NO
     - Norway
     - 1.8
   * - ES
     - Spain
     - 1.3
   * - SE
     - Sweden
     - 1.5
   * - CH
     - Switzerland
     - 1.8
   * - GB
     - United Kingdom
     - 1.7
   * - JP
     - Japan
     - 1.3
   * - NZ
     - New Zealand
     - 1.3

The anchor table is intentionally not overwritten with newer values.  For
current studies, :func:`~pystra.decision.ddo.index_swtp_record` and
``swtp_table(indexed=True)`` return a separately traceable indexed table.
High-level country-based LQI construction requires users to choose
``indexed=True`` or ``indexed=False`` explicitly.  The built-in indexed view
uses the World Bank WDI GDP per capita PPP indicator ``NY.GDP.PCAP.PP.CD`` to
scale each Rackwitz anchor value from 1999 to 2024 [WorldBankWDI]_.  This is a
practical update of the LQI income term :math:`g`; it is not a substitute for a
full recalculation of mortality tables, discounting, and age averaging.

Target reliabilities may be taken from published tables or calculated from an
explicit decision model.  Rackwitz [Rackwitz2000CodeMaking]_ formulates the
code-making problem as an economic optimization in which the annual failure
rate is the natural reliability measure.  Steenbergen, Rózsás, and
Vrouwenvelder [Steenbergen2018Target]_ revisit the same framework and
emphasize annual failure rates as a way to compare targets across design and
remaining working lives.  In normalized form, the Rackwitz/Steenbergen model
implemented by :class:`~pystra.decision.ddo.RackwitzTargetModel` maximizes

.. math::
   :label: eq:rackwitz_target_objective

   Z(p) = B - C(p)
          - U\frac{\lambda}{\gamma} P_{f,\mathrm{SLS}}(p)
          - (C(p)+A)\frac{\omega}{\gamma}
          - (C(p)+H)\frac{\lambda}{\gamma} P_{f,\mathrm{ULS}}(p)

where :math:`p = E[R]/E[S]`, :math:`C(p)=C_0+C_1p`, :math:`\gamma` is the
discount or interest rate, :math:`\omega` is the obsolescence rate, and
:math:`\lambda` is the load occurrence rate.  Pystra uses a closed-form
lognormal resistance-demand model for :math:`P_f(p)` in this calibration.

Every cost is normalized by the base construction cost :math:`C_0`
(``base_cost``), so the model's cost inputs are *ratios* to :math:`C_0`, mapped
to :class:`~pystra.decision.ddo.RackwitzTargetModel` parameters as follows.

.. list-table:: Normalized cost inputs (fractions of :math:`C_0`)
   :header-rows: 1

   * - Symbol
     - Parameter
     - Meaning
   * - :math:`C_1/C_0`
     - ``safety_cost_ratio``
     - marginal safety cost per unit of :math:`p`
   * - :math:`H/C_0`
     - ``failure_cost_ratio``
     - failure (ULS) consequence cost
   * - :math:`U/C_0`
     - ``serviceability_cost_ratio``
     - serviceability (SLS) cost
   * - :math:`A/C_0`
     - ``demolition_cost_ratio``
     - demolition / obsolescence cost
   * - :math:`b/C_0`
     - ``benefit_rate``
     - constant annual benefit (independent of :math:`p`)

The rates :math:`\gamma`, :math:`\omega`, and :math:`\lambda` are
``interest_rate``, ``obsolescence_rate``, and ``load_occurrence_rate``.  Because
the objective is normalized by :math:`C_0`, the resulting target table depends
only on these relative cost and consequence ratios, not on a particular
jurisdiction; the calibrated classes can be compared with the rounded target
reliabilities tabulated in the JCSS Probabilistic Model Code [JCSSPMC2001]_ and
ISO 2394 [ISO2394]_.  A table for other classes is recalculated by passing
``safety_costs`` (the :math:`C_1/C_0` values) and ``failure_costs`` (the
:math:`H/C_0` values) to
:meth:`~pystra.decision.ddo.RackwitzTargetModel.table`.

Fischer, Barnardo, and Faber [Fischer2012LQI]_ provide a convenient LQI route
for turning an SWTP value and expected fatalities given failure into minimum
target reliabilities; the marginal life-saving cost underlying this route is
examined in detail by Fischer et al. [Fischer2013MarginalCost]_.  For medium
variability, define the safety cost ratio

.. math::
   :label: eq:lqi_k1

   K_1 = \frac{C_1(\gamma_S + \omega)}
              {\mathrm{SWTP}\,N_F}

where :math:`C_1(\gamma_S + \omega)` is the marginal safety cost term and
:math:`N_F` is the expected number of fatalities given failure.  The
corresponding target classes are approximated as:

.. list-table:: LQI target reliability classes for medium variability
   :header-rows: 1

   * - Safety cost ratio :math:`K_1`
     - Cost class
     - :math:`\beta`
     - :math:`p_f`
   * - :math:`10^{-3}` to :math:`10^{-2}`
     - Large
     - 3.1
     - :math:`10^{-3}`
   * - :math:`10^{-4}` to :math:`10^{-3}`
     - Medium
     - 3.7
     - :math:`10^{-4}`
   * - :math:`10^{-5}` to :math:`10^{-4}`
     - Small
     - 4.2
     - :math:`10^{-5}`

For normal studies, ``ra.LQI`` (:class:`~pystra.decision.ddo.LQI`) builds this target
directly from a country SWTP value or a user-supplied SWTP value, expected
fatalities given failure or an explicit consequence model, and marginal safety
cost.  ``ra.LQI.lookup_target`` returns the rounded source-table target for a
given :math:`K_1`, while the lower-level :func:`~pystra.decision.ddo.lqi_k1` and
:func:`~pystra.decision.ddo.lqi_target_reliability` helpers remain available in
:mod:`pystra.decision.ddo`.  When the underlying resistance-demand model should be
calculated instead of looked up, ``ra.LQI.derive_target`` solves the marginal
target problem

.. math::
   :label: eq:lqi_marginal_target

   \min_p\; K_1p + P_f(p),
   \qquad\mathrm{or}\qquad
   K_1 = -\frac{dP_f(p)}{dp}.

``ra.DDO`` (:class:`~pystra.decision.ddo.DDO`) then evaluates the selected objective and
criterion without changing the underlying stochastic model.  The best feasible
alternative is obtained with ``DDO.optimize()``; the unconstrained economic
optimum is available separately as ``DDO.economic_optimum()``.

For direct JCSS-style optimization, the canonical life-safety risk-cost term
is

.. math::
   :label: eq:lqi_jcss_risk_cost

   S(p) = C(p) + \mathrm{SWTP}\,N_F\,h(p)

where :math:`C(p)` is the safety or construction cost and :math:`h(p)` is a
failure rate or annual failure probability.  The marginal acceptance condition
is

.. math::
   :label: eq:lqi_jcss_acceptance

   \frac{dC(p)}{dp} \ge
   -\mathrm{SWTP}\,N_F\,\frac{dh(p)}{dp}.

``ra.LQI`` exposes these operations as methods such as
``risk_cost``, ``acceptability_margin_at``, and ``acceptability_boundary``.
The underlying
:func:`~pystra.decision.ddo.jcss_lqi_risk_cost` and
:func:`~pystra.decision.ddo.jcss_lqi_acceptability` functions remain available for
direct reproduction of the JCSS equations.

The current implementation separates an objective from an acceptability
criterion and reserves solver logic for future work.  This keeps LQI in its
proper role as a minimum safety criterion rather than the optimizer itself.
Life-cycle cost and utility models based on stochastic renewal processes are a
natural source for future objective implementations [PandeyWangCheng2015Renewal]_.

**Use this method:** :doc:`/guides/assessment` · :doc:`/notebooks/ex_target_reliability` · :doc:`/api/decisions`

For coordinate conventions, see :doc:`notation`.
