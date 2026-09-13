Code calibration using normalized reliability
=============================================

Code calibration seeks factors that give suitable reliability across a
representative range of designs. The normalized reliability approach expresses
resistance and load contributions on dimensionless scales, so the study can
vary load ratios without selecting a particular structural size for each case.

The workflow is to specify probability models and characteristic values,
choose candidate factors, generate designs satisfying the code equation,
evaluate their reliability and compare it with the target. Factor derivation
from representative design values supplies candidate factors within this
workflow; it does not replace the final reliability verification.

Normalized design and reliability equations
-------------------------------------------

Let :math:`a_g` describe the self-weight share of permanent load and
:math:`a_q` the variable-load share of total demand. These ratios describe the
representative normalized load contributions, while :math:`R,G,P,Q` describe the
normalized random variables. For resistance scale :math:`z`, the design rule is

.. math::

   \phi z r_k = (1-a_q)\left[a_g\gamma_g g_k
       +(1-a_g)\gamma_p p_k\right]+a_q\gamma_q q_k.

The corresponding reliability limit state is

.. math::

   g = \omega_R zR - \omega_S\left[(1-a_q)
       \left(a_gG+(1-a_g)P\right)+a_qQ\right].

The characteristic values :math:`r_k,g_k,p_k,q_k` and probability models are
inputs separate from the candidate factors. Model-error factors
:math:`\omega_R,\omega_S` have explicitly specified distributions. Normalized
variables need not all have the same bias or coefficient of variation.

``NormalizedReliabilityModel`` describes these inputs. ``CodeCalibration``
solves the code equation for each load-ratio pair and calls FORM on the
resulting design. ``CodeCalibrationResult`` preserves each design and its
reliability result, including nonconvergence and target margins.

Interpreting a calibration study
--------------------------------

A candidate set can meet the target for some ratios and fall below it for
others. The reliability envelope shows that variation; choosing a final
factor set also requires a stated selection or optimization criterion and
representative design cases. The current normalized study evaluates supplied
factor sets; it does not infer those policy choices or optimize the factors.

Reference periods and load-process assumptions belong in the input models.
Changing a normalized load ratio does not by itself convert an annual load
distribution into a service-life maximum.

Deriving and verifying partial factors
--------------------------------------

For explicitly defined representative load cases, ``solve_designs`` obtains
designs at a supplied target reliability. ``derive_factors`` uses those design
values to produce candidate resistance, load and combination factors.
``select_factors`` records how the candidates are combined. Finally,
``design_with_factors`` and ``verify_designs`` check the designs obtained with
the selected common factors across all cases.

The coefficient method uses design-value/characteristic-value ratios. The
matrix method follows the leading-action effect convention of the cited
load-combination work. Both require the documented separable resistance-scale
design rule; their algebra is not a general decomposition of arbitrary
nonlinear limit states.

Worked examples and sources
---------------------------

* :doc:`/notebooks/ex_generic_calibration` demonstrates normalized reliability
  and candidate-factor comparisons.
* :doc:`/notebooks/ex_factor_calibration` demonstrates partial-factor derivation
  and verification, with the Sørensen and Caprani–Khan examples and references.
* :doc:`/notebooks/ex_load_combinations` explains the load-process assumptions.
* The `European Commission/JRC reliability background report
  <https://eurocodes.jrc.ec.europa.eu/sites/default/files/2024-11/JRC_Reliability_report_final_23Oct2024_with-ids_corrected.pdf>`_
  describes design-value and code-optimization approaches in Eurocode calibration.

**Use this method:** :doc:`/guides/calibration` · :doc:`/notebooks/ex_generic_calibration` · :doc:`/api/calibration`

For coordinate conventions, see :doc:`notation`.
