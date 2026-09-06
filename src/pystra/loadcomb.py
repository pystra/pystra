# -*- coding: utf-8 -*-

from collections import OrderedDict
import warnings

from .model import LimitState
from .model import StochasticModel
from .form import Form
from .correlation import CorrelationMatrix
from .distributions import Constant, Distribution
from .fbc import FbcProcess


class LoadCombination:
    """Named load-combination reliability cases.

    A load combination is represented as a transparent mapping of named
    reliability cases.  Each case is itself a mapping of limit-state variable
    names to ordinary Pystra :class:`Distribution` or :class:`Constant`
    objects.  This keeps the structural reliability model visible: users can
    inspect the variables in a case, build a
    :class:`~pystra.model.StochasticModel`, or use the convenience FORM runner.

    Preferred explicit-case interface::

        LoadCombination(
            lsf=lsf,
            cases={
                "Q1_leading": {"R": R, "G": G, "Q1": Q1max, "Q2": Q2pit},
                "Q2_leading": {"R": R, "G": G, "Q1": Q1pit, "Q2": Q2max},
            },
        )

    For variable actions represented by Ferry-Borges-Castanheta processes, use
    :meth:`LoadCombination.turkstra` to generate explicit leading-action cases
    using Turkstra's rule.  The FBC process defines the load magnitude
    distributions; Turkstra's rule defines the combination cases.

    The legacy action-based construction route remains transitional in v2,
    using ``action_distributions`` and ``leading_actions``. It is normalized
    internally to explicit cases and emits a deprecation warning. The former
    type-prefixed keyword names are no longer accepted.
    """

    def __init__(
        self,
        lsf=None,
        action_distributions=None,
        resistance=None,
        other_variables=None,
        corr=None,
        legacy_constants=None,
        opt=None,
        leading_actions=None,
        cases=None,
        constants=None,
    ):
        """Initialise a load-combination case set.

        Parameters
        ----------
        lsf : callable, optional
            Limit-state function used by :meth:`run_reliability_case` and
            :meth:`eval_lsf_kwargs`.  It is optional when the object is only
            used to inspect or generate cases.
        cases : mapping, optional
            Preferred interface.  Mapping of case name to variables for that
            case, for example ``{"Q1_leading": {"R": R, "Q1": Q1max}}``.
            Values may be supplied as mappings or sequences of Pystra
            variables.
        constants : mapping or sequence, optional
            Constants included in every stochastic model.
        corr : pandas.DataFrame, optional
            Correlation matrix indexed and columned by random-variable names.
        opt : AnalysisOptions, optional
            Options passed to the FORM analysis convenience runner.
        action_distributions, resistance, other_variables, legacy_constants : optional
            Deprecated legacy variables interface.
        leading_actions : optional
            Deprecated legacy action-based interface. These inputs are immediately
            converted to explicit ``cases``. Their v2 names differ from v1.
        """
        self.lsf = lsf
        self.correlation = corr
        self.options = opt

        if constants is not None and legacy_constants is not None:
            raise Exception("Specify only one of constants or legacy_constants")
        self.constant = self._variables_to_dict(
            constants if constants is not None else legacy_constants
        )
        self.constant_names = list(self.constant.keys())

        if cases is not None:
            legacy_args = [
                action_distributions,
                resistance,
                other_variables,
                leading_actions,
            ]
            if any(arg is not None for arg in legacy_args):
                raise Exception(
                    "Specify either cases or legacy load-combination inputs"
                )
            self._init_from_cases(cases)
        else:
            self._warn_legacy_inputs()
            self._init_from_legacy(
                action_distributions=action_distributions,
                resistance=resistance,
                other_variables=other_variables,
                leading_actions=leading_actions,
            )

        self._set_case_metadata()

    @classmethod
    def turkstra(
        cls,
        variable,
        reference_period,
        lsf=None,
        resistance=None,
        permanent=None,
        other=None,
        constants=None,
        corr=None,
        opt=None,
        companion_duration="leading_interval",
    ):
        """Create leading-action cases using Turkstra's rule.

        The generated object is still a normal :class:`LoadCombination`.
        Calling :meth:`case` reveals the generated distributions for each
        leading-action case.

        This constructor expects variable actions to be represented by
        :class:`~pystra.fbc.FbcProcess` objects.  The FBC model supplies the
        maximum and companion distributions; Turkstra's rule supplies the
        case structure: each variable action is considered as the leading
        action in turn, while the remaining variable actions are taken as
        companion values.  A companion value may be a point-in-time value or,
        for an FBC process with shorter basic intervals, an intermediate
        maximum over the leading action's interval.

        Parameters
        ----------
        variable : mapping
            Mapping of variable-action names to :class:`FbcProcess` objects.
        reference_period : float
            Duration over which the leading action maximum is taken.
        lsf : function, optional
            Limit-state function.
        resistance, permanent, other : mapping or sequence, optional
            Common variables included in every case.
        constants : mapping or sequence, optional
            Constants included in every stochastic model.
        companion_duration : {"leading_interval", "point_in_time"} or float
            Rule used for non-leading variable actions.  The default takes a
            companion maximum over the leading action's basic interval.

        Returns
        -------
        LoadCombination
            Load-combination object containing explicit leading-action cases.
        """
        if reference_period <= 0:
            raise Exception("reference_period must be positive")

        variable = cls._variables_to_dict(variable, allow_process=True)
        for process in variable.values():
            if not isinstance(process, FbcProcess):
                raise Exception("FBC variable actions must be FbcProcess objects")

        common = OrderedDict()
        for group in (resistance, permanent, other):
            common.update(cls._variables_to_dict(group))

        cases = OrderedDict()
        for lead_name, lead_process in variable.items():
            case = OrderedDict(common)
            for name, process in variable.items():
                if name == lead_name:
                    case[name] = process.maximum(duration=reference_period)
                elif companion_duration == "leading_interval":
                    case[name] = process.maximum(duration=lead_process.basic_interval)
                elif companion_duration == "point_in_time":
                    case[name] = process.point_in_time()
                else:
                    case[name] = process.maximum(duration=companion_duration)
            cases[f"{lead_name}_leading"] = case

        lc = cls(
            lsf=lsf,
            cases=cases,
            constants=constants,
            corr=corr,
            opt=opt,
        )

        lc.action_distributions = OrderedDict(variable)
        lc.maximum_distributions = OrderedDict(
            (name, lc.cases[f"{name}_leading"][name]) for name in variable
        )
        lc.point_in_time_distributions = OrderedDict(
            (name, process.point_in_time()) for name, process in variable.items()
        )
        lc.resistance_distributions = cls._variables_to_dict(resistance)
        lc.other_distributions = OrderedDict()
        lc.other_distributions.update(cls._variables_to_dict(permanent))
        lc.other_distributions.update(cls._variables_to_dict(other))
        lc.leading_actions = OrderedDict(
            (f"{name}_leading", [name]) for name in variable
        )
        lc.leading_action_groups = list(lc.leading_actions.values())
        lc._set_case_metadata()

        return lc

    @staticmethod
    def _variable_name(obj, allow_process=False):
        valid_types = (Distribution, Constant)
        if allow_process:
            valid_types = valid_types + (FbcProcess,)
        if not isinstance(obj, valid_types):
            if allow_process:
                raise Exception(
                    "Input is not a Distribution, Constant, or FbcProcess object"
                )
            raise Exception("Input is not a Distribution or Constant object")
        return obj.get_name() if hasattr(obj, "get_name") else obj.name

    @classmethod
    def _variables_to_dict(cls, variables, allow_process=False):
        if variables is None:
            return OrderedDict()

        if isinstance(variables, dict):
            out = OrderedDict()
            for key, value in variables.items():
                name = cls._variable_name(value, allow_process=allow_process)
                if key != name:
                    raise Exception(
                        f'variable key "{key}" does not match object name "{name}"'
                    )
                out[name] = value
            return out

        out = OrderedDict()
        for value in variables:
            out[cls._variable_name(value, allow_process=allow_process)] = value
        return out

    @classmethod
    def _normalise_cases(cls, cases):
        if not cases:
            raise Exception("At least one load-combination case is required")

        out = OrderedDict()
        for case_name, variables in cases.items():
            out[case_name] = cls._variables_to_dict(variables)
        return out

    @staticmethod
    def _warn_legacy_inputs():
        warnings.warn(
            "action_distributions, resistance, other_variables, legacy_constants, "
            "and leading_actions are deprecated. Use LoadCombination(cases=...) "
            "or LoadCombination.turkstra(...) instead.",
            DeprecationWarning,
            stacklevel=3,
        )

    def _init_from_cases(self, cases):
        self.cases = self._normalise_cases(cases)
        self.case_distributions = self.cases
        self.action_distributions = OrderedDict()
        self.maximum_distributions = OrderedDict()
        self.point_in_time_distributions = OrderedDict()
        self.other_distributions = OrderedDict()
        self.resistance_distributions = OrderedDict()
        self.leading_actions = {
            name: list(case.keys()) for name, case in self.cases.items()
        }
        self.leading_action_groups = list(self.leading_actions.values())

    def _init_from_legacy(
        self,
        action_distributions,
        resistance,
        other_variables=None,
        leading_actions=None,
    ):
        if action_distributions is None or resistance is None:
            raise Exception(
                "Specify cases=... or the legacy action_distributions/resistance inputs"
            )

        self.action_distributions = action_distributions
        self.maximum_distributions = OrderedDict(
            (name, values["max"]) for name, values in action_distributions.items()
        )
        self.point_in_time_distributions = OrderedDict(
            (name, values["pit"]) for name, values in action_distributions.items()
        )
        self.other_distributions = self._variables_to_dict(other_variables)
        self.resistance_distributions = self._variables_to_dict(resistance)
        self.leading_actions = (
            OrderedDict((f"{name}_max", [name]) for name in self.maximum_distributions)
            if leading_actions is None
            else OrderedDict(leading_actions)
        )
        self.leading_action_groups = list(self.leading_actions.values())
        self._check_input()
        self.case_distributions = self._build_case_distributions()
        self.cases = self.case_distributions

    def _set_case_metadata(self):
        self.case_names = list(self.cases.keys())
        self.n_cases = len(self.case_names)

        if self.action_distributions:
            self.variable_action_names = list(self.action_distributions.keys())
        else:
            self.variable_action_names = self._case_variable_names()

        self.resistance_names = list(self.resistance_distributions.keys())
        self.other_names = list(self.other_distributions.keys())
        if self.resistance_names or self.other_names or self.action_distributions:
            self.variable_names = (
                self.resistance_names
                + self.other_names
                + self.variable_action_names
                + self.constant_names
            )
        else:
            self.variable_names = self._case_variable_names()
            for name in self.constant_names:
                if name not in self.variable_names:
                    self.variable_names.append(name)

        self.name_groups = {
            "resist": self.resistance_names,
            "other": self.other_names,
            "comb_vrs": self.variable_action_names,
            "comb_cases": self.case_names,
            "const": self.constant_names,
            "all": self.variable_names,
        }

    def _case_variable_names(self):
        names = []
        for case in self.cases.values():
            for name in case:
                if name not in names:
                    names.append(name)
        return names

    def _check_input(self):
        """
        Check consistency of supplied input.
        """
        if len(self.maximum_distributions) != len(self.point_in_time_distributions):
            raise Exception(
                "\nLength of Max variables {} does not match\
                      length of point-in-time variables {}".format(
                    len(self.maximum_distributions),
                    len(self.point_in_time_distributions),
                )
            )

    def get_group_names(self, group):
        """
        Get labels corresponding to group.
        """
        return self.name_groups[group]

    def _set_case_count(self):
        """
        Legacy method retained for compatibility.
        """
        self.n_cases = len(self.cases)
        return self.n_cases

    def get_case_count(self):
        """
        Get the number of load-combination cases.
        """
        return self.n_cases

    def get_case_distributions(self):
        """
        Get the dictionary of distributions for all load-combination cases.
        """
        return self.case_distributions

    def _build_case_distributions(self):
        """
        Create explicit load-combination cases from legacy max/pit inputs.
        """
        case_distributions = OrderedDict()
        for loadc_name, loadc in self.leading_actions.items():
            case_variables = OrderedDict()
            case_variables.update(self.resistance_distributions)
            case_variables.update(self.other_distributions)
            for key, value in self.maximum_distributions.items():
                if key in loadc:
                    case_variables[key] = value
                else:
                    case_variables[key] = self.point_in_time_distributions[key]
            case_distributions[loadc_name] = case_variables
        return case_distributions

    def _resolve_case_name(self, case_name=None):
        case_name = self.case_names[0] if case_name is None else case_name
        if case_name not in self.cases:
            raise Exception(f'load-combination case "{case_name}" is not defined')
        return case_name

    def case(self, case_name=None):
        """Return a shallow copy of an explicit load-combination case.

        Parameters
        ----------
        case_name : str, optional
            Case name.  If omitted, the first case is returned.

        Returns
        -------
        OrderedDict
            Mapping of variable name to Pystra variable for the selected case.
        """
        return OrderedDict(self.cases[self._resolve_case_name(case_name)])

    def stochastic_model(self, case_name=None, **kwargs):
        """Create a :class:`StochasticModel` for a load-combination case.

        Parameters
        ----------
        case_name : str, optional
            Case name.  If omitted, the first case is used.
        **kwargs
            Variable overrides.  This is mainly retained for calibration and
            sensitivity workflows where constants or distributions are varied.

        Returns
        -------
        StochasticModel
            Model containing common constants and the selected case variables.
        """
        variables = OrderedDict()
        variables.update(self.constant)
        variables.update(self.case(case_name))
        for key, value in kwargs.items():
            if key in variables:
                variables[key] = value

        sm = StochasticModel()
        for variable in variables.values():
            sm.add_variable(variable)
        if self.correlation is not None:
            corr = self._get_corr_for_stochastic_model(sm)
            sm.set_correlation(CorrelationMatrix(corr))
        return sm

    def _get_corr_for_stochastic_model(self, stochastic_model):
        """
        Get correlation data for stochastic model.
        """
        sequence_rvs = list(stochastic_model.get_variables().keys())
        ordered_correlation = self.correlation.reindex(
            columns=sequence_rvs, index=sequence_rvs
        )
        corr = ordered_correlation.values
        return corr

    def run_reliability_case(self, case_name=None, **kwargs):
        """Create and run FORM analysis for a load-combination case.

        This is a convenience wrapper around :meth:`stochastic_model` and
        :class:`~pystra.form.Form`.  Users who want full control can call
        :meth:`stochastic_model` and instantiate the reliability method
        directly.

        Parameters
        ----------
        case_name : str, optional
            Case name.  If omitted, the first case is analysed.
        **kwargs
            Variable overrides passed to :meth:`stochastic_model`.

        Returns
        -------
        Form
            Completed FORM analysis object.
        """
        if self.lsf is None:
            raise Exception("LoadCombination requires an lsf to run reliability cases")
        ls = LimitState(self.lsf)
        sm = self.stochastic_model(case_name, **kwargs)
        form = Form(sm, ls) if self.options is None else Form(sm, ls, self.options)
        form.run()
        return form

    def eval_lsf_kwargs(self, set_value=0.0, set_const=None, **kwargs):
        """Evaluate the limit-state function with keyword arguments.

        Missing stochastic variables are assigned ``set_value``.  Missing
        constants are assigned their stored value unless ``set_const`` is
        supplied.

        Parameters
        ----------
        set_value : float, optional
            Value assigned to missing random variables.
        set_const : float, optional
            Value assigned to missing constants.
        **kwargs
            Explicit limit-state function arguments.

        Returns
        -------
        float
            Limit-state function value.
        """
        if self.lsf is None:
            raise Exception("LoadCombination requires an lsf to evaluate the LSF")

        set_miss = (
            set(self.variable_names) - set(kwargs.keys()) - set(self.constant.keys())
        )
        if len(set_miss) > 0:
            kwargs.update({xx: set_value for xx in set_miss})
        for key in self.constant:
            if key not in kwargs and set_const is None:
                kwargs.update({key: self.constant[key].get_value()})
            elif key not in kwargs and set_const is not None:
                kwargs.update({key: set_const})
        gX = self.lsf(**kwargs)
        return gX
