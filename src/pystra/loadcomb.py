# -*- coding: utf-8 -*-

from collections import OrderedDict
import warnings

from .model import LimitState
from .model import StochasticModel
from .form import Form
from .correlation import CorrelationMatrix
from .distributions import Constant, Distribution
from .fbc import FBCProcess


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

    The old ``dict_dist_comb`` / ``dict_comb_cases`` constructor interface is
    still accepted for compatibility, but is normalized internally to explicit
    cases and emits a deprecation warning.
    """

    def __init__(
        self,
        lsf=None,
        dict_dist_comb=None,
        list_dist_resist=None,
        list_dist_other=None,
        corr=None,
        list_const=None,
        opt=None,
        dict_comb_cases=None,
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
        dict_dist_comb, list_dist_resist, list_dist_other, list_const : optional
            Deprecated legacy variables interface.
        dict_comb_cases : optional
            Deprecated legacy interface.  These inputs are accepted for
            compatibility and immediately converted to explicit ``cases``.
        """
        self.lsf = lsf
        self.df_corr = corr
        self.options = opt

        if constants is not None and list_const is not None:
            raise Exception("Specify only one of constants or list_const")
        self.constant = self._variables_to_dict(
            constants if constants is not None else list_const
        )
        self.label_const = list(self.constant.keys())

        if cases is not None:
            legacy_args = [
                dict_dist_comb,
                list_dist_resist,
                list_dist_other,
                dict_comb_cases,
            ]
            if any(arg is not None for arg in legacy_args):
                raise Exception("Specify either cases or legacy load-combination inputs")
            self._init_from_cases(cases)
        else:
            self._warn_legacy_inputs()
            self._init_from_legacy(
                dict_dist_comb=dict_dist_comb,
                list_dist_resist=list_dist_resist,
                list_dist_other=list_dist_other,
                dict_comb_cases=dict_comb_cases,
            )

        self._set_common_labels()

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
        :class:`~pystra.fbc.FBCProcess` objects.  The FBC model supplies the
        maximum and companion distributions; Turkstra's rule supplies the
        case structure: each variable action is considered as the leading
        action in turn, while the remaining variable actions are taken as
        companion values.  A companion value may be a point-in-time value or,
        for an FBC process with shorter basic intervals, an intermediate
        maximum over the leading action's interval.

        Parameters
        ----------
        variable : mapping
            Mapping of variable-action names to :class:`FBCProcess` objects.
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
            if not isinstance(process, FBCProcess):
                raise Exception("FBC variable actions must be FBCProcess objects")

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

        lc.distributions_comb = OrderedDict(variable)
        lc.distributions_max = OrderedDict(
            (name, lc.cases[f"{name}_leading"][name]) for name in variable
        )
        lc.distributions_pit = OrderedDict(
            (name, process.point_in_time()) for name, process in variable.items()
        )
        lc.distributions_resistance = cls._variables_to_dict(resistance)
        lc.distributions_other = OrderedDict()
        lc.distributions_other.update(cls._variables_to_dict(permanent))
        lc.distributions_other.update(cls._variables_to_dict(other))
        lc.dict_comb_cases = OrderedDict(
            (f"{name}_leading", [name]) for name in variable
        )
        lc.comb_cases_max = list(lc.dict_comb_cases.values())
        lc._set_common_labels()

        return lc

    @staticmethod
    def _variable_name(obj, allow_process=False):
        valid_types = (Distribution, Constant)
        if allow_process:
            valid_types = valid_types + (FBCProcess,)
        if not isinstance(obj, valid_types):
            if allow_process:
                raise Exception(
                    "Input is not a Distribution, Constant, or FBCProcess object"
                )
            raise Exception("Input is not a Distribution or Constant object")
        return obj.getName() if hasattr(obj, "getName") else obj.name

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
            "dict_dist_comb, list_dist_resist, list_dist_other, list_const, "
            "and dict_comb_cases are deprecated. Use LoadCombination(cases=...) "
            "or LoadCombination.turkstra(...) instead.",
            DeprecationWarning,
            stacklevel=3,
        )

    def _init_from_cases(self, cases):
        self.cases = self._normalise_cases(cases)
        self.dict_dist_comb = self.cases
        self.distributions_comb = OrderedDict()
        self.distributions_max = OrderedDict()
        self.distributions_pit = OrderedDict()
        self.distributions_other = OrderedDict()
        self.distributions_resistance = OrderedDict()
        self.dict_comb_cases = {
            name: list(case.keys()) for name, case in self.cases.items()
        }
        self.comb_cases_max = list(self.dict_comb_cases.values())

    def _init_from_legacy(
        self,
        dict_dist_comb,
        list_dist_resist,
        list_dist_other=None,
        dict_comb_cases=None,
    ):
        if dict_dist_comb is None or list_dist_resist is None:
            raise Exception(
                "Specify cases=... or the legacy dict_dist_comb/list_dist_resist inputs"
            )

        self.distributions_comb = dict_dist_comb
        self.distributions_max = OrderedDict(
            (name, values["max"]) for name, values in dict_dist_comb.items()
        )
        self.distributions_pit = OrderedDict(
            (name, values["pit"]) for name, values in dict_dist_comb.items()
        )
        self.distributions_other = self._variables_to_dict(list_dist_other)
        self.distributions_resistance = self._variables_to_dict(list_dist_resist)
        self.dict_comb_cases = (
            OrderedDict((f"{name}_max", [name]) for name in self.distributions_max)
            if dict_comb_cases is None
            else OrderedDict(dict_comb_cases)
        )
        self.comb_cases_max = list(self.dict_comb_cases.values())
        self._check_input()
        self.dict_dist_comb = self._create_dict_dist_comb()
        self.cases = self.dict_dist_comb

    def _set_common_labels(self):
        self.label_comb_cases = list(self.cases.keys())
        self.num_comb = len(self.label_comb_cases)

        if self.distributions_comb:
            self.label_comb_vrs = list(self.distributions_comb.keys())
        else:
            self.label_comb_vrs = self._case_variable_names()

        self.label_resist = list(self.distributions_resistance.keys())
        self.label_other = list(self.distributions_other.keys())
        if self.label_resist or self.label_other or self.distributions_comb:
            self.label_all = (
                self.label_resist
                + self.label_other
                + self.label_comb_vrs
                + self.label_const
            )
        else:
            self.label_all = self._case_variable_names()
            for name in self.label_const:
                if name not in self.label_all:
                    self.label_all.append(name)

        self.dict_label = {
            "resist": self.label_resist,
            "other": self.label_other,
            "comb_vrs": self.label_comb_vrs,
            "comb_cases": self.label_comb_cases,
            "const": self.label_const,
            "all": self.label_all,
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
        if len(self.distributions_max) != len(self.distributions_pit):
            raise Exception(
                "\nLength of Max variables {} does not match\
                      length of point-in-time variables {}".format(
                    len(self.distributions_max), len(self.distributions_pit)
                )
            )

    def get_label(self, label_type):
        """
        Get labels corresponding to label_type.
        """
        return self.dict_label[label_type]

    def _set_num_comb(self):
        """
        Legacy method retained for compatibility.
        """
        self.num_comb = len(self.cases)
        return self.num_comb

    def get_num_comb(self):
        """
        Get the number of load-combination cases.
        """
        return self.num_comb

    def get_dict_dist_comb(self):
        """
        Get the dictionary of distributions for all load-combination cases.
        """
        return self.dict_dist_comb

    def _create_dict_dist_comb(self):
        """
        Create explicit load-combination cases from legacy max/pit inputs.
        """
        dict_dist = OrderedDict()
        for loadc_name, loadc in self.dict_comb_cases.items():
            dict_loadc = OrderedDict()
            dict_loadc.update(self.distributions_resistance)
            dict_loadc.update(self.distributions_other)
            for key, value in self.distributions_max.items():
                if key in loadc:
                    dict_loadc[key] = value
                else:
                    dict_loadc[key] = self.distributions_pit[key]
            dict_dist[loadc_name] = dict_loadc
        return dict_dist

    def _case_name(self, lcn=None):
        lcn = self.label_comb_cases[0] if lcn is None else lcn
        if lcn not in self.cases:
            raise Exception(f'load-combination case "{lcn}" is not defined')
        return lcn

    def case(self, lcn=None):
        """Return a shallow copy of an explicit load-combination case.

        Parameters
        ----------
        lcn : str, optional
            Case name.  If omitted, the first case is returned.

        Returns
        -------
        OrderedDict
            Mapping of variable name to Pystra variable for the selected case.
        """
        return OrderedDict(self.cases[self._case_name(lcn)])

    def stochastic_model(self, lcn=None, **kwargs):
        """Create a :class:`StochasticModel` for a load-combination case.

        Parameters
        ----------
        lcn : str, optional
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
        variables.update(self.case(lcn))
        for key, value in kwargs.items():
            if key in variables:
                variables[key] = value

        sm = StochasticModel()
        for variable in variables.values():
            sm.addVariable(variable)
        if self.df_corr is not None:
            corr = self._get_corr_for_stochastic_model(sm)
            sm.setCorrelation(CorrelationMatrix(corr))
        return sm

    def _get_corr_for_stochastic_model(self, stochastic_model):
        """
        Get correlation data for stochastic model.
        """
        sequence_rvs = list(stochastic_model.getVariables().keys())
        dfcorr_tmp = self.df_corr.reindex(columns=sequence_rvs, index=sequence_rvs)
        corr = dfcorr_tmp.values
        return corr

    def run_reliability_case(self, lcn=None, **kwargs):
        """Create and run FORM analysis for a load-combination case.

        This is a convenience wrapper around :meth:`stochastic_model` and
        :class:`~pystra.form.Form`.  Users who want full control can call
        :meth:`stochastic_model` and instantiate the reliability method
        directly.

        Parameters
        ----------
        lcn : str, optional
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
        sm = self.stochastic_model(lcn, **kwargs)
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

        set_miss = set(self.label_all) - set(kwargs.keys()) - set(self.constant.keys())
        if len(set_miss) > 0:
            kwargs.update({xx: set_value for xx in set_miss})
        for key in self.constant:
            if key not in kwargs and set_const is None:
                kwargs.update({key: self.constant[key].getValue()})
            elif key not in kwargs and set_const is not None:
                kwargs.update({key: set_const})
        gX = self.lsf(**kwargs)
        return gX
