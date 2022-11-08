#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 12:50:20 2022

@author: shihab
"""
import pystra as ra

class LoadCombination:
    """Class for running load combination cases.

    Methods:
    --------
    eval_lsf_kwargs -- 
    get_dict_dist_comb -- Get the dictionary of distributions for all load 
                            combinations
    get_label -- 
    get_num_comb -- Get the total number of load combinations
    run_reliability_case --- Run reliability analysis for a given load case
    
    
    Attributes:
    --------
    lsf --- Limit State Function
    distributions_max -- Dictionary of maximum distributions
    distributions_pit -- Dictionary of point-in-time distributions
    distributions_other -- Dictionary of static distributions
    distributions_resistance -- Dictionary of resistance distributions
    dict_dist_comb -- Dictionary of distributions for all load combinations
    dict_label -- 
    label_comb_vrs -- Labels of combination variables
    label_comb_cases -- Labels of combination variables
    label_resist -- Labels of resistance variables
    label_other -- Labels of static variables
    label_all -- Labels of all variables including design multiplier
    
    """
    def __init__(self, lsf, dict_dist_comb,
                 list_dist_other, list_dist_resist, list_const=None,
                 dict_comb_cases=None):
        """
        Initialize class instance.

        Parameters
        ----------
        lsf : Function
            Limit State Function.
        dict_dist_max : Dictionary
            Dictionary of maximum distributions.
        dict_dist_pit : Dictionary
            Dictionary of point-in-time distributions.
        dict_dist_static : Dictionary
            Dictionary of static distributions.
        dict_dist_resist : Dictionary
            Dictionary of resistance distributions.
        comb_cases_max : List, optional
            Nested list containing the identifiers of Max distributions per
            load case. The default is None.

        Returns
        -------
        None.

        """
        self.lsf = lsf
        self.distributions_comb = dict_dist_comb
        self.distributions_max = {xx:dict_dist_comb[xx]['max'] for xx in 
                                  dict_dist_comb}
        self.distributions_pit = {xx:dict_dist_comb[xx]['pit'] for xx in 
                                  dict_dist_comb}
        self.distributions_other = {xx.name:xx for xx in list_dist_other}
        self.distributions_resistance = {xx.name:xx for xx in list_dist_resist}
        self.dict_comb_cases = dict_comb_cases
        self.comb_cases_max = [dict_comb_cases[xx] for xx in dict_comb_cases]
        self._check_input()
        self.num_comb = self._set_num_comb()
        self.dict_dist_comb = self._create_dict_dist_comb()
        self.label_comb_cases = list(dict_comb_cases.keys())
        self.label_comb_vrs = list(dict_dist_comb.keys())
        self.label_resist = list(self.distributions_resistance.keys())
        self.label_other = list(self.distributions_other.keys())
        self.label_all = self.label_resist + self.label_other +\
                          self.label_comb_vrs
        if list_const is not None:
            self.constant = {xx.name:xx for xx in list_const}
            self.label_const = list(self.constant.keys())
            self.label_all = self.label_all + self.label_const
        else:
            self.constant = None
            self.label_const = None
            

        self.dict_label = {"resist":self.label_resist, 
                           "other":self.label_other,
                           "comb_vrs":self.label_comb_vrs,
                           "comb_cases":self.label_comb_cases,
                           "const":self.label_const,
                           "all":self.label_all}
    
    def _check_input(self):
        """
        Check consistency of supplied input.

        Raises
        ------
        Exception
            Raised when Length of Max variables does not match length of 
            point-in-time variables.

        Returns
        -------
        None.

        """
        if len(self.distributions_max) != len(self.distributions_pit):
            raise Exception('\nLength of Max variables {} does not match\
                      length of point-in-time variables {}'.format(
                      len(self.distributions_max),
                      len(self.distributions_pit)))


    def get_label(self, label_type):
        """
        Get Labels corresponding to label_type.

        Parameters
        ----------
        label_type : String
            Label type. Possible values: "resist", "other", "comb_vrs", 
            "comb_cases", "const", and "all".

        Returns
        -------
        label : List
            List of labels corresponding to label_type.

        """
        label = self.dict_label[label_type]
        return label

    def _set_num_comb(self):
        """
        Set the number of load combination cases.

        Returns
        -------
        num_comb : Float
            Number of load combination cases..

        """
        self.comb_cases_max = [[xx] for xx in self.distributions_max.keys()] if\
            self.comb_cases_max is None else self.comb_cases_max
        num_comb = len(self.comb_cases_max)
        return num_comb

    def get_num_comb(self):
        """
        Get the number of load combination cases.

        Returns
        -------
        Float
            Number of load combination cases.

        """
        return self.num_comb
    
    def get_dict_dist_comb(self):
        """
        Get the dictionary of distributions for all load combination cases.

        Returns
        -------
        Dictionary
            Dictionary of distributions for all load combination cases.

        """
        return self.dict_dist_comb
    
    def _create_dict_dist_comb(self):
        """
        Create a dictionary containing distributions for respective load 
        combination cases.

        Returns
        -------
        dict_dist : Dictionary
            Dictionary of distributions for all load combination cases.

        """
        dict_dist = {}
        for loadc_name, loadc in self.dict_comb_cases.items():
            dict_loadc = {}
            for key, value in self.distributions_resistance.items():
                dict_loadc.update({key:value})
            for key, value in self.distributions_other.items():
                dict_loadc.update({key:value})
            for key, value in self.distributions_max.items():
                if key in loadc:
                    dict_loadc.update({key:value})
                else:
                    dict_loadc.update({key:self.distributions_pit[key]})
            dict_dist.update({loadc_name:dict_loadc})
        return dict_dist
    
    def run_reliability_case(self, lcn=None, **kwargs):
        """
        Create and run reliability analysis using input LSF 
        for a given load case, lcn.

        Parameters
        ----------
        lcn : float, optional
            Load case number. The default is 1.
        **kwargs : Keyword arguments
            Specify any distribution overrides for the stochastic model random
            variables or constants as keyword arguments.
            Therefore, if kwargs contains any LSF argument, then kwarg specified
            distribution is used for that argument in the reliability analyses.

        Returns
        -------
        form : Pystra FORM object
            FORM reliability analysis object.

        """
        lcn = self.label_comb_cases[0] if lcn is None else lcn
        ls = ra.LimitState(self.lsf)
        sm = ra.StochasticModel()
        if self.constant is not None:
            for key, value in self.constant.items():
                if key in kwargs:
                    sm.addVariable(kwargs[key])
                else:
                    sm.addVariable(value)
        dict_dist = self.dict_dist_comb[lcn]
        for key, value in dict_dist.items():
            if key in kwargs:
                sm.addVariable(kwargs[key])
            else:
                sm.addVariable(value)
        form = ra.Form(sm,ls)
        form.run()
        
        return form
    
    def eval_lsf_kwargs(self, set_value=0.0, set_const=None, **kwargs):
        """
        Evaluate the LSF based on the supplied Keyword arguments, setting 
        all others to set_value.

        Parameters
        ----------
        set_value : Float, optional
            Set value of random variable LSF arguments other than those 
            supplied as keyword arguments. The default is 0.0.
        set_const : Float, optional
            Set value of constant LSF arguments other than those supplied as
            keyword arguments. The default is None.
        **kwargs : Keyword arguments
            LSF Keyword arguments.

        Returns
        -------
        gX : Float
            Evaluation of the LSF.

        """
        if self.constant is not None:
            set_miss = set(self.label_all) - set(kwargs.keys()) - \
                set(self.constant.keys())
        else:
            set_miss = set(self.label_all) - set(kwargs.keys())
        if len(set_miss) > 0:
            kwargs.update({xx:set_value for xx in set_miss})
        for key, values in self.constant.items():
            if key not in kwargs and set_const is None:
                kwargs.update({key:self.constant[key].getValue()})
            elif key not in kwargs and set_const is not None:
                kwargs.update({key:set_const})
        gX = self.lsf(**kwargs)
        return gX
