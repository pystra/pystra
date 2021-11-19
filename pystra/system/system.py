#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np

from ..model import *


class Component():
    """
    A component within a system described by a limit state
    """
    def __init__(self,name,limit_state):
        """
        Parameters
        ----------
        name : string
            description of component
        limit_state : LimitState class object
            Information about the limit state
        """
        self.name = name
        self.limit_state = limit_state
        self.stochastic_model = StochasticModel()  
        # Also instance a stochastic model
        self.options = AnalysisOptions()  
        # For now, sing default options

    def addVariable(self,obj):
        """
        Add variable for stochastic model of component
        """
        self.stochastic_model.addVariable(obj)
        

class System():
    """
    Abstract base class for a collection of components 
    (either in series/paralel/both)
    """

    
    def __init__(self,obj_list):
        
        self.components = []# list of components objs
        self.event_vector = None
        # used to describe which components events occurs
    
        # TODO: stochasticmodel?
        self.add_components(obj_list)
        # populate component list
         
    def add_components(self,obj_list):
        """Append component objects to system list."""
        
        # TODO: Check if object is already in list
        
        for obj in obj_list: # loop through the objects
            if isinstance(obj,System): # if the object is a system
                self.components = self.components + obj.components
                # append the components of that system
            else: # if the object is a component
                self.components.append(obj)
                # append the component
        
    
class SeriesSystem(System):
    """
    A collection of components (or systems) in series system
    """
    def __init__(self,obj_list):
        """
        Parameters
        ----------
        obj_list : list
            List of components (or system) arranged in a series system

        """
        super().__init__(obj_list)
        

    
class ParallelSystem(System):
    """
    A collection of components (or systems) in parallel system
    """
    def __init__(self,obj_list):
        """
        Parameters
        ----------
        obj_list : list
            List of components (or system) arranged in a series system

        """
        super().__init__(obj_list)