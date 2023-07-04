# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 12:35:59 2023

@author: solis
"""
import numpy as np
import pandas as pd

import littleLogging as myLogging


class Ions():
  
    
    def __init__(self):
        """
        Weights and charge of major ions in groundwater 
        Ions weight from https://webqc.org/
        This class is for use from GWChemPlot only and the considered ions
            are related with the internals of GWChemPlot. If you add new ions
            consider the possible side effects in the class GWChemPlot first.
        """
        # anions
        self._a_weight = {'Cl': 35.453, 'SO4': 96.0626, 'CO3': 60.0089,
                          'HCO3': 61.0168, 'NO3': 62.0049}
        self._a_charge = {'Cl': -1, 'SO4': -2, 'CO3': -2, 'HCO3': -1, 
                          'NO3': -1}
        if sorted(self._a_weight.keys()) != sorted(self._a_charge.keys()):
            msg = 'The keys in Ions._a_weight and Ions._a_charge must be ' +\
                'the same.'
            myLogging.append(msg)
            raise ValueError(msg)
        
        # cations
        self._c_weight = {'Ca': 40.078, 'Mg': 24.305, 'K': 39.0983,
                          'Na': 22.989769}
        self._c_charge = {'Ca': 2, 'Mg': 2, 'K': 1, 'Na': 1}
        if sorted(self._c_weight.keys()) != sorted(self._c_charge.keys()):
            msg = 'The keys in Ions.c_weight and Ions.c_charge must be ' +\
                'the same.'
            myLogging.append(msg)
            raise ValueError(msg)


    def anions_names_get(self) -> [str]:
        return list(self._a_weight.keys())
    
    
    def cations_names_get(self) -> [str]:
        return list(self._c_weight.keys())


    def ions_names_get(self) -> [str]:
        return self.anions_names_get() + self.cations_names_get()
    
    
    def weight_get(self, ion_names: [str]=[]) -> pd.DataFrame:
        iw = {**self._a_weight, **self._c_weight}  # Merge dictionaries
        if ion_names:
            pd.DataFrame([iw.values()], columns=iw.keys())
            return pd.DataFrame([iw])[ion_names]
        else:
            ion_names = [k for k in iw]
            return pd.DataFrame([iw])[ion_names]

    
    def charge_get(self, ion_names: [str]=[]) -> pd.DataFrame:
        ich = {**self._a_charge, **self._c_charge}  # Merge dictionaries
        if ion_names:
            return pd.DataFrame([ich])[ion_names]
        else:
            ion_names = [k for k in ich]
            return pd.DataFrame([ich])[ion_names]

    
    def charge_weight_ratio_get(self, ion_names: [str]=[]) -> pd.DataFrame:
        iw = self.weight_get(ion_names)
        ich = self.charge_get(ion_names)
        return np.abs(ich.div(iw.iloc[0]))


    def anions_in_df(self, df: pd.DataFrame) -> [str]:
        cols = [c1 for c1 in df.columns if c1 in self._a_charge]
        return cols


    def cations_in_df(self, df: pd.DataFrame) -> [str]:
        cols = [c1 for c1 in df.columns if c1 in self._c_charge]
        return cols


    def ions_in_df(self, df: pd.DataFrame) -> [str]:
        c = self.cations_in_df()
        a = self.anions_in_df()
        return c + a


