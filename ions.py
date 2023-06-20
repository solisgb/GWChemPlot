# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 12:35:59 2023

@author: solis
"""
import numpy as np
import pandas as pd

import littleLogging as logging


class Ions():
    
    def __init__(self):
        """
        Weights and charge in main ions in groundwater 
        ions weight from https://webqc.org/
        """
        # anions
        self.a_weight = {'Cl': 35.4530, 'SO4': 96.0626, 'CO3': 60.0089,
                         'HCO3': 61.0168, 'NO3': 62.0049}
        self.a_charge = {'Cl': -1, 'SO4': -2, 'CO3': -2, 'HCO3': -1, 
                         'NO3': -1}
        if sorted(self.a_weight.keys()) != sorted(self.a_charge.keys()):
            msg = 'The keys in Ions.a_weight and Ions.a_charge must be ' +\
                'the same.'
            logging.append(msg)
            raise ValueError(msg)
        
        # cations
        self.c_weight = {'Ca': 40.078, 'Mg': 24.305, 'K': 39.0983,
                         'Na': 22.989769}
        self.c_charge = {'Ca': 2, 'Mg': 2, 'K': 1, 'Na': 1}
        if sorted(self.c_weight.keys()) != sorted(self.c_charge.keys()):
            msg = 'The keys in Ions.c_weight and Ions.c_charge must be ' +\
                'the same.'
            logging.append(msg)
            raise ValueError(msg)
    
    
    def weight_get(self, ion_names: [str]=[]) -> pd.DataFrame:
        # Merge dictionaries
        iw = {**self.a_weight, **self.c_weight}
        if ion_names:
            pd.DataFrame([iw.values()], columns=iw.keys())
            return pd.DataFrame([iw])[ion_names]
        else:
            ion_names = [k for k in iw]
            return pd.DataFrame([iw])[ion_names]

    
    def charge_get(self, ion_names: [str]=[]) -> pd.DataFrame:
        # Merge dictionaries
        ich = {**self.a_charge, **self.c_charge}
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
        a = [c1 for c1 in df.columns if c1 in self.a_charge]
        return a


    def cations_in_df(self, df: pd.DataFrame) -> [str]:
        c = [c1 for c1 in df.columns if c1 in self.c_charge]
        return c


    def ions_in_df(self, df: pd.DataFrame) -> [str]:
        c = self.cations_in_df()
        a = self.anions_in_df()
        return c + a


