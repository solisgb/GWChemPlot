# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 18:46:14 2023

@author: solis
"""
import json
import pathlib


class Inputs():
    """
    Specific parameters for a run of the package GWChemPlot
    Atributes: They are stores in a dict named input with keys:
	data_path : str. Path to GWChemPlot data file (type csv o xlsx)
	sheet_number : str. If is xlsx sheet number (first 0)
	fo_facies_xlsx : str. Path to store ionic classification (xlsx file)
	fig_Piper : str. Path to png file to store Piper diagram
	fig_Schoeller : str. Path to png file to store Schoeller diagram
	fig_Stiff : str. Path to png file to store Stiff diagrams
    """
    
    def __init__(self, fi: str='GWChemPlot_inputs.json'):
        """
        Attributes are read from the file fi of type json 
        Parameters
        ----------
        fi : path to json file
        """
        self._fi = fi
        with open(fi, 'r') as file:
            self._inputs = json.load(file)


    def file_data_is_csv(self) -> bool:
       suffix = pathlib.Path(self.data_path).suffix 
       if suffix.lower() == '.csv':
           return True
       else:
           return False


    def file_data_is_xlsx(self) -> bool:
       suffix = pathlib.Path(self.data_path).suffix 
       if suffix.lower() == '.xlsx':
           return True
       else:
           return False    

    
    @property 
    def data_path(self):
        return self._inputs["data_path"]
    
    
    @property 
    def sheet_number(self):
        return self._inputs["sheet_number"]
     
    
    @property 
    def fo_facies_xlsx(self):
        return self._inputs["fo_facies_xlsx"]   
     
    
    @property 
    def fig_Piper(self):
        return self._inputs["fig_Piper"]   
     
    
    @property 
    def fig_Schoeller(self):
        return self._inputs["fig_Schoeller"]   
    
    
    @property 
    def fig_Stiff(self):
        return self._inputs["fig_Stiff"]
