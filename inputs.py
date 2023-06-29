# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 18:46:14 2023

@author: solis
"""
import json
import pathlib


class Inputs():
    """
    Parameters for a run of the package GWChemPlot
    Atributes: They are stores in a dict named input with keys:
	data_path : str. Path to GWChemPlot data file (type csv o xlsx)
	sheet_number : str. If is xlsx sheet number (first 0)
    max_cbe : float. Analysis with cbe > max_cbe will not be considered
    dir_output : str. Output directory
    fo_cbe : str. Name of charge balance error file
	fo_ion_dominant_classification : str. Name of ion dominant
        classification file (xlsx file)
	fig_Piper : str. Name of Piper diagram file
	fig_Schoeller : str. Name of Schoeller diagram file
	fig_Stiff : str. Name of Stiff diagram file
    """
    
    def __init__(self, fi: str='GWChemPlot_inputs.json'):
        """
        Attributes are read from the file fi of type json 
        Parameters
        ----------
        fi : path to json file containing parameters
        """
        self._fi = fi
        with open(fi, 'r') as file:
            self._inputs = json.load(file)
        
        # checks
        suffix = pathlib.Path(self.data_path).suffix
        valid_suffixes = ('.csv', '.xlsx')
        if suffix.lower() not in valid_suffixes:
            raise ValueError(f'data file must be of type {valid_suffixes}')
        self._inputs["sheet_number"] = int(self._inputs["sheet_number"])
        self._inputs["max_cbe"] = float(self._inputs["max_cbe"])
        self._inputs["dir_output"] = pathlib.Path(self._inputs["dir_output"])
        if not self._inputs["dir_output"].exists():
            raise ValueError(f'{self._inputs["dir_output"]} not exists') 


    def file_is_of_type(self, file: str, suffix_to_check: str) -> bool:
       suffix = pathlib.Path(file).suffix 
       if suffix.lower() == suffix_to_check:
           return True
       else:
           return False


    def file_data_is_csv(self) -> bool:
       return self.file_is_of_type(self.data_path, '.csv')


    def file_data_is_xlsx(self) -> bool:
       return self.file_is_of_type(self.data_path, '.xlsx')

    
    @property 
    def data_path(self):
        return self._inputs["data_path"]
    
    
    @property 
    def sheet_number(self):
        return self._inputs["sheet_number"]
    
    
    @property 
    def max_cbe(self):
        return self._inputs["max_cbe"]     
    
    
    @property 
    def dir_output(self):
        return self._inputs["dir_output"]


    @property 
    def fo_cbe(self):
        return self.dir_output.joinpath(self._inputs["fo_cbe"])
    
    
    @property 
    def fo_ion_dominant_classif(self):
        return self.dir_output.joinpath(self._inputs\
                                        ["fo_ion_dominant_classification"])
     
    @property 
    def fig_Piper(self):
        return self.dir_output.joinpath(self._inputs["fig_Piper"])
     
    
    @property 
    def fig_Schoeller(self):
        return self.dir_output.joinpath(self._inputs["fig_Schoeller"])
    
    
    @property 
    def fig_Stiff(self):
        return self.dir_output.joinpath(self._inputs["fig_Stiff"])
