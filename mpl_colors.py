# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 11:03:50 2023

@author: solis
"""
from matplotlib import colorbar
import matplotlib.pyplot as plt
from math import fmod
import numpy as np
from time import time
from typing import Union, List
import traceback

import littleLogging as logging

class MplColors():
    
    def __init__(self):
        pass

    @staticmethod
    def get_colormaps_names(ordered:bool=True) -> None:
        cmaps_names = [cmn1 for cmn1 in plt.colormaps()]
        if ordered:
            cmaps_names.sort()
        return cmaps_names
    
    
    @staticmethod
    def print_colormaps_names(cmap_line:int=5) -> None:
        cmaps_names = [f'{cmn1}: {plt.get_cmap(cmn1).N} colors' \
                       for cmn1 in plt.colormaps()]
        cmaps_names.sort()
        n = len(cmaps_names)
        for i in range(0, n, cmap_line):
            print(", ".join(cmaps_names[i:i+cmap_line]))
    
    
    @staticmethod
    def display_colormaps(cmap_names: Union[str, List[str]]=[], 
                          ndisplays:int=5) -> None:
        def plot_colormap(cmap_name:str) -> None:
            fig, ax = plt.subplots(figsize=(4,0.4))
            col_map = plt.get_cmap(cmap_name)
            colorbar.ColorbarBase(ax, cmap=col_map, orientation = 'horizontal')
            plt.title(f"Mapcolor {cmap_name}: {col_map.N} colors")
            plt.show()
    
        if cmap_names:
            if isinstance(cmap_names, str):
                cmap_names = [cmap_names]
    
        cmaps_names_set = [cmn1 for cmn1 in plt.colormaps()]
        cmaps_names_set.sort()
        ncmaps = len(cmaps_names_set)
    
        if cmap_names:
            valid_cmap_names = [cmn1 for cmn1 in cmap_names \
                                if cmn1 in cmaps_names_set]
            for cmn1 in valid_cmap_names:
                plot_colormap(cmn1)
        else:
            for i, cmn1 in enumerate(cmaps_names_set):
                if i > 0:
                    if fmod(i, ndisplays) == 0:
                        nremain = i + 1 - ncmaps
                        ans = input(f'\n{nremain} colormaps to be displayed, ' 
                                    'continue (y/n):? ')
                        if ans.lower() != 'y':
                            break
                plot_colormap(cmn1)
    
    
    @staticmethod
    def display_colors_in_colormap(cmap_name:str='hsv', ncolors:int=10) ->None:
        cmaps_names_set = [cmn1 for cmn1 in plt.colormaps()]
        if cmap_name not in cmaps_names_set:
            logging.append(f'{cmap_name} is not a valid colormap name')
        
        colormap = plt.get_cmap(cmap_name)
        
        # Create a range of numerical values (between 0 and 1)
        values = np.linspace(0, 1, ncolors)  
        
        # Map the numerical values to colors in the colormap
        colors = [colormap(value) for value in values]
        
        # Print the RGB color codes
        for color in colors:
            print(f"RGB Color Code: {color[:3]}")
        
        # Plot a bar chart using the colors
        plt.bar(range(len(colors)), [1] * len(colors), color=colors)
        plt.show()


if __name__ == "__main__":

    startTime = time()

    try:

        mplc = MplColors()
        mplc.display_colors_in_colormap(ncolors=3) 

    except Exception:
        msg = traceback.format_exc()
        logging.append(msg)
    finally:
        logging.dump()
        xtime = time() - startTime
        print(f'El script tardó {xtime:0.1f} s')

        