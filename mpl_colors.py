# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 11:03:50 2023

@author: solis
"""
from matplotlib import colorbar
import matplotlib.pyplot as plt
import numpy as np
from time import time
import traceback

import littleLogging as logging


class MPL_colors():

    def __init__(self):
        pass


    def get_colormaps_names(self):
        cmaps = [cmap_id for cmap_id in plt.colormaps()]
        cmaps.sort()
        for cm1 in cmaps:
            col_map = plt.get_cmap(cm1)
            print(f"{cm1}: {col_map.N} colors")


    def display_colormaps(self, cmap:str=''):
        def plot_colorMaps(cmap):
        
            fig, ax = plt.subplots(figsize=(4,0.4))
            col_map = plt.get_cmap(cmap)
            
            colorbar.ColorbarBase(ax, cmap=col_map, orientation = 'horizontal')
            
            plt.title(f"Mapolor {cmap}: {col_map.N} colors")
            plt.show()

        if cmap:
            plot_colorMaps(cmap)
        else:
            for cmap_id in plt.colormaps():
                plot_colorMaps(cmap_id)


    def display_colors_in_colormap(self, cmap_name:str='viridis', ncolors:int=10):
        # Get the colormap
        colormap = plt.get_cmap(cmap_name)
        
        # Create a range of numerical values (between 0 and 1)
        values = np.linspace(0, 1, ncolors)  # You can adjust the number of values as needed
        
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

        mplc = MPL_colors()
        mplc.display_colors_in_colormap() 

    except Exception:
        msg = traceback.format_exc()
        logging.append(msg)
    finally:
        logging.dump()
        xtime = time() - startTime
        print(f'El script tardó {xtime:0.1f} s')

        