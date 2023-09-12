# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:55:25 2023

@author: solis

fork of WQChartPy https://github.com/jyangfsu/WQChartPy
"""

try:
    import pandas as pd
    from time import time
    import traceback
    
    import littleLogging as logging
    from gw_chem_plot import GWChemPlot as gwplot
    
except ImportError as e:
    print( getattr(e, 'message', repr(e)))
    raise SystemExit(0)
    

if __name__ == "__main__":

    startTime = time()

    try:

        fdata = './data/data_template.xlsx'        
        data = pd.read_excel(fdata, sheet_name=2)
        cols_for_label = ['Toma']
        gwplot.set_labels(data, autonumbering=False, 
                              cols_for_label=cols_for_label)
        gwplot.color_labels_set_manual(data, ['red', 'green'])

    except Exception:
        msg = traceback.format_exc()
        logging.append(msg)
    finally:
        logging.dump()
        xtime = time() - startTime
        print(f'El script tardó {xtime:0.1f} s')

