# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:55:25 2023

@author: solis

fork of WQChartPy https://github.com/jyangfsu/WQChartPy
"""

try:
    import numpy as np
    import pandas as pd
    from time import time
    import traceback
    
    from inputs import Inputs
    import littleLogging as myLogging
    from gw_chem_plot import GWChemPlot
except ImportError as e:
    print( getattr(e, 'message', repr(e)))
    raise SystemExit(0)
    

if __name__ == "__main__":

    startTime = time()

    try:
        
        ipt = Inputs()
        
        if ipt.file_data_is_csv():
            df = pd.read_csv(ipt.data_path)
        else:
            df = pd.read_excel(ipt.data_path, sheet_name=ipt.sheet_number)
        
        gwp = GWChemPlot(df)
        
        resp = input("Check charge balance error (y/n) ?: ")
        if resp.lower() in ('y', 'yes', '1'):
            er_cb = gwp.cbe()
            er_cb.to_excel(ipt.fo_cbe, index=False, float_format='%.3f')
            resp = input("Charge balance error saved, continue (y/n) ?: ")
            if resp.lower() not in ('y', 'yes', '1'):
                raise SystemExit(0)
            mask = (np.abs(er_cb['cbe']) <= ipt.max_cbe)
            gwp.data = gwp.data[mask]

        resp = input("Generate ion dominant classif. file (y/n) ?: ")
        if resp.lower() in ('y', 'yes', '1'):
            df = gwp.ion_dominant_classification()
            df.to_excel(ipt.fo_ion_dominant_classif, index=False,
                        float_format='%.3f')

        resp = input("Generate Piper diagram (y/n) ?: ")
        if resp.lower() in ('y', 'yes', '1'):
            gwp.plot_Piper2(ipt.fig_Piper)
            
        resp = input("Generate Shoeller figure (y/n) ?: ")
        if resp.lower() in ('y', 'yes', '1'):
            gwp.plot_Shoeller(ipt.fig_Schoeller)
            
        resp = input("Generate Stiff diagram (y/n) ?: ")
        if resp.lower() in ('y', 'yes', '1'):
            gwp.plot_Stiff(ipt.fig_Stiff)            

    except ValueError:
        msg = traceback.format_exc()
        myLogging.append(msg)
    except Exception:
        msg = traceback.format_exc()
        myLogging.append(msg)
    finally:
        myLogging.dump()
        xtime = time() - startTime
        print(f'El script tardó {xtime:0.1f} s')

