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
        
        gwch = GWChemPlot(df)
        
        resp = input("Check charge balance data (y/n) ?: ")
        if resp.lower() in ('y', 'yes', '1'):
            fo = './out/cbe_data_template.xlsx'
            df = gwch.cbe()
            df.to_csv(fo, index=False, float_format='%.3f')
            resp = input("Charge balance error saved, continue (y/n) ?: ")
            if resp.lower() not in ('y', 'yes', '1'):
                raise SystemExit(0)

        resp = input("Do you want to generate the Piper diagram?: ")
        fo = './out/facies_data_template.xlsx'
        df = gwch.ion_dominant_classification()
        df.to_excel(fo, index=False, float_format='%.3f')

        fig_Piper = './out/piper2_data_template.png'
        fig_Schoeller = './out/schoeller_data_template.png'
        fig_Stiff = './out/stiff_data_template.png'

        gwch.plot_Piper2(fig_Piper)
        gwch.plot_Shoeller(fig_Schoeller)
        gwch.plot_Stiff(fig_Stiff)

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

