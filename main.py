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
    from gw_chem_plot import GWChemPlot
except ImportError as e:
    print( getattr(e, 'message', repr(e)))
    raise SystemExit(0)
    

if __name__ == "__main__":

    startTime = time()

    try:
        data_path = './data/data_template.csv'
        df = pd.read_csv(data_path)
        
        gwch = GWChemPlot(df)
        
        resp = input("Run charge balance check (y/n) ?: ")
        if resp.lower() in ('y', 'yes', '1'):
            fo = './out/cbe_data_template.csv'
            df = gwch.cbe()
            df.to_csv(fo, index=False)
            resp = input("Charge balance error saved, continue (y/n) ?: ")
            if resp.lower() not in ('y', 'yes', '1'):
                raise SystemExit(0)

        fig_Piper = './out/piper_data_template.png'
        fig_Schoeller = './out/schoeller_data_template.png'
        fig_Stiff = './out/stiff_data_template.png'

        gwch.plot_Piper(fig_Piper)
        gwch.plot_Shoeller(fig_Schoeller)
        gwch.plot_Stiff(fig_Stiff)

    except ValueError:
        msg = traceback.format_exc()
        logging.append(msg)
    except Exception:
        msg = traceback.format_exc()
        logging.append(msg)
    finally:
        logging.dump()
        xtime = time() - startTime
        print(f'El script tardó {xtime:0.1f} s')

