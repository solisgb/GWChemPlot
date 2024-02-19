# -*- coding: utf-8 -*-
"""
Created on 2024-02-17

@author: solisgb
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ions = {'Cl': (35.453, -1), 'SO4': (96.0626, -2),
        'CO3': (60.0089, -2), 'HCO3': (61.0168, -1), 'NO3': (62.0049, -1), 
        'Ca': (40.078, 2), 'Mg': (24.305, 2), 'K': (39.0983, 1), 'Na': (22.989769, 1)}


def cbe(df:pd.DataFrame):
    """
    Calcula el CBE (Charge Balance Error)

    df tiene las columnas de los mayoritarios en mg/L. Se crean las siguientes columnas 
    * Para cada ion mayoritario, por ej Ca, (en el diccionario ions) se crea la columna rCa con la
      la concentración expresada en meq/L
    * Se calculan las columnas sum_anions y sum_cations
    * Se calcula el CBE
    """
    col_names = df.columns.tolist()

    for k, v in ions.items():
        if k not in col_names:
            print(f'{k} is not present')
            continue
        new_column = 'r'+ k
        df[new_column] = df[k] * np.abs(v[1]) / v[0]

    df['sum_anions'] = 0
    df['sum_cations'] = 0
    for k, v in ions.items():
        if k not in col_names:
            continue        
        col_name = 'r'+ k    
        if v[1] < 0:
            df['sum_anions'] += df[col_name]
        else:
            df['sum_cations'] += df[col_name]
    df['cbe'] = 100 * (df['sum_cations'] - np.abs(df['sum_anions'])) / (df['sum_cations'] + np.abs(df['sum_anions']))


def cation_dominant(row: pd.core.series.Series) -> str:
    """
    Groundwater cation dominant classification
    Custodio E (1983). Hidrogeoquímica. 
    In Hidrología subterránea pp 1001-1095. Ed. Omga
    """

    if row['rNa'] + row['rK'] > row['rMg'] > row['rCa']:
        if row['rNa'] + row['rK'] >= 0.5:
            return 'Na (Mg-Ca)'
        else:
            return 'Na-Mg-Ca'
    elif row['rNa'] + row['rK'] > row['rCa'] > row['rMg']:
        if row['rNa'] >= 0.5:
            return'Na (Mg-Ca)'
        else:
            return'Na-Ca-Mg)'
    elif row['rMg'] > row['rNa'] + row['rK'] > row['rCa']:
        if row['rMg'] >= 0.5:
            return'Mg (Na-Ca)'
        else:
            return'Mg-Na-Ca'               
    elif row['rMg'] > row['rCa'] > row['rNa'] + row['rK']:
        if row['rMg'] >= 0.5:
            return'Mg (Ca-Na)'
        else:
            return'Mg-Ca-Na'                
    elif row['rCa'] > row['rNa'] + row['rK'] > row['rMg']:
        if row['rCa'] >= 0.5:
            return'Ca (Na-Mg)'
        else:
            return'Ca-Na-Mg'                
    elif row['rCa'] > row['rMg'] > row['rNa'] + row['rK']:
        if row['rCa'] >= 0.5:
            return'Ca (Mg-Na)'
        else:
            return'Ca-Mg-Na)'                


def anion_dominant(row: pd.core.series.Series, NO3=True) -> str:
    """
    Groundwater anion dominant classification
    Custodio E (1983). Hidrogeoquímica. 
    In Hidrología subterránea pp 1001-1095. Ed. 
    
    row: Must have the anion keys in dictionary ions with a 'r' as prefix indicating
        concentrations in meq/L. rNO3 and rCO3 couldn't be present. 
    NO3: 
        *If NO3 is True
            and rNO3 is in the columns of row, rNO3 is added to rSO4
        *Otherwise, rNO3 is not considered

    """
    if NO3:
        try: 
            rno3 = row.iloc[0]['rNO3']
        except:
            rno3 = 0.
    else:
         rno3 = 0.

    try:
        rco3 = row.iloc[0]['rCO3']
    except:
        rco3 = 0.

    if row['rCl'] + row['rK'] > row['rSO4'] + rno3 > row['rHCO3'] + rco3:
        if row['rCl'] >= 0.5:
            return 'Cl (SO4-HCO3)'
        else:
            return 'Cl-SO4-HCO3'               
    elif row['rCl'] + row['rK'] > row['rHCO3'] + rco3 > row['rSO4'] + rno3:
        if row['rCl'] + row['rK'] >= 0.5:
            return 'Cl (HCO3-SO4)'
        else:
            return 'Cl-HCO3-SO4'             
    elif row['rSO4'] + rno3 > row['rCl'] + row['rK'] > row['rHCO3'] + rco3 :
        if row['rSO4'] + rno3 >= 0.5:
            return ('SO4 (Cl-HCO3)')
        else:
            return ('SO4-Cl-HCO3')               
    elif row['rSO4'] + rno3 > row['rHCO3'] + rco3 > row['rCl'] + row['rK']:
        if row['rSO4'] >= 0.5:
            return ('SO4 (HCO3-Cl)')
        else:
            return ('SO4-HCO3-Cl')
    elif row['rHCO3'] + rco3 > row['rCl'] + row['rK'] > row['rSO4'] + rno3:
        if row['rHCO3'] + rco3 >= 0.5:
            return ('HCO3 (Cl-SO4)')
        else:
            return ('HCO3-Cl-SO4)')            
    elif row['rHCO3'] + rco3 > row['rSO4'] + rno3 > row['rCl'] + row['rK']:
        if row['rHCO3'] + rco3 >= 0.5:
            return ('HCO3 (SO4-Cl)')
        else:
            return ('HCO3-(SO4-Cl)')            


def triangle_piper(df, file_path, unit='mg/L', dpi=300) -> None:
    """
    Plots the Piper diagram.
    https://github.com/jyangfsu/WQChartPy/tree/main/wqchartpy
    
    Parameters
    ----------
    df : pandas.DataFrame. Geochemical data.
    file_path : str or Path. File path where figure wil saved.
    unit : str. Units used in df (mg/L or meq/L). 
    """
    # Basic data check 
    # -------------------------------------------------------------------------
    # Determine if the required geochemical parameters are defined. 
    if not {'Ca', 'Mg', 'Na', 'K', 
            'HCO3', 'CO3', 'Cl', 'SO4'}.issubset(df.columns):
        raise ValueError("""
        Piper diagram requires:
        Ca, Mg, Na, K, HCO3, CO3, Cl, and SO4""")
        
    # Determine if the provided unit is allowed
    ALLOWED_UNITS = ['mg/L', 'meq/L']
    if unit not in ALLOWED_UNITS:
        raise ValueError("Unit must be mg/L or meq/L")

    ions_WEIGHT = {key: value[0] for key, value in ions.items()}
    ions_CHARGE = {key: value[1] for key, value in ions.items()}

    # Global plot settings
    # -------------------------------------------------------------------------
    # Change default settings for figures
    plt.style.use('default')
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 9
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.titlesize'] = 10   
    
    # Plot background settings
    # -------------------------------------------------------------------------
    # Define the offset between the diamond and traingle
    offset = 0.10                         
    offsety = offset * np.tan(np.pi / 3.0)
    h = 0.5 * np.tan(np.pi / 3.0)
    
    # Calculate the traingles' location 
    ltriangle_x = np.array([0, 0.5, 1, 0])
    ltriangle_y = np.array([0, h, 0, 0])
    rtriangle_x = ltriangle_x + 2 * offset + 1
    rtriangle_y = ltriangle_y
    
    # Calculate the diamond's location 
    diamond_x = np.array([0.5, 1, 1.5, 1, 0.5]) + offset
    diamond_y = h * (np.array([1, 2, 1, 0, 1])) + (offset * np.tan(np.pi / 3))
    
    # Plot the traingles and diamond
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111, aspect='equal', frameon=False, 
                         xticks=[], yticks=[])
    ax.plot(ltriangle_x, ltriangle_y, '-k', lw=1.0)
    ax.plot(rtriangle_x, rtriangle_y, '-k', lw=1.0)
    ax.plot(diamond_x, diamond_y, '-k', lw=1.0)
    
    # Plot the lines with interval of 20%
    interval = 0.2
    ticklabels = ['0', '20', '40', '60', '80', '100']
    for i, x in enumerate(np.linspace(0, 1, int(1/interval+1))):
        # the left traingle
        ax.plot([x, x - x / 2.0], 
                [0, x / 2.0 * np.tan(np.pi / 3)], 
                'k:', lw=1.0)
        ## the bottom ticks
        if i in [1, 2, 3, 4]: 
            ax.text(x, 0-0.03, ticklabels[-i-1], 
                    ha='center', va='center')
        ax.plot([x, (1-x)/2.0+x], 
                 [0, (1-x)/2.0*np.tan(np.pi/3)], 
                 'k:', lw=1.0)
        ## the right ticks
        if i in [1, 2, 3, 4]:
            ax.text((1-x)/2.0+x + 0.026, (1-x)/2.0*np.tan(np.pi/3) + 0.015, 
                    ticklabels[i], ha='center', va='center', rotation=-60)
        ax.plot([x/2, 1-x/2], 
                [x/2*np.tan(np.pi/3), x/2*np.tan(np.pi/3)], 
                'k:', lw=1.0)
        ## the left ticks
        if i in [1, 2, 3, 4]:
            ax.text(x/2 - 0.026, x/2*np.tan(np.pi/3) + 0.015, 
                    ticklabels[i], ha='center', va='center', rotation=60)
        
        # the right traingle
        ax.plot([x+1+2*offset, x-x/2.0+1+2*offset], 
                [0, x/2.0*np.tan(np.pi/3)], 
                'k:', lw=1.0)
        ## the bottom ticks
        if i in [1, 2, 3, 4]:
            ax.text(x+1+2*offset, 0-0.03, 
                    ticklabels[i], ha='center', va='center')
        ax.plot([x+1+2*offset, (1-x)/2.0+x+1+2*offset],
                 [0, (1-x)/2.0*np.tan(np.pi/3)], 
                 'k:', lw=1.0)
        ## the right ticks
        if i in [1, 2, 3, 4]:
            ax.text((1-x)/2.0+x+1+2*offset  + 0.026, (1-x)/2.0*np.tan(np.pi/3) + 0.015, 
                    ticklabels[-i-1], ha='center', va='center', rotation=-60)
        ax.plot([x/2+1+2*offset, 1-x/2+1+2*offset], 
                [x/2*np.tan(np.pi/3), x/2*np.tan(np.pi/3)], 
                'k:', lw=1.0)
        ## the left ticks
        if i in [1, 2, 3, 4]:
            ax.text(x/2+1+2*offset - 0.026, x/2*np.tan(np.pi/3) + 0.015, 
                    ticklabels[-i-1], ha='center', va='center', rotation=60)
        
        # the diamond
        ax.plot([0.5+offset+0.5/(1/interval)*x/interval, 1+offset+0.5/(1/interval)*x/interval], 
                 [h+offset*np.tan(np.pi/3)+0.5/(1/interval)*x/interval*np.tan(np.pi/3),
                  offset*np.tan(np.pi/3)+0.5/(1/interval)*x/interval*np.tan(np.pi/3)], 
                 'k:', lw=1.0)
        ## the upper left and lower right
        if i in [1, 2, 3, 4]: 
            ax.text(0.5+offset+0.5/(1/interval)*x/interval  - 0.026,
             h+offset*np.tan(np.pi/3)+0.5/(1/interval)*x/interval*np.tan(np.pi/3) + 0.015, ticklabels[i], 
             ha='center', va='center', rotation=60)
            ax.text(1+offset+0.5/(1/interval)*x/interval + 0.026,
             offset*np.tan(np.pi/3)+0.5/(1/interval)*x/interval*np.tan(np.pi/3) - 0.015, ticklabels[-i-1], 
                    ha='center', va='center', rotation=60)
        ax.plot([0.5+offset+0.5/(1/interval)*x/interval, 1+offset+0.5/(1/interval)*x/interval], 
                 [h+offset*np.tan(np.pi/3)-0.5/(1/interval)*x/interval*np.tan(np.pi/3),
                  2*h+offset*np.tan(np.pi/3)-0.5/(1/interval)*x/interval*np.tan(np.pi/3)], 
                 'k:', lw=1.0)
        ## the lower left and upper right
        if i in [1, 2, 3, 4]:  
            ax.text(0.5+offset+0.5/(1/interval)*x/interval- 0.026,
             h+offset*np.tan(np.pi/3)-0.5/(1/interval)*x/interval*np.tan(np.pi/3) - 0.015, ticklabels[i], 
             ha='center', va='center', rotation=-60)
            ax.text(1+offset+0.5/(1/interval)*x/interval + 0.026,
             2*h+offset*np.tan(np.pi/3)-0.5/(1/interval)*x/interval*np.tan(np.pi/3) + 0.015, ticklabels[-i-1], 
             ha='center', va='center', rotation=-60)
    
    # Labels and title
    plt.text(0.5, -offset, '%' + '$Ca^{2+}$', 
             ha='center', va='center')
    plt.text(1+2*offset+0.5, -offset, '%' + '$Cl^{-}$', 
            ha='center', va='center')
    plt.text(0.25-offset*np.cos(np.pi/30), 0.25*np.tan(np.pi/3)+offset*np.sin(np.pi/30), '%' + '$Mg^{2+}$',  
             ha='center', va='center', rotation=60)
    plt.text(1.75+2*offset+offset*np.cos(np.pi/30),
             0.25*np.tan(np.pi/3)+offset*np.sin(np.pi/30), '%' + '$SO_4^{2-}$',  
             ha='center', va='center', rotation=-60)
    plt.text(0.75+offset*np.cos(np.pi/30),
             0.25*np.tan(np.pi/3)+offset*np.sin(np.pi/30), '%' + '$Na^+$' + '+%' + '$K^+$',  
             ha='center', va='center', rotation=-60)
    plt.text(1+2*offset+0.25-offset*np.cos(np.pi/30), 0.25*np.tan(np.pi/3)+offset*np.sin(np.pi/30),
             '%' + '$HCO_3^-$' + '+%' + '$CO_3^{2-}$',  
              ha='center', va='center', rotation=60)
    
    plt.text(0.5+offset+0.5*offset+offset*np.cos(np.pi/30),
             h+offset*np.tan(np.pi/3)+0.25*np.tan(np.pi/3)+offset*np.sin(np.pi/30), '%' + '$SO_4^{2-}$' + '+%' + '$Cl^-$',  
             ha='center', va='center', rotation=60)
    plt.text(1.5+offset-0.25+offset*np.cos(np.pi/30),
             h+offset*np.tan(np.pi/3)+0.25*np.tan(np.pi/3)+offset*np.sin(np.pi/30), '%' + '$Ca^{2+}$' + '+%' + '$Mg^{2+}$', 
              ha='center', va='center', rotation=-60)
    
    # Fill the water types domain
    ## the left traingle
    plt.fill([0.25, 0.5, 0.75, 0.25], 
             [h/2, 0, h/2, h/2], color = (0.8, 0.8, 0.8), zorder=0)
    ## the right traingle
    plt.fill([1+2*offset+0.25, 1+2*offset+0.5, 1+2*offset+0.75, 1+2*offset+0.25], 
             [h/2, 0, h/2, h/2], color = (0.8, 0.8, 0.8), zorder=0)
    ## the diamond
    plt.fill([0.5+offset+0.25, 0.5+offset+0.25+0.5, 0.5+offset+0.25+0.25, 0.5+offset+0.25],
             [h+offset*np.tan(np.pi/3) - 0.5*np.sin(np.pi/3),
              h+offset*np.tan(np.pi/3) - 0.5*np.sin(np.pi/3), h+offset*np.tan(np.pi/3),
              h+offset*np.tan(np.pi/3) - 0.5*np.sin(np.pi/3)], 
             color = (0.8, 0.8, 0.8), zorder=0)
    plt.fill([0.5+offset+0.25, 0.5+offset+0.25+0.25, 0.5+offset+0.25+0.5, 0.5+offset+0.25],
             [h+offset*np.tan(np.pi/3) + 0.5*np.sin(np.pi/3), h+offset*np.tan(np.pi/3),
              h+offset*np.tan(np.pi/3) + 0.5*np.sin(np.pi/3), h+offset*np.tan(np.pi/3) + 0.5*np.sin(np.pi/3)], 
             color = (0.8, 0.8, 0.8), zorder=0)
    
    # Convert unit if needed
    if unit == 'mg/L':
        gmol = np.array([ions_WEIGHT['Ca'], ions_WEIGHT['Mg'], ions_WEIGHT['Na'], ions_WEIGHT['K'], 
                         ions_WEIGHT['HCO3'], ions_WEIGHT['CO3'], ions_WEIGHT['Cl'], ions_WEIGHT['SO4']])
    
        eqmol = np.array([ions_CHARGE['Ca'], ions_CHARGE['Mg'], ions_CHARGE['Na'], ions_CHARGE['K'], 
                          ions_CHARGE['HCO3'], ions_CHARGE['CO3'], ions_CHARGE['Cl'], ions_CHARGE['SO4']])
    
        tmpdf = df[['Ca', 'Mg', 'Na', 'K', 'HCO3', 'CO3', 'Cl', 'SO4']]
        dat = tmpdf.values
        
        meqL = (dat / abs(gmol)) * abs(eqmol)
        
    elif unit == 'meq/L':
        meqL = df[['Ca', 'Mg', 'Na', 'K', 'HCO3', 'CO3', 'Cl', 'SO4']].values
    
    else:
        raise ValueError("""
        Currently only mg/L and meq/L are supported.
        Convert the unit if needed.""")
    
    # Calculate the percentages
    sumcat = np.sum(meqL[:, 0:4], axis=1)
    suman = np.sum(meqL[:, 4:8], axis=1)
    cat = np.zeros((dat.shape[0], 3))
    an = np.zeros((dat.shape[0], 3))
    cat[:, 0] = meqL[:, 0] / sumcat                  # Ca
    cat[:, 1] = meqL[:, 1] / sumcat                  # Mg
    cat[:, 2] = (meqL[:, 2] + meqL[:, 3]) / sumcat   # Na+K
    an[:, 0] = (meqL[:, 4] + meqL[:, 5]) / suman     # HCO3 + CO3
    an[:, 2] = meqL[:, 6] / suman                    # Cl
    an[:, 1] = meqL[:, 7] / suman                    # SO4

    # Convert into cartesian coordinates
    cat_x = 0.5 * (2 * cat[:, 2] + cat[:, 1])
    cat_y = h * cat[:, 1]
    an_x = 1 + 2 * offset + 0.5 * (2 * an[:, 2] + an[:, 1])
    an_y = h * an[:, 1]
    d_x = an_y / (4 * h) + 0.5 * an_x - cat_y / (4 * h) + 0.5 * cat_x
    d_y = 0.5 * an_y + h * an_x + 0.5 * cat_y - h * cat_x

    # Plot the scatters
    Labels = []
    for i in range(len(df)):
        if (df.at[i, 'Label'] in Labels or df.at[i, 'Label'] == ''):
            TmpLabel = ''
        else:
            TmpLabel = df.at[i, 'Label']
            Labels.append(TmpLabel)
         
        try:
            if (df['Color'].dtype is np.dtype('float')) or \
                (df['Color'].dtype is np.dtype('int64')):
                vmin = np.min(df['Color'].values)
                vmax = np.max(df['Color'].values)
                cf = plt.scatter(cat_x[i], cat_y[i], 
                                marker=df.at[i, 'Marker'],
                                s=df.at[i, 'Size'], 
                                c=df.at[i, 'Color'], vmin=vmin, vmax=vmax,
                                alpha=df.at[i, 'Alpha'],
                                #label=TmpLabel, 
                                edgecolors='black')
                plt.scatter(an_x[i], an_y[i], 
                            marker=df.at[i, 'Marker'],
                            s=df.at[i, 'Size'], 
                            c=df.at[i, 'Color'], vmin=vmin, vmax=vmax,
                            alpha=df.at[i, 'Alpha'],
                            label=TmpLabel, 
                            edgecolors='black')
                plt.scatter(d_x[i], d_y[i], 
                            marker=df.at[i, 'Marker'],
                            s=df.at[i, 'Size'], 
                            c=df.at[i, 'Color'], vmin=vmin, vmax=vmax,
                            alpha=df.at[i, 'Alpha'],
                            #label=TmpLabel, 
                            edgecolors='black')
                
            else:
                plt.scatter(cat_x[i], cat_y[i], 
                            marker=df.at[i, 'Marker'],
                            s=df.at[i, 'Size'], 
                            c=df.at[i, 'Color'], 
                            alpha=df.at[i, 'Alpha'],
                            #label=TmpLabel, 
                            edgecolors='black')
                plt.scatter(an_x[i], an_y[i], 
                            marker=df.at[i, 'Marker'],
                            s=df.at[i, 'Size'], 
                            c=df.at[i, 'Color'], 
                            alpha=df.at[i, 'Alpha'],
                            label=TmpLabel, 
                            edgecolors='black')
                plt.scatter(d_x[i], d_y[i], 
                            marker=df.at[i, 'Marker'],
                            s=df.at[i, 'Size'], 
                            c=df.at[i, 'Color'], 
                            alpha=df.at[i, 'Alpha'],
                            #label=TmpLabel, 
                            edgecolors='black')
                
        except(ValueError):
            pass
            
    # Creat the legend
    if (df['Color'].dtype is np.dtype('float')) or (df['Color'].dtype is np.dtype('int64')):
        cb = plt.colorbar(cf, extend='both', spacing='uniform',
                          orientation='vertical', fraction=0.025, pad=0.05)
        cb.ax.set_ylabel('$TDS$' + ' ' + '$(mg/L)$', rotation=90, labelpad=-75)
    
    plt.legend(bbox_to_anchor=(0.15, 0.875), markerscale=1,
               frameon=False, 
               labelspacing=0.25, handletextpad=0.25)
   
    # Save the figure
    plt.savefig(file_path, bbox_inches='tight', dpi=dpi)
    print('Se ha grabado:', file_path)

    plt.clf()  # Clear the current figure.
    plt.close()  # Close the figure window.
    plt.rcParams.update(plt.rcParamsDefault)  # Reset the parameters to their default values.    


def stiff(df: pd.DataFrame, dir_path: Path, unit='mg/L', figformat='png', dpi=200, verbose=True) -> None:
    """
    Plots the Stiff diagram for each Sample in df
    https://github.com/jyangfsu/WQChartPy/tree/main/wqchartpy
    
    Parameters
    ----------
    df : pandas.DataFrame
        Geochemical data to draw Gibbs diagram.
    dir_path : str or Path
        Directory path where figures will be saved.        
    unit : str
        The unit used in df (mg/L or meq/L). 
    figformat : str default 'png'
        The file format, e.g. 'png', 'pdf', 'svg'
    dpi : image figure resolution
    verbose : print the Sample of each figure
    """
    # Basic data check 
    # -------------------------------------------------------------------------
    # Determine if the required geochemical parameters are defined. 
    if not {'Sample', 'Ca', 'Mg', 'Na', 'K', 'HCO3', 'Cl', 'SO4'}.issubset(df.columns):
        raise ValueError("Dataframe must have columns: Sample, Ca, Mg, Na, K, HCO3, Cl, SO4")
        
    # Determine if the provided unit is allowed
    ALLOWED_UNITS = ['mg/L', 'meq/L']
    if unit not in ALLOWED_UNITS:
        raise ValueError("Unit must be mg/L or meq/L")

    plt.style.use('default')
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 11
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 11   

    ions_WEIGHT = {key: value[0] for key, value in ions.items()}
    ions_CHARGE = {key: value[1] for key, value in ions.items()}

    # Convert unit if needed
    if unit == 'mg/L':
        gmol = np.array([ions_WEIGHT['Ca'], ions_WEIGHT['Mg'], ions_WEIGHT['Na'], ions_WEIGHT['K'], 
                         ions_WEIGHT['HCO3'], ions_WEIGHT['Cl'], ions_WEIGHT['SO4']])
    
        eqmol = np.array([ions_CHARGE['Ca'], ions_CHARGE['Mg'], ions_CHARGE['Na'], ions_CHARGE['K'], 
                          ions_CHARGE['HCO3'], ions_CHARGE['Cl'], ions_CHARGE['SO4']])
    
        tmpdf = df[['Ca', 'Mg', 'Na', 'K', 'HCO3', 'Cl', 'SO4']]
        dat = tmpdf.values
        
        meqL = (dat / abs(gmol)) * abs(eqmol)
        
    else:
        meqL = df[['Ca', 'Mg', 'Na', 'K', 'HCO3', 'Cl', 'SO4']].values
    
   
    cat_max = np.max(np.array(((meqL[:, 2] + meqL[:, 3]), meqL[:, 0], meqL[:, 1])))
    an_max = np.max(meqL[:, 4:])
    
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

    figformat = figformat.replace('.','')
    for i, (index, row) in enumerate(df.iterrows()):
        
        x = [-(meqL[i, 2] + meqL[i, 3]), -meqL[i, 0], -meqL[i, 1], 
                meqL[i, 6], meqL[i, 4], meqL[i, 5], -(meqL[i, 2] + meqL[i, 3])]
        y = [3, 2, 1, 1, 2, 3, 3]
        
        plt.figure(figsize=(3, 3))
        plt.fill(x, y, facecolor='w', edgecolor='k', linewidth=1.25)
        
        plt.plot([0, 0], [1, 3], 'k-.', linewidth=1.25)
        plt.plot([-0.5, 0.5], [2, 2], 'k-')

        cmax = cat_max if cat_max > an_max else an_max
        plt.xlim([-cmax, cmax])
        plt.text(-cmax, 2.9, 'Na$^+$' + '+' + 'K$^+$', ha= 'right')
        plt.text(-cmax, 1.9, 'Ca$^{2+}$', ha= 'right')
        plt.text(-cmax, 1.0, 'Mg$^{2+}$', ha= 'right')
        
        plt.text(cmax, 2.9,'Cl$^-$', ha= 'left')
        plt.text(cmax, 1.9,'HCO'+'$_{3}^-$',ha= 'left')
        plt.text(cmax, 1.0,'SO'+'$_{4}^{2-}$',ha= 'left')
        
        ax = plt.gca()
        ax.spines['left'].set_color('None')
        ax.spines['right'].set_color('None')
        ax.spines['top'].set_color('None')
        ax.spines['bottom'].set_linewidth(1.25)
        ax.spines['bottom'].set_color('k')
        plt.ylim(0.8, 3.2)
        plt.yticks([])
        plt.gca().yaxis.set_visible(False)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        ticks = np.array([-cmax, -cmax/2, 0, cmax/2, cmax])
        tickla = [f'{tick:1.0f}' for tick in abs(ticks)]
        ax.xaxis.set_ticks(ticks)
        ax.xaxis.set_ticklabels(tickla)

        ax.set_xlabel('Stiff diagram (meq/L)', weight='normal')
            
        ax.set_title(row['Sample'], weight='normal')

        file_name = str(row['Sample']) + '_stiff.' + figformat
        if verbose:
            print(file_name)
        plt.savefig(dir_path.joinpath(file_name), bbox_inches='tight', dpi=dpi)
        
        plt.clf()  # Clear the current figure.
        plt.close()  # Close the figure window.
    plt.rcParams.update(plt.rcParamsDefault)  # Reset the parameters to their default values.    


def schoeller(df, file_path: Path, unit='mg/L', dpi=200) -> None:
    """
    Plots Schoeller  diagram.
    
    Parameters
    ----------
    df : pd.DataFrame
        Geochemical data to draw HFE-D diagram.
    file_path : str or Path. 
        File path where figure wil saved.
    unit : str. 
        Units used in df (mg/L or meq/L).
    dpi : int
        Resolution of the digital image (dots per inch)
    """
    # Determine if the required geochemical parameters are defined. 
    if not {'Ca', 'Mg', 'Na', 'K', 'Cl', 'SO4', 'HCO3'}.issubset(df.columns):
        raise RuntimeError("Schoeller diagram needs Ca, Mg, Na, K, Cl, SO4, and HCO3 in the columns of df")
        
    # Determine if the provided unit is allowed
    ALLOWED_UNITS = ['mg/L', 'meq/L']
    if unit not in ALLOWED_UNITS:
        raise ValueError("unit must be mg/L or  meq/L")

    plt.style.use('default')
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 9
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.titlesize'] = 10

    ions_WEIGHT = {key: value[0] for key, value in ions.items()}
    ions_CHARGE = {key: value[1] for key, value in ions.items()}
    
    # Convert unit if needed
    # -------------------------------------------------------------------------
    if unit == 'mg/L':
        gmol = np.array([ions_WEIGHT['Ca'], ions_WEIGHT['Mg'], ions_WEIGHT['Na'], 
                         ions_WEIGHT['K'], ions_WEIGHT['Cl'], ions_WEIGHT['SO4'],
                         ions_WEIGHT['HCO3']])
    
        eqmol = np.array([ions_CHARGE['Ca'], ions_CHARGE['Mg'], ions_CHARGE['Na'], 
                          ions_CHARGE['K'], ions_CHARGE['Cl'], ions_CHARGE['SO4'],
                          ions_CHARGE['HCO3']])
    
        tmpdf = df[['Ca', 'Mg', 'Na', 'K', 'Cl', 'SO4', 'HCO3']]
        dat = tmpdf.values
        
        meqL = (dat / abs(gmol)) * abs(eqmol)
        
    else:
        meqL = df[['Ca', 'Mg', 'Na', 'K', 'Cl', 'SO4', 'HCO3']].values
        
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(111)
    ax.semilogy()
    
    # Plot the lines
    # -------------------------------------------------------------------------
    Labels = []
    for i in range(len(df)):
        if (df.at[i, 'Label'] in Labels or df.at[i, 'Label'] == ''):
            TmpLabel = ''
        else:
            TmpLabel = df.at[i, 'Label']
            Labels.append(TmpLabel)
    
        try:
            ax.plot([1, 2, 3, 4, 5, 6, 7], meqL[i, :], 
                    marker=df.at[i, 'Marker'],
                    color=df.at[i, 'Color'], 
                    alpha=df.at[i, 'Alpha'],
                    label=TmpLabel) 
        except(ValueError):
                pass
            
    # Background settings
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_xticklabels(['Ca$^{2+}$', 'Mg$^{2+}$', 'Na$^+$', 'K$^+$', 
                        'Cl$^-$', 'SO$_4^{2-}$', 'HCO$_3^-$'])
    ax.set_ylabel('meq/L', fontsize=12, weight='normal')
    
    # Set the limits
    ax.set_xlim([1, 7])
    ax.set_ylim([np.min(meqL) * 0.5, np.max(meqL) * 1.5])
    
    # Plot the vertical lines
    for xtick in [1, 2, 3, 4, 5, 6, 7]:
        plt.axvline(xtick, linewidth=1, color='grey', linestyle='dashed')
            
    # Creat the legend
    ax.legend(loc='best', markerscale=1, frameon=False,
              labelspacing=0.25, handletextpad=0.25)
    
    # Save the figure
    plt.savefig(file_path, bbox_inches='tight', dpi=dpi)
    print('Se ha grabado:', file_path)
    
    plt.clf()  # Clear the current figure.
    plt.close()  # Close the figure window.
    plt.rcParams.update(plt.rcParamsDefault)  # Reset the parameters to their default values.    