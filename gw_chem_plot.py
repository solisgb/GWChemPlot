# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:55:25 2023

@author: solis

fork of WQChartPy https://github.com/jyangfsu/WQChartPy
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import traceback
    
from ions import Ions
import littleLogging as myLogging


_graph_parameters = \
    {'Sample':'Identificador del análisis',
     'Label':'Identificador del análisis en el diagrama (gráfico)',
     'Color': 'Color de Label',
     'Marker': 'Marcador (símbolo) de Label',
     'Size': 'Tamaño en dpi del marcador',
     'Alpha': 'Transparencia del marcador (superposiciones en el gráfico)'}


def graph_parameters_get() -> dict:
    return _graph_parameters


class GWChemPlot():
    """
    Class to create Piper, Stiff and Scoeller diagrams to be saved an
        image file.
    """

    
    def __init__(self, df: pd.DataFrame, unit: str='mg/L', dpi: int=150):
        """
        Instanciate
        
        Parameters
        ----------
        _data. Hydrochemical data to draw diagrams.
        _unit. The unit used in df (mg/L or meq/L). 
        _dpi. Dot per inch in graph output files
        _ions. Ions object
        """
        if unit not in ['mg/L', 'meq/L']:
           raise ValueError('units must be mg/L or meq/L')

        self.check_column_names(df)

        df.dropna(inplace=True)
        
        #self._atr = {'df': df.copy(), 'unit': unit, 'dpi': dpi}
        
        self._data: pd.DataFrame = df.copy()
        self._unit: str = unit
        self._dpi: int = dpi
        self._ions: Ions = Ions()


    @property
    def df(self):
        return self._data    

    
    @property
    def data(self):
        return self._data


    @data.setter
    def data(self, new_df: pd.DataFrame) -> None:
        self.check_column_names(new_df)
        new_df = new_df.dropna()
        self._data = new_df.copy()


    @property
    def unit(self):
        return self._unit
    
    
    @property
    def dpi(self):
        return self._dpi


    def check_column_names(self, df: pd.DataFrame) -> None:
        """
        Checks if self._data has a subset of columns required to make graphs
    
        Raises
        ------
        ValueError
            If not required a column name is present.
    
        """
        required_col_names = ['Sample', 'Label', 'Color', 'Marker', 'Size',
                              'Alpha']
        
        absent = [c1 for c1 in required_col_names if c1 not in df.columns]
        if len(absent) > 0:
            a = ','.join(absent)
            msg = f'The data file lacks the required columns {a}'
            myLogging.append(msg)
            raise ValueError(msg)

        required_col_names = ['HCO3', 'Cl', 'SO4', 'Na', 'K', 'Ca', 'Mg']
        absent = [c1 for c1 in required_col_names if c1 not in df.columns]
        if len(absent) > 0:
            a = ','.join(absent)
            msg = f'The data file lacks the required columns {a}'
            myLogging.append(msg)
            raise ValueError(msg)


    def check_columns_are_present(self, cols: list[str]) -> bool:
        """
        Determines if the column names in cols are present in self.df.
        If any column is not present throws an exception; 
        otherwise returns True

        Parameters
        ----------
        cols : list
            column names.
        """
        absent = [c1 for c1 in cols if c1 not in self.df.columns]
        if len(absent) > 0:
            a = ','.join(absent)
            msg = f'The data file lacks the columns {a}'
            myLogging.append(msg)
            raise ValueError(msg)
        return True


    def required_columns_graph(self, graph_name: str) -> list:
        """
        Depending on the type of chart, columns are required in an order given
        by the implementation of the function that builds the chart.
        This function returns the columns in the required order and checks that
        they are present in self.df.

        Parameters
        ----------
        graph_name : graph's name

        Raises
        ------
        ValueError
            graph_name is not correct
            some required columns are not present en self.df
        """
        if graph_name == 'Shoeller':
            cols = ['Ca', 'Mg', 'Na', 'K', 'Cl', 'SO4', 'HCO3']
        elif graph_name == 'Stiff':
            cols = ['Ca', 'Mg', 'Na', 'K', 'HCO3', 'Cl', 'SO4']
        elif graph_name == 'Piper':
            cols = ['Ca', 'Mg', 'Na', 'K', 'HCO3', 'CO3', 'Cl', 'SO4']
        else:
            msg = f'Graph name not supported: {graph_name}'
            myLogging.append(msg)
            raise ValueError (msg)
        
        self.check_columns_are_present(cols)
        return cols


    def meqL_get(self, ion_names: [str]=[]) -> pd.DataFrame:
        """
        Calculate the meq/L of the ions in ion_names.

        Parameters
        ----------
        col_names : list of ion's names 
        """
        if not ion_names:
            ion_names = self._ions.ions_in_df(self.df)
  
        if self.unit == 'meq/L':
            return self.df[ion_names]
        else:
            x = self._ions.charge_weight_ratio_get(ion_names)
            return self.df[ion_names].mul(x.iloc[0])


    def cbe(self) -> pd.DataFrame:
        """
        Get the charge balance error (cbe). The returned DataFrame has
        the columns: 'Sample' and 'cbe'
        """
        anions = self._ions.anions_in_df(self.df)
        cations = self._ions.cations_in_df(self.df)
        if self.unit == 'mg/L': 
            anions_meqL = self.meqL_get(anions)
            cations_meqL = self.meqL_get(cations)
        else:
            anions_meqL = self.df[anions]
            cations_meqL = self.df[cations]

        sum_anions = np.sum(anions_meqL.values, axis=1).reshape(-1, 1)
        sum_cations = np.sum(cations_meqL.values, axis=1).reshape(-1, 1)

        x = 100. * (sum_cations - sum_anions) / (sum_cations + sum_anions)
        
        samples = self.df[['Sample']].copy()
        
        cbe = pd.concat([samples, anions_meqL, cations_meqL], axis=1)
        cbe['cbe'] = x 
        return cbe


    def plot_Shoeller(self, figname:str) -> None:
        """
        Builds a file with the Schoeller chart
        References
            Güler, et al. 2002. Evaluation of graphical and multivariate
            statistical methods for classification of water chemistry data
            Hydrogeology Journal 10(4):455-474
            https://doi.org/10.1007/s10040-002-0196-6        

        Parameters
        ----------
        figname : Name and extension of the output file (path included).
        """
    
        col_names = self.required_columns_graph('Shoeller')
        meqL = self.meqL_get(col_names).values

        fig = plt.figure(figsize=(6, 3))
        ax = fig.add_subplot(111)
        ax.semilogy()
        
        # Plot the lines
        Labels = []
        for i, row in self.df.iterrows():
            if (row.Label in Labels or row.Label == ''):
                TmpLabel = ''
            else:
                TmpLabel = row.Label
                Labels.append(row.Label)
        
            try:
                ax.plot([1, 2, 3, 4, 5, 6, 7], meqL[i, :], 
                        marker=row.Marker,
                        color=row.Color, 
                        alpha=row.Alpha,
                        label=TmpLabel) 
            except Exception:
                msg = traceback.format_exc()
                myLogging.append(msg)
                
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
        ax.legend(loc='best', markerscale=1, frameon=False, fontsize=10,
                  labelspacing=0.25, handletextpad=0.25)
        
        # Save the figure
        plt.savefig(figname, bbox_inches='tight', dpi=self.dpi)
        
        # Display the info
        print("Schoeller plot saved")


    def plot_Stiff(self, figname:str) -> None:
        """
        Builds a Stiff chart file for each water anaysis in self.df
        The name of each of the files is formed by inserting before the
        figname extension the Label of each water analysis.
        References
        ----------
            Stiff, H.A. 1951. The Interpretation of Chemical Water Analysis by 
            Means of Patterns. Journal of Petroleum Technology 3(10): 15-3
            https://doi.org/10.2118/951376-G        

        Parameters
        ----------
        figname : Template name and extension of the output file
        (path included)
        """
        col_names = self.required_columns_graph('Stiff')
        meqL = self.meqL_get(col_names).values
   
        cat_max = np.max(np.array(((meqL[:, 2] + meqL[:, 3]), meqL[:, 0],
                                   meqL[:, 1])))
        an_max = np.max(meqL[:, 4:])
    
        parent = pathlib.Path(figname).parent
        stem = pathlib.Path(figname).stem
        suffix = pathlib.Path(figname).suffix
        
        # Plot the Stiff diagrams for each sample
        nsucces = 0
        for i, row in self.df.iterrows():

            x = [-(meqL[i, 2] + meqL[i, 3]), -meqL[i, 0], -meqL[i, 1], 
                 meqL[i, 6], meqL[i, 4], meqL[i, 5],
                 -(meqL[i, 2] + meqL[i, 3])]
            y = [3, 2, 1, 1, 2, 3, 3]
            
            plt.figure(figsize=(3, 3))
            plt.fill(x, y, facecolor='w', edgecolor='k', linewidth=1.25)
            
            plt.plot([0, 0], [1, 3], 'k-.', linewidth=1.25)
            plt.plot([-0.5, 0.5], [2, 2], 'k-')
  
            cmax = cat_max if cat_max > an_max else an_max
            plt.xlim([-cmax, cmax])
            plt.text(-cmax, 2.9, 'Na$^+$' + '+' + 'K$^+$', fontsize=12,
                     ha= 'right')
            plt.text(-cmax, 1.9, 'Ca$^{2+}$', fontsize=12, ha= 'right')
            plt.text(-cmax, 1.0, 'Mg$^{2+}$', fontsize=12, ha= 'right')
            
            plt.text(cmax, 2.9,'Cl$^-$',fontsize=12, ha= 'left')
            plt.text(cmax, 1.9,'HCO'+'$_{3}^-$',fontsize=12,ha= 'left')
            plt.text(cmax, 1.0,'SO'+'$_{4}^{2-}$',fontsize=12,ha= 'left')
            
            ax = plt.gca()
            ax.spines['left'].set_color('None')
            ax.spines['right'].set_color('None')
            ax.spines['top'].set_color('None')
            plt.minorticks_off()
            plt.tick_params(which='major', direction='out', length=4,
                            width=1.25)
            plt.tick_params(which='minor', direction='in', length=2,
                            width=1.25)
            ax.spines['bottom'].set_linewidth(1.25)
            ax.spines['bottom'].set_color('k')
            #ylim(0.8, 3.2)
            plt.setp(plt.gca(), yticks=[], yticklabels=[])
            #plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            ticks = np.array([-cmax, -cmax/2, 0, cmax/2, cmax])
            tickla = [f'{tick:1.0f}' for tick in abs(ticks)]
            ax.xaxis.set_ticks(ticks)
            ax.xaxis.set_ticklabels(tickla)
            
            labels = ax.get_xticklabels()
            [label.set_fontsize(10) for label in labels]
            ax.set_xlabel('meq/L', fontsize=12,
                          weight='normal')
                
            ax.set_title(row.Sample, fontsize=14, weight='normal')

            a = row.Sample
            if '/' in a:
                a = a.replace('/', '-')

            fo_stiff = \
                pathlib.PureWindowsPath.joinpath(parent,\
                    f'{stem}_{a}{suffix}')
    
            # Save the figure
            plt.savefig(fo_stiff, bbox_inches='tight', dpi=self.dpi)
            plt.close()
            nsucces += 1
            print(f'Stiff diagram {stem}_{a}{suffix} saved')

        
        # Display the info
        if len(self.df) == nsucces:
            print("Stiff plots saved")
        elif nsucces == 0:
            print("Stiff plots have not been saved, exceptions have occurred")
        else:
            print("Some Stiff plots have not been saved, " +\
                  "exceptions have occurred")


    def anions_meqL_normalization(self) -> pd.DataFrame: 
        """
        anions (meq/L) normalization. Used in:
            Ion dominant classification
            Piper Plot
        """
        col_names = self._ions.anions_in_df(self.df)
        meqL = self.meqL_get(col_names)
        xsum = meqL.sum(axis=1)
        if 'CO3' not in col_names:
            xan = pd.DataFrame(meqL.HCO3 / xsum, columns=['HCO3'])
        else:
            x = meqL.HCO3 + meqL.CO3 
            xan = pd.DataFrame(x / xsum, columns=['HCO3_CO3'])
        if 'NO3' not in col_names:
            xan['Cl'] = meqL.Cl / xsum
        else: 
            x = meqL.Cl + meqL.NO3
            xan['Cl_NO3'] = x / xsum
        xan['SO4'] = meqL.SO4 / xsum

        return xan


    def cations_meqL_normalization(self) -> pd.DataFrame: 
        """
        cations (meq/L) normalization. Used in:
            Ion dominant classification
            Piper Plot
        """
        col_names = self._ions.cations_in_df(self.df)
        meqL = self.meqL_get(col_names)
        xsum = meqL.sum(axis=1)
        x = meqL.Na + meqL.K 
        xca = pd.DataFrame(x / xsum, columns=['Na_K'])
        xca['Ca'] = meqL.Ca / xsum
        xca['Mg'] = meqL.Mg / xsum

        return xca


    def _carbonate_col_name(self, col_names: [str]) -> 'str':
        if 'HCO3_CO3' in col_names:
            return 'HCO3_CO3'
        else:
            return 'HCO3'


    def _carbonate_piper_axis(self, col_names: [str]) -> 'str':
        if 'HCO3_CO3' in col_names:
            return '%' + '$HCO_3^-$' + '+%' + '$CO_3^{2-}$'
        else:
            return '%' + '$HCO_3^-$'       


    def _cloride_col_name(self, col_names: [str]) -> 'str':
        if 'Cl_NO3' in col_names:
            return 'Cl_NO3'
        else:
            return 'Cl'


    def _cloride_piper_axis(self, col_names: [str]) -> 'str':
        if 'Cl_NO3' in col_names:
            return '%' + '$Cl^{-}$' + '%' + '$NO_3^{-}$'
        else:
            return '%' + '$Cl^{-}$'


    def _cloride_label(self, col_names: [str]) -> 'str':
        if 'Cl_NO3' in col_names:
            return 'Cloruro nitratada'
        else:
            return 'Clorurada'


    def ion_dominant_classification(self) -> pd.DataFrame:
        """
        Groundwater ion dominant classification
        Custodio E (1983). Hidrogeoquímica. 
        In Hidrología subterránea pp 1001-1095. Ed. Omga
        """

        an = self.anions_meqL_normalization()
        ca = self.cations_meqL_normalization()

        samples = self.df[['Sample']].copy()
        
        idc = pd.concat([samples, an, ca], axis=1)
        col_names = idc.columns
        carbonate_col_name = self._carbonate_col_name(col_names)
        cloride_col_name = self._cloride_col_name(col_names)
        cloride_label = self._cloride_label(col_names)
        
        idc_cations = []
        idc_anions = []
        for i, r in idc.iterrows():        
            if r.Na_K > r.Mg > r.Ca:
                if r.Na_K >= 0.5:
                    idc_cations.append('Sódica (magnésico-cálcica)')
                else:
                    idc_cations.append('Mixta (sódica-magnésico-cálcica)')
            elif r.Na_K > r.Ca > r.Mg:
                if r.Na >= 0.5:
                    idc_cations.append('Sódica (magnésico-cálcica)')
                else:
                    idc_cations.append('Mixta (sódica-cálcica-magnésica)')
            elif r.Mg > r.Na_K > r.Ca:
                if r.Mg >= 0.5:
                    idc_cations.append('Magnésica (sódico-cálcica)')
                else:
                    idc_cations.append('Mixta (magnésico-sódico-cálcica)')               
            elif r.Mg > r.Ca > r.Na_K:
                if r.Mg >= 0.5:
                    idc_cations.append('Magnésica (cálcico-sódica)')
                else:
                    idc_cations.append('Mixta (magnésico-cálcico-sódica)')                
            elif r.Ca > r.Na_K > r.Mg:
                if r.Ca >= 0.5:
                    idc_cations.append('Cálcica (sódico-magnésica)')
                else:
                    idc_cations.append('Mixta (cálcico-sódico-magnésica)')                
            elif r.Ca > r.Mg > r.Na_K:
                if r.Ca >= 0.5:
                    idc_cations.append('Cálcica (magnésico-sódica)')
                else:
                    idc_cations.append('Mixta (cálcico-magnésico-sódica)')                

            if r[cloride_col_name] > r['SO4'] > r[carbonate_col_name]:
                if r[cloride_col_name] >= 0.5:
                    idc_anions.append(f'{cloride_label} (sulfatada-bicarbonatada)')
                else:
                    idc_anions.append(f'Mixta ({cloride_label.lower()}-sulfatada-bicarbonatada)')               
            elif r[cloride_col_name] > r[carbonate_col_name] > r['SO4']:
                if r[cloride_col_name] >= 0.5:
                    idc_anions.append(f'{cloride_label} (bicarbonatada-sulfatada)')
                else:
                    idc_anions.append(f'Mixta ({cloride_label.lower()}-bicarbonatada-sulfatada)')               
            elif r['SO4'] > r[cloride_col_name] > r[carbonate_col_name] :
                if r['SO4'] >= 0.5:
                    idc_anions.append(f'Sulfatada ({cloride_label.lower()}-bicarbonatada)')
                else:
                    idc_anions.append(f'Mixta (sulfatada-{cloride_label.lower()}-bicarbonatada)')               
            elif r['SO4'] > r[carbonate_col_name] > r[cloride_col_name]:
                if r['SO4'] >= 0.5:
                    idc_anions.append(f'Sulfatada (bicarbonatada-{cloride_label.lower()})')
                else:
                    idc_anions.append(f'Mixta (sulfatada-bicarbonatada-{cloride_label.lower()})')
            elif r[carbonate_col_name] > r[cloride_col_name] > r['SO4']:
                if r[carbonate_col_name] >= 0.5:
                    idc_anions.append(f'Bicarbonatada ({cloride_label.lower()}-sulfatada)')
                else:
                    idc_anions.append(f'Mixta (bicarbonatada-{cloride_label.lower()}-sulfatada)')            
            elif r[carbonate_col_name] > r['SO4'] > r[cloride_col_name]:
                if r[carbonate_col_name] >= 0.5:
                    idc_anions.append(f'Bicarbonatada (sulfatada-{cloride_label.lower()})')
                else:
                    idc_anions.append(f'Mixta (bicarbonatada-(sulfatada-{cloride_label.lower()})')            

        idc['cations_classified'] = idc_cations
        idc['anions_classified'] = idc_anions
        return idc


    def plot_Piper(self, figname:str) -> None:
        """
        Builds a file with the Piperr chart

        Parameters
        ----------
        figname : Name and extension of the output file (path included)
        """
        col_names = self.required_columns_graph('Piper')
        meqL = self.meqL_get(col_names).values
            
        # Global plot settings
        # --------------------------------------------------------------------
        # Change default settings for figures
        plt.style.use('default')
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.titlesize'] = 10
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 10   
        
        # Plot background settings
        # --------------------------------------------------------------------
        # Define the offset between the diamond and triangle
        offset = 0.10                         
        # offsety = offset * np.tan(np.pi / 3.0)
        h = 0.5 * np.tan(np.pi / 3.0)
        
        # Calculate the triangles' location 
        ltriangle_x = np.array([0, 0.5, 1, 0])
        ltriangle_y = np.array([0, h, 0, 0])
        rtriangle_x = ltriangle_x + 2 * offset + 1
        rtriangle_y = ltriangle_y
        
        # Calculate the diamond's location 
        diamond_x = np.array([0.5, 1, 1.5, 1, 0.5]) + offset
        diamond_y = h * (np.array([1, 2, 1, 0, 1])) + \
            (offset * np.tan(np.pi / 3))
        
        # Plot the triangles and diamond
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
                ax.text((1-x)/2.0+x + 0.026, 
                        (1-x)/2.0*np.tan(np.pi/3) + 0.015, 
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
                ax.text((1-x)/2.0+x+1+2*offset  + 0.026,
                        (1-x)/2.0*np.tan(np.pi/3) + 0.015, 
                        ticklabels[-i-1], ha='center', va='center',
                        rotation=-60)
            ax.plot([x/2+1+2*offset, 1-x/2+1+2*offset], 
                    [x/2*np.tan(np.pi/3), x/2*np.tan(np.pi/3)], 'k:', lw=1.0)
            ## the left ticks
            if i in [1, 2, 3, 4]:
                ax.text(x/2+1+2*offset - 0.026, x/2*np.tan(np.pi/3) + 0.015, 
                        ticklabels[-i-1], ha='center', va='center',
                        rotation=60)
            
            # the diamond
            ax.plot([0.5 + offset + 0.5 / (1/interval) * x/interval,
                     1 + offset + 0.5 / (1/interval) * x / interval], 
                     [h + offset * np.tan(np.pi/3) + 0.5 / (1/interval) *\
                      x / interval * np.tan(np.pi / 3), 
                      offset * np.tan(np.pi / 3) + 0.5 / (1/interval) *\
                          x/interval*np.tan(np.pi/3)], 'k:', lw=1.0)
            ## the upper left and lower right
            if i in [1, 2, 3, 4]: 
                ax.text(0.5+offset+0.5/(1/interval)*x/interval - 0.026, 
                        h+offset*np.tan(np.pi/3)+0.5/(1/interval)*x\
                            /interval*np.tan(np.pi/3) + 0.015, ticklabels[i], 
                        ha='center', va='center', rotation=60)
                ax.text(1+offset+0.5/(1/interval)*x/interval + 0.026,
                        offset*np.tan(np.pi/3)+0.5/(1/interval)*x \
                            /interval*np.tan(np.pi/3) - 0.015,
                            ticklabels[-i-1], 
                        ha='center', va='center', rotation=60)
            ax.plot([0.5+offset+0.5/(1/interval)*x/interval,
                     1+offset+0.5/(1/interval)*x/interval], 
                     [h+offset*np.tan(np.pi/3)-0.5/(1/interval)*x\
                      /interval*np.tan(np.pi/3),
                      2*h+offset*np.tan(np.pi/3)-0.5/(1/interval)*\
                          x/interval*np.tan(np.pi/3)],'k:', lw=1.0)
            ## the lower left and upper right
            if i in [1, 2, 3, 4]:  
                ax.text(0.5+offset+0.5/(1/interval)*x/interval- 0.026,
                        h+offset*np.tan(np.pi/3)-0.5/(1/interval)*x\
                            /interval*np.tan(np.pi/3) - 0.015, ticklabels[i], 
                        ha='center', va='center', rotation=-60)
                ax.text(1+offset+0.5/(1/interval)*x/interval + 0.026,
                        2*h+offset*np.tan(np.pi/3)-0.5/(1/interval)*\
                            x/interval*np.tan(np.pi/3) + 0.015,
                            ticklabels[-i-1], ha='center', va='center',
                            rotation=-60)
        
        # Labels and title
        plt.text(0.5, -offset, '%' + '$Ca^{2+}$', 
                 ha='center', va='center', fontsize=12)
        plt.text(1+2*offset+0.5, -offset, '%' + '$Cl^{-}$', 
                ha='center', va='center', fontsize=12)
        plt.text(0.25-offset*np.cos(np.pi/30), 0.25*np.tan(np.pi/3)+\
                 offset*np.sin(np.pi/30), '%' + '$Mg^{2+}$',  
                 ha='center', va='center', rotation=60, fontsize=12)
        plt.text(1.75+2*offset+offset*np.cos(np.pi/30), 0.25*np.tan(np.pi/3)+\
                 offset*np.sin(np.pi/30), '%' + '$SO_4^{2-}$',  
                  ha='center', va='center', rotation=-60, fontsize=12)
        plt.text(0.75+offset*np.cos(np.pi/30), 0.25*np.tan(np.pi/3)+offset*\
                 np.sin(np.pi/30), '%' + '$Na^+$' + '+%' + '$K^+$',  
                  ha='center', va='center', rotation=-60, fontsize=12)
        plt.text(1+2*offset+0.25-offset*np.cos(np.pi/30),
                 0.25*np.tan(np.pi/3)+offset*np.sin(np.pi/30),
                 '%' + '$HCO_3^-$' + '+%' + '$CO_3^{2-}$',  
                 ha='center', va='center', rotation=60, fontsize=12)
        
        plt.text(0.5+offset+0.5*offset+offset*np.cos(np.pi/30),
                 h+offset*np.tan(np.pi/3)+0.25*np.tan(np.pi/3)+\
                     offset*np.sin(np.pi/30), '%' + '$SO_4^{2-}$' +\
                         '+%' + '$Cl^-$',  
                  ha='center', va='center', rotation=60, fontsize=12)
        plt.text(1.5+offset-0.25+offset*np.cos(np.pi/30),
                 h+offset*np.tan(np.pi/3)+0.25*np.tan(np.pi/3)+\
                     offset*np.sin(np.pi/30), '%' + '$Ca^{2+}$' +\
                         '+%' + '$Mg^{2+}$', 
                  ha='center', va='center', rotation=-60, fontsize=12)
        
        # Fill the water types domain
        ## the left traingle
        plt.fill([0.25, 0.5, 0.75, 0.25], 
                 [h/2, 0, h/2, h/2], color = (0.8, 0.8, 0.8), zorder=0)
        ## the right traingle
        plt.fill([1+2*offset+0.25, 1+2*offset+0.5, 1+2*offset+0.75, 1+2*\
                  offset+0.25], 
                 [h/2, 0, h/2, h/2], color = (0.8, 0.8, 0.8), zorder=0)
        ## the diamond
        plt.fill([0.5+offset+0.25, 0.5+offset+0.25+0.5, 0.5+offset+0.25+0.25,
                  0.5+offset+0.25],
                 [h+offset*np.tan(np.pi/3) - 0.5*np.sin(np.pi/3),
                  h+offset*np.tan(np.pi/3) - 0.5*np.sin(np.pi/3),
                  h+offset*np.tan(np.pi/3),
                  h+offset*np.tan(np.pi/3) - 0.5*np.sin(np.pi/3)], 
                 color = (0.8, 0.8, 0.8), zorder=0)
        plt.fill([0.5+offset+0.25, 0.5+offset+0.25+0.25,
                  0.5+offset+0.25+0.5, 0.5+offset+0.25],
                 [h+offset*np.tan(np.pi/3) + 0.5*np.sin(np.pi/3),
                  h+offset*np.tan(np.pi/3), 
                  h+offset*np.tan(np.pi/3) + 0.5*np.sin(np.pi/3),
                  h+offset*np.tan(np.pi/3) + 0.5*np.sin(np.pi/3)], 
                 color = (0.8, 0.8, 0.8), zorder=0)
        
        # Calculate the percentages
        sumcat = np.sum(meqL[:, 0:4], axis=1)
        suman = np.sum(meqL[:, 4:8], axis=1)
        cat = np.zeros((meqL.shape[0], 3))
        an = np.zeros((meqL.shape[0], 3))
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
        for i, row in self.df.iterrows():
            if (row.Label in Labels or row.Label == ''):
                TmpLabel = ''
            else:
                TmpLabel = row.Label
                Labels.append(TmpLabel)
             
            if (self.df['Color'].dtype is np.dtype('float')) or \
                (self.df['Color'].dtype is np.dtype('int64')):
                vmin = np.min(self.df['Color'].values)
                vmax = np.max(self.df['Color'].values)
                cf = plt.scatter(cat_x[i], cat_y[i], 
                                 marker=row.Marker,
                                 s=row.Size, 
                                 c=row.Color, vmin=vmin, vmax=vmax,
                                 alpha=row.Alpha,
                                 edgecolors='black')
                plt.scatter(an_x[i], an_y[i], 
                            marker=row.Marker,
                            s=row.Size, 
                            c=row.Color, vmin=vmin, vmax=vmax,
                            alpha=row.Alpha,
                            label=TmpLabel, 
                            edgecolors='black')
                plt.scatter(d_x[i], d_y[i], 
                            marker=row.Marker,
                            s=row.Size, 
                            c=row.Color, vmin=vmin, vmax=vmax,
                            alpha=row.Alpha,
                            edgecolors='black')
            else:
                plt.scatter(cat_x[i], cat_y[i], 
                            marker=row.Marker,
                            s=row.Size, 
                            c=row.Color, 
                            alpha=row.Alpha,
                            edgecolors='black')
                plt.scatter(an_x[i], an_y[i], 
                            marker=row.Marker,
                            s=row.Size, 
                            c=row.Color, 
                            alpha=row.Alpha,
                            label=TmpLabel, 
                            edgecolors='black')
                plt.scatter(d_x[i], d_y[i], 
                            marker=row.Marker,
                            s=row.Size, 
                            c=row.Color, 
                            alpha=row.Alpha,
                            edgecolors='black')
                    
        # Create the legend
        if (self.df['Color'].dtype is np.dtype('float')) or \
            (self.df['Color'].dtype is np.dtype('int64')):
            cb = plt.colorbar(cf, extend='both', spacing='uniform',
                              orientation='vertical', fraction=0.025, pad=0.05)
            cb.ax.set_ylabel('$TDS$' + ' ' + '$(mg/L)$', rotation=90,
                             labelpad=-75, fontsize=14)
        
        plt.legend(bbox_to_anchor=(0.15, 0.875), markerscale=1, fontsize=12,
                   frameon=False, 
                   labelspacing=0.25, handletextpad=0.25)
        
        # Save the figure
        plt.savefig(figname, bbox_inches='tight', dpi=self.dpi)
        
        # Display the info
        print("Piper plot created")    
        
        return


    def plot_Piper2(self, figname:str) -> None:
        """
        Builds a file with the Piper chart, considering NO3 if present

        Parameters
        ----------
        figname : Name and extension of the output file (path included)
        """
        # data
        an = self.anions_meqL_normalization()
        cat = self.cations_meqL_normalization()
        an_col_names = an.columns
            
        # Global plot settings
        # --------------------------------------------------------------------
        # Change default settings for figures
        plt.style.use('default')
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.titlesize'] = 10
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 10   
        
        # Plot background settings
        # --------------------------------------------------------------------
        # Define the offset between the diamond and triangle
        offset = 0.10                         
        # offsety = offset * np.tan(np.pi / 3.0)
        h = 0.5 * np.tan(np.pi / 3.0)
        
        # Calculate the triangles' location 
        ltriangle_x = np.array([0, 0.5, 1, 0])
        ltriangle_y = np.array([0, h, 0, 0])
        rtriangle_x = ltriangle_x + 2 * offset + 1
        rtriangle_y = ltriangle_y
        
        # Calculate the diamond's location 
        diamond_x = np.array([0.5, 1, 1.5, 1, 0.5]) + offset
        diamond_y = h * (np.array([1, 2, 1, 0, 1])) + \
            (offset * np.tan(np.pi / 3))
        
        # Plot the triangles and diamond
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
                ax.text((1-x)/2.0+x + 0.026, 
                        (1-x)/2.0*np.tan(np.pi/3) + 0.015, 
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
                ax.text((1-x)/2.0+x+1+2*offset  + 0.026,
                        (1-x)/2.0*np.tan(np.pi/3) + 0.015, 
                        ticklabels[-i-1], ha='center', va='center',
                        rotation=-60)
            ax.plot([x/2+1+2*offset, 1-x/2+1+2*offset], 
                    [x/2*np.tan(np.pi/3), x/2*np.tan(np.pi/3)], 'k:', lw=1.0)
            ## the left ticks
            if i in [1, 2, 3, 4]:
                ax.text(x/2+1+2*offset - 0.026, x/2*np.tan(np.pi/3) + 0.015, 
                        ticklabels[-i-1], ha='center', va='center',
                        rotation=60)
            
            # the diamond
            ax.plot([0.5 + offset + 0.5 / (1/interval) * x/interval,
                     1 + offset + 0.5 / (1/interval) * x / interval], 
                     [h + offset * np.tan(np.pi/3) + 0.5 / (1/interval) *\
                      x / interval * np.tan(np.pi / 3), 
                      offset * np.tan(np.pi / 3) + 0.5 / (1/interval) *\
                          x/interval*np.tan(np.pi/3)], 'k:', lw=1.0)
            ## the upper left and lower right
            if i in [1, 2, 3, 4]: 
                ax.text(0.5+offset+0.5/(1/interval)*x/interval - 0.026, 
                        h+offset*np.tan(np.pi/3)+0.5/(1/interval)*x\
                            /interval*np.tan(np.pi/3) + 0.015, ticklabels[i], 
                        ha='center', va='center', rotation=60)
                ax.text(1+offset+0.5/(1/interval)*x/interval + 0.026,
                        offset*np.tan(np.pi/3)+0.5/(1/interval)*x \
                            /interval*np.tan(np.pi/3) - 0.015,
                            ticklabels[-i-1], 
                        ha='center', va='center', rotation=60)
            ax.plot([0.5+offset+0.5/(1/interval)*x/interval,
                     1+offset+0.5/(1/interval)*x/interval], 
                     [h+offset*np.tan(np.pi/3)-0.5/(1/interval)*x\
                      /interval*np.tan(np.pi/3),
                      2*h+offset*np.tan(np.pi/3)-0.5/(1/interval)*\
                          x/interval*np.tan(np.pi/3)],'k:', lw=1.0)
            ## the lower left and upper right
            if i in [1, 2, 3, 4]:  
                ax.text(0.5+offset+0.5/(1/interval)*x/interval- 0.026,
                        h+offset*np.tan(np.pi/3)-0.5/(1/interval)*x\
                            /interval*np.tan(np.pi/3) - 0.015, ticklabels[i], 
                        ha='center', va='center', rotation=-60)
                ax.text(1+offset+0.5/(1/interval)*x/interval + 0.026,
                        2*h+offset*np.tan(np.pi/3)-0.5/(1/interval)*\
                            x/interval*np.tan(np.pi/3) + 0.015,
                            ticklabels[-i-1], ha='center', va='center',
                            rotation=-60)
        
        # Labels and title
        plt.text(0.5, -offset, '%' + '$Ca^{2+}$', 
                 ha='center', va='center', fontsize=12)
        plt.text(1+2*offset+0.5, -offset, self._cloride_piper_axis(an_col_names), 
                ha='center', va='center', fontsize=12)
        plt.text(0.25-offset*np.cos(np.pi/30), 0.25*np.tan(np.pi/3)+\
                 offset*np.sin(np.pi/30), '%' + '$Mg^{2+}$',  
                 ha='center', va='center', rotation=60, fontsize=12)
        plt.text(1.75+2*offset+offset*np.cos(np.pi/30), 0.25*np.tan(np.pi/3)+\
                 offset*np.sin(np.pi/30), '%' + '$SO_4^{2-}$',  
                  ha='center', va='center', rotation=-60, fontsize=12)
        plt.text(0.75+offset*np.cos(np.pi/30), 0.25*np.tan(np.pi/3)+offset*\
                 np.sin(np.pi/30), '%' + '$Na^+$' + '+%' + '$K^+$',  
                  ha='center', va='center', rotation=-60, fontsize=12)
        plt.text(1+2*offset+0.25-offset*np.cos(np.pi/30),
                 0.25*np.tan(np.pi/3)+offset*np.sin(np.pi/30),
                 self._carbonate_piper_axis(an_col_names),  
                 ha='center', va='center', rotation=60, fontsize=12)
        
        plt.text(0.5+offset+0.5*offset+offset*np.cos(np.pi/30),
                 h+offset*np.tan(np.pi/3)+0.25*np.tan(np.pi/3)+\
                     offset*np.sin(np.pi/30), '%' + '$SO_4^{2-}$' +\
                         self._cloride_piper_axis(an_col_names),  
                  ha='center', va='center', rotation=60, fontsize=12)
        plt.text(1.5+offset-0.25+offset*np.cos(np.pi/30),
                 h+offset*np.tan(np.pi/3)+0.25*np.tan(np.pi/3)+\
                     offset*np.sin(np.pi/30), '%' + '$Ca^{2+}$' +\
                         '+%' + '$Mg^{2+}$', 
                  ha='center', va='center', rotation=-60, fontsize=12)
        
        # Fill the water types domain
        ## the left traingle
        plt.fill([0.25, 0.5, 0.75, 0.25], 
                 [h/2, 0, h/2, h/2], color = (0.8, 0.8, 0.8), zorder=0)
        ## the right traingle
        plt.fill([1+2*offset+0.25, 1+2*offset+0.5, 1+2*offset+0.75, 1+2*\
                  offset+0.25], 
                 [h/2, 0, h/2, h/2], color = (0.8, 0.8, 0.8), zorder=0)
        ## the diamond
        plt.fill([0.5+offset+0.25, 0.5+offset+0.25+0.5, 0.5+offset+0.25+0.25,
                  0.5+offset+0.25],
                 [h+offset*np.tan(np.pi/3) - 0.5*np.sin(np.pi/3),
                  h+offset*np.tan(np.pi/3) - 0.5*np.sin(np.pi/3),
                  h+offset*np.tan(np.pi/3),
                  h+offset*np.tan(np.pi/3) - 0.5*np.sin(np.pi/3)], 
                 color = (0.8, 0.8, 0.8), zorder=0)
        plt.fill([0.5+offset+0.25, 0.5+offset+0.25+0.25,
                  0.5+offset+0.25+0.5, 0.5+offset+0.25],
                 [h+offset*np.tan(np.pi/3) + 0.5*np.sin(np.pi/3),
                  h+offset*np.tan(np.pi/3), 
                  h+offset*np.tan(np.pi/3) + 0.5*np.sin(np.pi/3),
                  h+offset*np.tan(np.pi/3) + 0.5*np.sin(np.pi/3)], 
                 color = (0.8, 0.8, 0.8), zorder=0)
        
        # Calculate the percentages

        an = self.anions_meqL_normalization()
        cat = self.cations_meqL_normalization()

        an_col_names = an.columns
        carbonate_col_name = self._carbonate_col_name(an_col_names)
        cloride_col_name = self._cloride_col_name(an_col_names)

        # Convert into cartesian coordinates
        cat_x = 0.5 * (2 * cat['Na_K'] + cat['Mg'])
        cat_y = h * cat['Mg']
        an_x = 1 + 2 * offset + 0.5 * (2 * an[cloride_col_name] + an['SO4'])
        an_y = h * an['SO4']
        d_x = an_y / (4 * h) + 0.5 * an_x - cat_y / (4 * h) + 0.5 * cat_x
        d_y = 0.5 * an_y + h * an_x + 0.5 * cat_y - h * cat_x

        # cat[:, 0] = meqL[:, 0] / sumcat                  # Ca
        # cat[:, 1] = meqL[:, 1] / sumcat                  # Mg
        # cat[:, 2] = (meqL[:, 2] + meqL[:, 3]) / sumcat   # Na+K
        # an[:, 0] = (meqL[:, 4] + meqL[:, 5]) / suman     # HCO3 + CO3
        # an[:, 2] = meqL[:, 6] / suman                    # Cl
        # an[:, 1] = meqL[:, 7] / suman                    # SO4
    
        # Convert into cartesian coordinates
        # cat_x = 0.5 * (2 * cat[:, 2] + cat[:, 1])
        # cat_y = h * cat[:, 1]
        # an_x = 1 + 2 * offset + 0.5 * (2 * an[:, 2] + an[:, 1])
        # an_y = h * an[:, 1]
        # d_x = an_y / (4 * h) + 0.5 * an_x - cat_y / (4 * h) + 0.5 * cat_x
        # d_y = 0.5 * an_y + h * an_x + 0.5 * cat_y - h * cat_x
    
        # Plot the scatters
        Labels = []
        for i, row in self.df.iterrows():
            if (row.Label in Labels or row.Label == ''):
                TmpLabel = ''
            else:
                TmpLabel = row.Label
                Labels.append(TmpLabel)
             
                if (self.df['Color'].dtype is np.dtype('float')) or \
                    (self.df['Color'].dtype is np.dtype('int64')):
                    vmin = np.min(self.df['Color'].values)
                    vmax = np.max(self.df['Color'].values)
                    cf = plt.scatter(cat_x[i], cat_y[i], 
                                     marker=row.Marker,
                                     s=row.Size, 
                                     c=row.Color, vmin=vmin, vmax=vmax,
                                     alpha=row.Alpha,
                                     edgecolors='black')
                    plt.scatter(an_x[i], an_y[i], 
                                marker=row.Marker,
                                s=row.Size, 
                                c=row.Color, vmin=vmin, vmax=vmax,
                                alpha=row.Alpha,
                                label=TmpLabel, 
                                edgecolors='black')
                    plt.scatter(d_x[i], d_y[i], 
                                marker=row.Marker,
                                s=row.Size, 
                                c=row.Color, vmin=vmin, vmax=vmax,
                                alpha=row.Alpha,
                                edgecolors='black')
                else:
                    plt.scatter(cat_x[i], cat_y[i], 
                                marker=row.Marker,
                                s=row.Size, 
                                c=row.Color, 
                                alpha=row.Alpha,
                                edgecolors='black')
                    plt.scatter(an_x[i], an_y[i], 
                                marker=row.Marker,
                                s=row.Size, 
                                c=row.Color, 
                                alpha=row.Alpha,
                                label=TmpLabel, 
                                edgecolors='black')
                    plt.scatter(d_x[i], d_y[i], 
                                marker=row.Marker,
                                s=row.Size, 
                                c=row.Color, 
                                alpha=row.Alpha,
                                edgecolors='black')

        # Create the legend
        if (self.df['Color'].dtype is np.dtype('float')) or \
            (self.df['Color'].dtype is np.dtype('int64')):
            cb = plt.colorbar(cf, extend='both', spacing='uniform',
                              orientation='vertical', fraction=0.025, pad=0.05)
            cb.ax.set_ylabel('$TDS$' + ' ' + '$(mg/L)$', rotation=90,
                             labelpad=-75, fontsize=14)

        plt.legend(bbox_to_anchor=(0.15, 0.875), markerscale=1, fontsize=12,
                   frameon=False, 
                   labelspacing=0.25, handletextpad=0.25)

        # Save the figure
        plt.savefig(figname, bbox_inches='tight', dpi=self.dpi)
        
        # Display the info
        print("Piper plot created")    
        
        return


