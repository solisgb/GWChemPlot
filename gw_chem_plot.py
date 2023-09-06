# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:55:25 2023

@author: solis

fork of WQChartPy https://github.com/jyangfsu/WQChartPy

Inside the class GWChemPlot, functions are organized in the following sections:
    Section 1. Getters and setters
    Section 2. Functions to acces to class constants
    Section 3. Functions to manage data before instantiate the class
    Section 4. Function to set columns related with graphs symbols (Label,
               Color, Marker)
    Section 5. Class methods that don't do grahs
    Section 6. Class methods that do grahs or are used in them
    
"""
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.markers as mmarkers
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import traceback
from typing import Union
    
import littleLogging as myLogging


class GWChemPlot():
    """
    Class to create Piper, Stiff and Schoeller diagrams to be saved as 
        image files.
    """
    # anions ({key: [weight, charge]})
    _anions = {'Cl': (35.453, -1), 'SO4': (96.0626, -2), 'CO3': (60.0089, -2),
               'HCO3': (61.0168, -1), 'NO3': (62.0049, -1)}
       
    # cations
    _cations = {'Ca': (40.078, 2), 'Mg': (24.305, 2), 'K': (39.0983, 1),
                          'Na': (22.989769, 1)}
    _required_ion_names = ('HCO3', 'Cl', 'SO4', 'Na', 'K', 'Ca', 'Mg')
    _optional_ion_names = ('CO3', 'NO3')
    
    _required_analysis_name = ('Sample',)
    _required_graph_names =  ('Label', 'Color', 'Marker', 'Size', 'Alpha')
    _defaults_graph_params = {'size': 30, 'alpha': 0.6} 
    _default_color = 'blue'
    _default_marker_size = 20

    
    def __init__(self, df: pd.DataFrame, unit: str='mg/L', dpi: int=150):
        """
        Instanciate the class
        Parameters
        ----------
        df. Hydrochemical data to draw diagrams.
        unit. The unit used in df (mg/L or meq/L). 
        dpi. Dot per inch in graph output files
        
        Atributes
        ---------
        _data. Checked df
        _unit. Checked unit
        _dpi. Checked dpi 
        _piper. Matlib parameter in plot_Piper method 
        """
        if unit not in ['mg/L', 'meq/L']:
           raise ValueError('Units must be mg/L or meq/L')

        if not GWChemPlot.check_dataframe_column_types(df):
            raise ValueError('You must modify the input DataFrame')
            
        df = df.dropna(subset = GWChemPlot._required_ions_names)
        if len(df):
            raise ValueError('There are not valid data, review the data file')
        
        self._data: pd.DataFrame = df.copy()
        self._unit: str = unit
        self._dpi: int = dpi
        self._piper: dict = {'font.size': 9, 'axes.labelsize': 9,
                             'axes.labelweight': 'bold', 'axes.titlesize' : 9,
                             'xtick.labelsize': 9, 'ytick.labelsize': 9,
                             'legend.fontsize' : 9, 'figure.titlesize': 10}


    """ Section 1. Getters and setters """

    @property
    def df(self):
        return self._data    

    
    @property
    def data(self):
        return self._data


    @data.setter
    def data(self, new_df: pd.DataFrame) -> None:
        if not GWChemPlot.data_are_ok(new_df):
            raise ValueError('Correct the new data')
        new_df = new_df.dropna()
        if len(new_df):
            raise ValueError('There are any data')
        self._data = new_df.copy()


    @property
    def unit(self):
        return self._unit
    
    
    @property
    def dpi(self):
        return self._dpi


    """ Section 2. Functions to acces to class constants"""

    @staticmethod
    def anions_names_get() -> [str]:
        return GWChemPlot._anions.keys()
    
    @staticmethod
    def cations_names_get() -> [str]:
        return GWChemPlot._cations.keys()

    @staticmethod
    def ions_names_get() -> [str]:
        return GWChemPlot._anions.keys() + GWChemPlot._cations.keys()
    

    @staticmethod
    def analysis_name():
        return GWChemPlot._required_analysis_name

    
    @staticmethod
    def required_ion_names():
        return GWChemPlot._required_ion_names


    @staticmethod
    def optional_ion_names():
        return GWChemPlot._optional_ion_names


    @staticmethod
    def required_graph_names():
        return GWChemPlot._required_graph_names

    
    """ Section 3. Functions to manage data before instantiate the class"""
    
    @staticmethod
    def check_dataframe_column_types(df: pd.DataFrame) -> bool:
        """
        Checks:
            1. df has the required column names
            2. The required columns have the required types
            3. If optional columns are presente they have the required types
        Returns
        ------
        True if df pass all the tests correctly
        """
        def match_types(df: pd.DataFrame, col_names: [str],
                        dtypes: [str]) -> bool:
            """
            For each element in col_names, checks that is of type dtypes; if
            not, try to change its type to dtypes

            Parameters
            ----------
            df. Data
            col_names. Column names 
            dtypes : str or [str]; if is of type str, a [str] is formed
                with all elements equal to dtypes.
            Returns
            -------
            True if OK
            """
            
            if isinstance(dtypes, str):
                dtypes = [dtypes for i in range(0, len(col_names))]
                
            columns_in_df = df.columns.tolist()
            result = True
            for col1, dtype in zip(col_names, dtypes):
                if col1 in columns_in_df:
                    if df[col1].dtype != dtype:
                        try:
                            df[col1].astype(dtype)
                        except ValueError as e:
                            myLogging.append(f'Column {col1}: {e.args[0]}')
                            if result:
                                result = False

            return result
        
        result = True
        if 'Sample' not in df.columns:
            GWChemPlot.create_column_sample(df)
        else:
            if df['Sample'].nunique() != len(df):
                myLogging.append('There are repeated values in the column',
                                 ' Sample')
                result = False
                
        if GWChemPlot.columns_not_in_df(df, GWChemPlot._required_graph_names):
            if result: result = False

        if not match_types(df, ['Sample', 'Size', 'Alpha'],
                           ['string', 'UInt32', 'Float32']):
            if result: result = False

        if GWChemPlot.columns_not_in_df(df, GWChemPlot._required_ion_names):
            if result: result = False

        if not match_types(df, GWChemPlot._required_ion_names, 'Float64'):
            if result: result = False
        
        opt_col = [c1 for c1 in GWChemPlot._optional_ion_names]
        if not match_types(df, opt_col, 'Float64'):
            if result: result = False
        
        return result    


    @staticmethod
    def anions_in_df(df: pd.DataFrame) -> [str]:
        cols = [c1 for c1 in df.columns if c1 in GWChemPlot._anions.keys()]
        return cols


    @staticmethod
    def cations_in_df(df: pd.DataFrame) -> [str]:
        cols = [c1 for c1 in df.columns if c1 in GWChemPlot._cations.keys()]
        return cols


    @staticmethod
    def ions_in_df(df: pd.DataFrame) -> [str]:
        c = GWChemPlot.cations_in_df()
        a = GWChemPlot.anions_in_df()
        return c + a


    @staticmethod
    def charge_get(ion_names: [str]=[]) -> pd.DataFrame:
        ions = {**GWChemPlot._anions, **GWChemPlot._cations}  # Merge dictionaries
        ich = {key: value[0] for key, value in ions.items()} 
        df = pd.DataFrame([ich], columns=ich.keys())        
        if ion_names:
            return df[ion_names]
        else:
            return df


    @staticmethod
    def charge_weight_ratio_get(ion_names: [str]=[]) -> pd.DataFrame:
        iw = GWChemPlot.weight_get(ion_names)
        ich = GWChemPlot.charge_get(ion_names)
        return np.abs(ich.div(iw.iloc[0]))


    @staticmethod
    def columns_exists(df: pd.DataFrame, col_names:[str]) -> bool:
        absent = [c1 for c1 in col_names if c1 not in df.columns]
        if len(absent) > 0:
            a = ','.join(absent)
            msg = f'df has not the columns {a}'
            myLogging.append(msg)
            return False
        return True


    @staticmethod 
    def columns_not_in_df(df: pd.DataFrame, col_names: [str]) -> [str]:
        missing = [c1 for c1 in col_names if c1 not in df.columns]
        if missing:
            a = ','.join(missing)
            msg = 'The data file does not contain the following required' +\
                f' columns: {a}'
            myLogging.append(msg)
        return missing


    @staticmethod
    def unique_rows(df: pd.DataFrame, col_names: [str]):
        if not GWChemPlot.columns_exists(df, col_names):
            return []
        return df[col_names].drop_duplicates()

    
    @staticmethod
    def weight_get(ion_names: [str]=[]) -> pd.DataFrame:
        ions = {**GWChemPlot._anions, **GWChemPlot._cations}  # Merge dictionaries
        iw = {key: value[0] for key, value in ions.items()} 
        df = pd.DataFrame([iw], columns=iw.keys())        
        if ion_names:
            return df[ion_names]
        else:
            return df


    """
    Section 4. Function to set columns related with graphs symbols
    
    Section 4.0. Sample
    """
    @staticmethod
    def create_column_sample\
        (df: pd.DataFrame, first_id: int = 1, suffix: str = 'S',
         separator: str= '-', insert_at_beginning:bool=True) -> bool:
        if 'Sample' not in df.columns:
            sequence = [suffix + separator + str(i) for i in \
                        range(first_id, len(df) + first_id)]
            if insert_at_beginning:
                df.insert(0, 'Sample', sequence)
            else:
                df['Sample'] = sequence
            myLogging.append('Column Sample has been automatically created')
            return True
        else:
            return False

    """ Section 4.1. Labels """

    @staticmethod
    def get_labels(df: pd.DataFrame):
        if 'Label' not in df.columns:
            return []
        else:
            return df['Label'].drop_duplicates()


    @staticmethod
    def set_labels(df: pd.DataFrame, autonumbering: bool, first_id: int = 1,
                   suffix: str = '', cols_for_label: [str] = [],
                   separator: str= '-') -> bool:
        """
        By employing this method, you can dynamically set the 'Label' column 
        with unique or not unique values for drawing samples in the graphs.
        
        It sets the column 'Label' based on specific conditions. The behavior
        depends on the value of the 'unique_row' parameter.
        When 'unique_row' is set to True, the 'Label' column is assigned using
        the suffix parameter and a number begining by first_id parameter
        On the other hand, when 'unique_row' is set to False, the 'Label'
        column is determined by concatenating the values from one or 
        multiple columns. These columns are specified in the 'cols_for_label'
        list.

        Parameters
        ----------
        df : data read from data file
        unique_row_id. If True, each row will have a unique Label identifier;
            else multiple samples could have the same Label 
        autonumbering. When is True, an automatic number for each
            Sample is assigned begining by first_number
        suffix. When autonumbering is True, a suffix can be added
        cols_for_label. List of columns in df
        separator. A separator in joined columns in cols_for_label
        """
        def autonumbering_set(df: pd.DataFrame, first_id: int, suffix: str):
            series = pd.Series(range(first_id, first_id+len(df)))
            series = series.astype(str)
            if len(suffix) > 0:
                series = suffix + series
            df['Label'] = series                

        if 'Sample' not in df.columns:
            GWChemPlot.create_column_sample(df)        
        
        if not autonumbering and not cols_for_label:
            autonumbering = True
            myLogging.append('autonumbering has been set to True')
        
        if autonumbering:
            autonumbering_set(df, first_id, suffix)
            return True

        present_cols = [col1 for col1 in cols_for_label if col1 in df.columns]
        if len(present_cols) != len(cols_for_label):
            invalid_cols = [col1 for col1 in cols_for_label \
                            if col1 not in df.columns]
            invalid_cols = ', '.join(invalid_cols)
            myLogging.append('The following columns do not exists: '
                             f'{invalid_cols}.\nLabels has been set using' 
                             ' autonumbering')
            autonumbering_set(df, 1, '')
            return True

        df['Label'] = suffix + df[cols_for_label[0]].astype(str)
        for col1 in cols_for_label[1:]:
            df['Label'] += separator + df[col1].astype(str)
                
        n = df['Label'].drop_duplicates().count()
        myLogging.append(f'{n:d} labels has been assigned')            
        

    """ Section 4.2. Column Color """

        
    @staticmethod
    def color_table_exists(color_table: str) -> bool:
        if color_table not in ('base', 'css4', 'tableau'):
            myLogging.append("color_table not in ('base', 'css4', 'tableau')"+\
                             ", its value is changed to 'tableau'")
            return False
        return True

        
    @staticmethod
    def color_names_get(color_table: str='tableau', hex_code: bool=False) -> []:
        if not GWChemPlot.color_table_exists(color_table):
            color_table = 'tableau'

        if color_table.lower() == 'base':
            colors = mcolors.BASE_COLORS
        elif color_table.lower() == 'css4':
            colors = mcolors.CSS4_COLORS
        elif color_table.lower() == 'tableau':
            colors = mcolors.TABLEAU_COLORS
        else:
            colors = mcolors.BASE_COLORS

        if hex_code:
            if color_table.lower() == 'base':
                return [(k, mcolors.to_hex(v)) for k, v in colors.items()]
            else:
                return [(k, v) for k, v in colors.items()]
        else:
            return [k for k in colors]


    @staticmethod
    def color_labels_set(df: pd.DataFrame, method:str = 'default', 
                         color_table:str='tableau', 
                         colormap:str='brg', 
                         single_color: bool=False, 
                         ) -> bool:
        """
        Sets th column Color if column Label exists in df
        Parameters
        ----------
        df : data read from data file
        method: Categorical. How colors are assigned.
            1.  'default': if nlabels <= colors in color table, colors in color
                table are used: else nlabels colors in colormap
            2.  'color_table' 
            3.  'colormap'
        color_table. Mathplotlib named colors list: 'tableau' (recommended),
            'css4' or 'base'.
        colormap. Mathplotlib colormap name
        single_color. You can specify a single color for all the labels. So
            labels remain color indistinguishable
        """
        def get_colors_in_colormap(cmap_name:str, ncolors:int):
           cmap = plt.get_cmap(cmap_name)
           values = np.linspace(0, 1, ncolors)
           colors = [cmap(value) for value in values]
           hex_colors = [mcolors.to_hex(color, keep_alpha=True) \
                         for color in colors]
           return hex_colors

               
        VALID_METHODS = ('default', 'color_table', 'colormap')
        
        if 'Label' not in df.columns:
            myLogging.append('Label column must exists to set colors')
            return False

        if method not in VALID_METHODS:
            vm = ', '.join(VALID_METHODS)
            myLogging.append(f'Method {method} must be one between: {vm}. ', 
                             'Default has been assigned')
            method = 'default'
        
        if not GWChemPlot.color_table_exists(color_table):
            color_table = 'tableau'
        
        colors_t = GWChemPlot.color_names_get(color_table)
        n_colors_t = len(colors_t)
        if single_color:
            df['Color'] = colors_t[0]    
            return True

        cmaps_names = [cmn1 for cmn1 in plt.colormaps()]
        cmaps_names.sort()
        if colormap not in cmaps_names:
            colormap = 'brg'        
        listed_colormap = plt.get_cmap(colormap)
        n_colors_cm = listed_colormap.N

        labels =  df['Label'].unique()
        nlabels = len(labels)

        if nlabels <= n_colors_t:
            if method in ('default', 'color_table'):
                colors_to_apply = colors_t
            else:
                colors_to_apply = get_colors_in_colormap(colormap, nlabels)
        else:
            colors_to_apply = get_colors_in_colormap(colormap, nlabels)
            
        for i, lab in enumerate(labels):
            df.loc[df['Label'] == lab, 'Color'] = colors_to_apply[i]            

        return True

        
    @staticmethod
    def color_labels_set0(df: pd.DataFrame, color_table: str='tableau', 
                         single_color: bool=False, 
                         cycle_colors: bool=False,
                         my_colors: [str]=[]) -> bool:
        """
        Sets th column Color if column Label exists in df
        Parameters
        ----------
        df : data read from data file
        color_table. Mathplotlib table of colors
        single_color. If there ara a single label and single_color is True
            color is set to its default value
        cycle_colors. When df has more labels that colors in color_table
            if cycle_colors is True it cycles colors in color table, or
            generates extra colors if cycle_colors is  False
        my_colors. List of color's names
        """
        def fill_rgb_array(n:int):
            """
            Fills the whole array having each row a valid RGB color 
            representation
            """
            c = np.empty((n, 3))
            for i in range(n):
                c[:, i] = np.random.random(n)
            hex_code = [mcolors.to_hex(c[i,:]) for i in range(n)]
            return hex_code                           

        
        if 'Label' not in df.columns:
            myLogging.append('Label column must exists to set colors')
            return False
        
        if not GWChemPlot.color_table_exists(color_table):
            color_table = 'tableau'
        
        colors = GWChemPlot.color_names_get(color_table)
        if my_colors:
            valid_colors = [color1 for color1 in my_colors if color1 in colors]
        else:
            valid_colors = colors

        if single_color:
            if valid_colors:
                df['Color'] = valid_colors[0]    
            else:
                df['Color'] = GWChemPlot._default_color
            return True

        labels =  df['Label'].unique()
        nlabels = len(labels)
        
        ncolors = len(valid_colors)
        if nlabels <= ncolors:
            for i, lab in enumerate(labels):
                df.loc[df['Label'] == lab, 'Color'] = valid_colors[i]
        else:
            myLogging.append('Number of labes > number of colors')
            if cycle_colors:
                myLogging.append('There are labels with the same color')
                icolor = -1
                for lab in labels:
                    icolor += 1
                    if icolor == ncolors:
                        icolor = 0 
                    df.loc[df['Label'] == lab, 'Color'] = valid_colors[icolor]
            else:
                myLogging.append('Extra colors are generated randomly')
                for i in range(0, ncolors):
                    df.loc[df['Label'] == labels[i], 'Color'] = valid_colors[i]
                m = nlabels - ncolors
                c = fill_rgb_array(m)
                for j in range(m):
                    df.loc[df['Label'] == labels[i+j+1], 'Color'] = c[j]
        return True

    """ Section 4.3. Column Marker """

    @staticmethod
    def get_filled_markers(draw_markers:bool = False) -> [str]:
        """ Get filled markers and optionally draws them """

        def format_axes(ax):
            ax.margins(0.2)
            ax.set_axis_off()
            ax.invert_yaxis()
        
        
        def split_list(a_list):
            i_half = len(a_list) // 2
            return a_list[:i_half], a_list[i_half:]
        
        if draw_markers:
            text_style = dict(horizontalalignment='right', 
                              verticalalignment='center',
                              fontsize=12, fontfamily='monospace')
            marker_style = dict(linestyle=':', color='0.8', markersize=10,
                                markerfacecolor="tab:blue", 
                                markeredgecolor="tab:blue")
            
            fig, axs = plt.subplots(ncols=2)
            fig.suptitle('Filled markers', fontsize=14)
            for ax, markers in zip(axs, split_list(Line2D.filled_markers)):
                for y, marker in enumerate(markers):
                    ax.text(-0.5, y, repr(marker), **text_style)
                    ax.plot([y] * 3, marker=marker, **marker_style)
                format_axes(ax) 
        
        return [m for m in mmarkers.MarkerStyle.filled_markers]

    
    @staticmethod
    def set_markers(df: pd.DataFrame, single_marker: bool=True, 
                    my_markers: [str] =[]) -> bool:
        """
        Sets column Marker in df
        df : data
        single_marker : all rows have same marker
        my_markers : list of markers
        """

        markers = GWChemPlot.get_filled_markers()        
        if my_markers:
            absent = [mk1 for mk1 in my_markers if mk1 not in markers]
            if len(absent) > 0:
                a = ', '.join(absent)
                myLogging.append(f'Markers {a} are not valid filled markers')
                present_markers = [mk1 for mk1 in my_markers if mk1 in markers]
                if len(present_markers) > 0:
                    markers = present_markers 
            else:
                markers = my_markers

        if single_marker:
            df['Marker'] = markers[0]
            return True

        if not GWChemPlot.columns_exists(df, ('Label', 'Color')):
            df['Marker'] = markers[0]
            return True
        
        unique_pairs = df[['Label', 'Color']].drop_duplicates()
        imarker = -1
        for index, row in unique_pairs.iterrows():
            imarker += 1
            if imarker == len(markers):
                imarker = 0  
                myLogging.append('Number of pairs Label, Color > number of' +\
                                 '  markers; markers are recycled')
            mask = (df['Label'] == row['Label']) & (df['Color'] == row['Color'])
            df.loc[mask, 'Marker'] = markers[imarker]

    
    @staticmethod
    def set_markers_size(df: pd.DataFrame, unique_marker_size: bool=True, 
                         marker_size: int = 20, 
                         my_markers_sizes: {str:int}= {}) -> None:
        
        if marker_size<=0 or marker_size>30:
            myLogging.append('Markers size has been changed')
            marker_size = GWChemPlot._default_marker_size
        
        if unique_marker_size:
            df['Size'] = marker_size

    """ Section 5. Class methods that don't do grahs """

    def cbe(self) -> pd.DataFrame:
        """
        Get the charge balance error (cbe). The returned DataFrame has
        the columns: 'Sample' and 'cbe'
        """
        anions = GWChemPlot.anions_in_df(self.df)
        cations = GWChemPlot.cations_in_df(self.df)
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


    def check_columns_are_present(self, col_names: [str]) -> bool:
        """
        Determines if the column names in cols are present in self.data.
        If any column is not present throws an exception; 
        otherwise returns True
        """
        return GWChemPlot.columns_exists(self.df, col_names)


    def meqL_get(self, ion_names: [str]=[]) -> pd.DataFrame:
        """
        Calculate the meq/L of the ions in ion_names.

        Parameters
        ----------
        col_names : list of ion's names 
        """
        if not ion_names:
            ion_names = GWChemPlot.ions_in_df(self.df)
  
        if self.unit == 'meq/L':
            return self.df[ion_names]
        else:
            x = GWChemPlot.charge_weight_ratio_get(ion_names)
            return self.df[ion_names].mul(x.iloc[0])


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

    """ Section 6. Class methods that do grahs or are used in them """

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
        col_names = GWChemPlot.anions_in_df(self.df)
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
        col_names = GWChemPlot.cations_in_df(self.df)
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
            return '$HCO_3^-$' + '+' + '$CO_3^{2-}$'
        else:
            return '$HCO_3^-$'       


    def _cloride_col_name(self, col_names: [str]) -> 'str':
        if 'Cl_NO3' in col_names:
            return 'Cl_NO3'
        else:
            return 'Cl'


    def _cloride_piper_axis(self, col_names: [str]) -> 'str':
        if 'Cl_NO3' in col_names:
            return '$Cl^{-}$' + '+' + '$NO_3^{-}$'
        else:
            return '$Cl^{-}$'


    def _Cl_label(self, col_names: [str]) -> 'str':
        if 'Cl_NO3' in col_names:
            return 'Cl+NO3'
        else:
            return 'Cl'
        
        
    def _Cl_label_sp(self, col_names: [str]) -> 'str':
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
        Cl_label = self._Cl_label(col_names)
        
        idc_cations = []
        idc_anions = []
        for i, r in idc.iterrows():        
            if r.Na_K > r.Mg > r.Ca:
                if r.Na_K >= 0.5:
                    idc_cations.append('Na (Mg-Ca)')
                else:
                    idc_cations.append('Na-Mg-Ca')
            elif r.Na_K > r.Ca > r.Mg:
                if r.Na >= 0.5:
                    idc_cations.append('Na (Mg-Ca)')
                else:
                    idc_cations.append('Na-Ca-Mg)')
            elif r.Mg > r.Na_K > r.Ca:
                if r.Mg >= 0.5:
                    idc_cations.append('Mg (Na-Ca)')
                else:
                    idc_cations.append('Mg-Na-Ca')               
            elif r.Mg > r.Ca > r.Na_K:
                if r.Mg >= 0.5:
                    idc_cations.append('Mg (Ca-Na)')
                else:
                    idc_cations.append('Mg-Ca-Na')                
            elif r.Ca > r.Na_K > r.Mg:
                if r.Ca >= 0.5:
                    idc_cations.append('Ca (Na-Mg)')
                else:
                    idc_cations.append('Ca-Na-Mg')                
            elif r.Ca > r.Mg > r.Na_K:
                if r.Ca >= 0.5:
                    idc_cations.append('Ca (Mg-Na)')
                else:
                    idc_cations.append('Ca-Mg-Na)')                

            if r[cloride_col_name] > r['SO4'] > r[carbonate_col_name]:
                if r[cloride_col_name] >= 0.5:
                    idc_anions.append(f'{Cl_label} (SO4-HCO3)')
                else:
                    idc_anions.append(f'{Cl_label}-SO4-HCO3')               
            elif r[cloride_col_name] > r[carbonate_col_name] > r['SO4']:
                if r[cloride_col_name] >= 0.5:
                    idc_anions.append(f'{Cl_label} (HCO3-SO4)')
                else:
                    idc_anions.append(f'{Cl_label}-HCO3-SO4')               
            elif r['SO4'] > r[cloride_col_name] > r[carbonate_col_name] :
                if r['SO4'] >= 0.5:
                    idc_anions.append(f'SO4 ({Cl_label}-HCO3)')
                else:
                    idc_anions.append(f'SO4-{Cl_label}-HCO3')               
            elif r['SO4'] > r[carbonate_col_name] > r[cloride_col_name]:
                if r['SO4'] >= 0.5:
                    idc_anions.append(f'SO4 (HCO3-{Cl_label})')
                else:
                    idc_anions.append(f'SO4-HCO3-{Cl_label}')
            elif r[carbonate_col_name] > r[cloride_col_name] > r['SO4']:
                if r[carbonate_col_name] >= 0.5:
                    idc_anions.append(f'HCO3 ({Cl_label}-SO4)')
                else:
                    idc_anions.append(f'HCO3-{Cl_label}-SO4)')            
            elif r[carbonate_col_name] > r['SO4'] > r[cloride_col_name]:
                if r[carbonate_col_name] >= 0.5:
                    idc_anions.append(f'HCO3 (SO4-{Cl_label})')
                else:
                    idc_anions.append(f'HCO3-(SO4-{Cl_label})')            

        idc['cations_classified'] = idc_cations
        idc['anions_classified'] = idc_anions
        return idc


    def ion_dominant_classification_sp(self) -> pd.DataFrame:
        """
        Groundwater ion dominant classification (Spanish labels)
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
        Cl_lab = self._Cl_label_sp(col_names)
        
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
                    idc_anions.append(f'{Cl_lab} (sulfatada-bicarbonatada)')
                else:
                    idc_anions.append(f'Mixta ({Cl_lab.lower()}-sulfatada-bicarbonatada)')               
            elif r[cloride_col_name] > r[carbonate_col_name] > r['SO4']:
                if r[cloride_col_name] >= 0.5:
                    idc_anions.append(f'{Cl_lab} (bicarbonatada-sulfatada)')
                else:
                    idc_anions.append(f'Mixta ({Cl_lab.lower()}-bicarbonatada-sulfatada)')               
            elif r['SO4'] > r[cloride_col_name] > r[carbonate_col_name] :
                if r['SO4'] >= 0.5:
                    idc_anions.append(f'Sulfatada ({Cl_lab.lower()}-bicarbonatada)')
                else:
                    idc_anions.append(f'Mixta (sulfatada-{Cl_lab.lower()}-bicarbonatada)')               
            elif r['SO4'] > r[carbonate_col_name] > r[cloride_col_name]:
                if r['SO4'] >= 0.5:
                    idc_anions.append(f'Sulfatada (bicarbonatada-{Cl_lab.lower()})')
                else:
                    idc_anions.append(f'Mixta (sulfatada-bicarbonatada-{Cl_lab.lower()})')
            elif r[carbonate_col_name] > r[cloride_col_name] > r['SO4']:
                if r[carbonate_col_name] >= 0.5:
                    idc_anions.append(f'Bicarbonatada ({Cl_lab.lower()}-sulfatada)')
                else:
                    idc_anions.append(f'Mixta (bicarbonatada-{Cl_lab.lower()}-sulfatada)')            
            elif r[carbonate_col_name] > r['SO4'] > r[cloride_col_name]:
                if r[carbonate_col_name] >= 0.5:
                    idc_anions.append(f'Bicarbonatada (sulfatada-{Cl_lab.lower()})')
                else:
                    idc_anions.append(f'Mixta (bicarbonatada-(sulfatada-{Cl_lab.lower()})')            

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
        for key, value in self._piper.items():
            plt.rcParams[key] = value
        
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
        
        # Labels
        plt.text(0.5, -offset, '$Ca^{2+}$', ha='center', va='center')
        plt.text(1+2*offset+0.5, -offset, 
                 self._cloride_piper_axis(an_col_names), ha='center',
                 va='center')
        plt.text(0.25-offset*np.cos(np.pi/30), 0.25*np.tan(np.pi/3)+\
                 offset*np.sin(np.pi/30), '$Mg^{2+}$',  
                 ha='center', va='center', rotation=60)
        plt.text(1.75+2*offset+offset*np.cos(np.pi/30), 0.25*np.tan(np.pi/3)+\
                 offset*np.sin(np.pi/30), '$SO_4^{2-}$',  
                  ha='center', va='center', rotation=-60)
        plt.text(0.75+offset*np.cos(np.pi/30), 0.25*np.tan(np.pi/3)+offset*\
                 np.sin(np.pi/30), '$Na^+$' + '+' + '$K^+$',  
                  ha='center', va='center', rotation=-60)
        plt.text(1+2*offset+0.25-offset*np.cos(np.pi/30),
                 0.25*np.tan(np.pi/3)+offset*np.sin(np.pi/30),
                 self._carbonate_piper_axis(an_col_names),  
                 ha='center', va='center', rotation=60)
        
        plt.text(0.5+offset+0.5*offset+offset*np.cos(np.pi/30),
                 h+offset*np.tan(np.pi/3)+0.25*np.tan(np.pi/3)+\
                     offset*np.sin(np.pi/30), '$SO_4^{2-}$' + '+' +\
                         self._cloride_piper_axis(an_col_names),  
                  ha='center', va='center', rotation=60)
        plt.text(1.5+offset-0.25+offset*np.cos(np.pi/30),
                 h+offset*np.tan(np.pi/3)+0.25*np.tan(np.pi/3)+\
                     offset*np.sin(np.pi/30), '$Ca^{2+}$' +\
                         '+' + '$Mg^{2+}$', 
                  ha='center', va='center', rotation=-60)
        
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
        Clcn = self._cloride_col_name(an_col_names)

        # Convert into cartesian coordinates
        cat_x = 0.5 * (2 * cat['Na_K'] + cat['Mg'])
        cat_y = h * cat['Mg']
        an_x = 1 + 2 * offset + 0.5 * (2 * an[Clcn] + an['SO4'])
        an_y = h * an['SO4']
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
             
            plt.scatter(cat_x[i], cat_y[i], marker=row.Marker, s=row.Size, 
                        c=row.Color, alpha=row.Alpha, edgecolors='black')
            plt.scatter(an_x[i], an_y[i], marker=row.Marker, s=row.Size, 
                        c=row.Color, alpha=row.Alpha, label=TmpLabel, 
                        edgecolors='black')
            plt.scatter(d_x[i], d_y[i], marker=row.Marker, s=row.Size, 
                        c=row.Color, alpha=row.Alpha, edgecolors='black')

        plt.legend(bbox_to_anchor=(0.1, 0.95), markerscale=1, frameon=False, 
                   labelspacing=0.25, handletextpad=0.25)

        # Save the figure
        plt.savefig(figname, bbox_inches='tight', dpi=self.dpi)
        
        # Display the info
        print("Piper plot saved")    
        
        return


