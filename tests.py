# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 13:29:21 2023

@author: solis
"""
import numpy as np
import pandas as pd

# Assuming df is your DataFrame

# Define a function to attempt conversion using astype and catch errors
def detect_string_conversion_errors(df, column_name, data_type):
    try:
        df[column_name] = df[column_name].astype(data_type)
        return False  # No error, conversion successful
    except (ValueError, TypeError):
        return True  # Error occurred during conversion

if __name__ == "__main__":
    
    df = pd.DataFrame(
        {'C0': ['a', 'b', 'c', 'd', 'e'],
         'C1': ['2003-01-03', '2003-01-05', '2003-01-05', '2003-01-02', '2003-01-02'],
         'C2': ['Acumulado', np.nan, 3, 4, 5],
         'C3': [5, 6, np.nan, 8, 9],
         'C4': [5, 6, 7, 8, 'hi']}
    )
    print('data')    
    print(df)
    print()

    # Columns to check for conversion errors
    columns_to_check = ['C2', 'C3', 'C4']
    
    mask = df[columns_to_check].applymap(lambda x: isinstance(x, str))
    
    # Find the columns with string values
    columns_with_str = df[columns_to_check].columns[mask.any()].tolist()
    
    # Find the rows with string values
    rows_with_str = df[mask.any(axis=1)]
    
    print("Columns with string values:", columns_with_str)
    print("Rows with string values:\n", rows_with_str)

    # Create a mask that checks each value in the columns in `columns_to_check` to see if it is NaN
    mask = df[columns_to_check].isna()
    
    # Find the columns with NaN values
    columns_with_nan = df[columns_to_check].columns[mask.any()].tolist()
    
    # Find the rows with NaN values
    rows_with_nan = df[mask.any(axis=1)]

    print("\nColumns with NaN values:", columns_with_nan)
    print("Rows with NaN values:\n", rows_with_nan)

    l1 = ['a', 'c', 'd', 'w', 'a']
    l2 = ['c', 'f', 'g', 'c']
    
    # Merge the lists and remove duplicates by converting to a set
    merged_list = list(set(l1 + l2))

print(merged_list)
