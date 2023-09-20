# Module gw_chem_plot
gw_chem_plot stands for Groundwater chemical plots

Python module to make some groundwater hydrochemical graphs. At the moment the graphs are: Schoeller, Piper and Stiff.

The functions that build the graphs have been adapted with slight modifications from the WQChartPy package https://github.com/jyangfsu/WQChartPy.

There are 2 methods to create the Piper diagram: Piper and Piper2. If the water analysis contains $NO_3^{-}$, the Piper2 method represents it added together with $Cl{-}$.

Two tables can be saved to files:

1. Charge balance of the hydrochemical analyses.

2. Classification of the analyses according to the most abundant ions.

Three Jupyter notebooks are included to teach users with little experience in Python or Jupyter notebooks how to handle the module.
