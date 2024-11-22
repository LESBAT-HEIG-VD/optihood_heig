# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 10:26:36 2024

@author: stefano.pauletta
"""
import pandas as pd
import os
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
from optihood.energy_network import EnergyNetworkIndiv as EnergyNetwork
import optihood.plot_sankey as snk
import optihood.plot_functions as fnc


# plot sankey diagram
UseLabelDict = True     # a dictionary defining the labels to be used for different flows
figureFilePath = r"..\figures"
resultFilePath =r"..\results"
clN=48
cl_hh=True
numberOfBuildings=3
opt=1 #"costs"
opt_type='group'
# resultFileName = f"results_pareto_mrgON_IamLenz_10_BiomassOnly_cluster{clN}_hh_{cl_hh}" + str(numberOfBuildings) + '_' + str(opt) + '.xlsx' 

# resultFileName = f"results_pareto_mrgON_IamLenz_10_Wood_PV_ST_TES_cluster{clN}_hh_{cl_hh}" + str(numberOfBuildings) + '_' + str(opt) + '.xlsx' 
resultFileName = f"results_pareto_mrgON_EcoThierrens_real2_HP_cluster0_hh_True3_1.xlsx" 

if not os.path.exists(figureFilePath):
    os.makedirs(figureFilePath)

sankeyFileName = f"Sankey_EcoThierrens_{numberOfBuildings}_{opt}.html"

if not os.path.exists(figureFilePath):
    os.makedirs(figureFilePath)

snk.plot(os.path.join(resultFilePath, resultFileName), os.path.join(figureFilePath, sankeyFileName),
               numberOfBuildings, UseLabelDict, labels='default', optimType=opt_type,mergedLinks=True)
