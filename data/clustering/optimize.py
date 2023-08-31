import numpy as np
import pandas as pd
import os
from datetime import datetime
import sys

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

## import energy network class
# EnergyNetworkIndiv for individual optimization
# EnergyNetworkGroup for grouped optimization

from optihood.energy_network import EnergyNetworkIndiv as EnergyNetwork

# import plotting methods for Sankey and detailed plots

import optihood.plot_sankey as snk
import optihood.plot_functions as fnc

if __name__ == '__main__':

    clustering = True
    if clustering:
        # In this example we consider 12 clusters
        # 12 representative days for the whole year
        # The number of days belonging to each cluster is defined in the dictionary 'cluster'
        cluster = {"2021-07-30": 26,
                   "2021-02-03": 44,
                   "2021-07-23": 32,
                   "2021-09-18": 28,
                   "2021-04-15": 22,
                   "2021-10-01": 32,
                   "2021-11-04": 32,
                   "2021-10-11": 37,
                   "2021-01-24": 15,
                   "2021-08-18": 26,
                   "2021-05-28": 23,
                   "2021-02-06": 48}
        cluster = {'2021-01-11': 1,
                 '2021-01-13': 8,
                 '2021-01-17': 5,
                 '2021-01-18': 11,
                 '2021-01-27': 3,
                 '2021-01-31': 4,
                 '2021-02-12': 1,
                 '2021-02-22': 5,
                 '2021-02-25': 6,
                 '2021-03-01': 4,
                 '2021-03-04': 5,
                 '2021-03-09': 4,
                 '2021-03-21': 3,
                 '2021-03-23': 4,
                 '2021-03-28': 4, #FIXME THIS FAKE NEWS WITH A HOUR CHANGE: IS 28-03-2021
                 '2021-04-13': 5,
                 '2021-04-19': 1,
                 '2021-04-20': 4,
                 '2021-04-21': 6,
                 '2021-05-04': 10,
                 '2021-05-08': 8,
                 '2021-05-12': 1,
                 '2021-05-20': 3,
                 '2021-05-25': 4,
                 '2021-05-26': 3,
                 '2021-06-05': 8,
                 '2021-06-14': 4,
                 '2021-06-16': 6,
                 '2021-06-18': 4,
                 '2021-06-24': 8,
                 '2021-06-27': 3,
                 '2021-07-07': 9,
                 '2021-07-18': 6,
                 '2021-07-22': 4,
                 '2021-07-24': 3,
                 '2021-07-28': 7,
                 '2021-08-04': 5,
                 '2021-08-15': 4,
                 '2021-08-24': 9,
                 '2021-09-01': 10,
                 '2021-09-05': 5,
                 '2021-09-08': 4,
                 '2021-09-15': 6,
                 '2021-09-17': 5,
                 '2021-09-20': 5,
                 '2021-09-26': 7,
                 '2021-09-28': 5,
                 '2021-09-30': 5,
                 '2021-10-03': 3,
                 '2021-10-09': 6,
                 '2021-10-10': 5,
                 '2021-10-13': 11,
                 '2021-10-21': 6,
                 '2021-10-23': 9,
                 '2021-10-29': 9,
                 '2021-11-07': 2,
                 '2021-11-16': 10,
                 '2021-11-22': 11,
                 '2021-11-23': 7,
                 '2021-11-24': 7,
                 '2021-12-04': 1,
                 '2021-12-09': 7,
                 '2021-12-12': 5,
                 '2021-12-15': 5,
                 '2021-12-18': 4,
                 '2021-12-21': 3,
                 '2021-12-25': 3,
                 '2021-12-26': 6}
        # cluster = {'2021-01-01': 12, # now using medoids
        #          '2021-02-05': 8,
        #          '2021-02-07': 9,
        #          '2021-02-14': 1,
        #          '2021-02-20': 6,
        #          '2021-02-22': 5,
        #          '2021-02-23': 6,
        #          '2021-02-25': 6,
        #          '2021-02-28': 3,
        #          '2021-03-02': 4,
        #          '2021-03-06': 2,
        #          '2021-03-09': 4,
        #          '2021-03-20': 2,
        #          '2021-03-23': 5,
        #          '2021-03-31': 4,
        #          '2021-04-03': 4,
        #          '2021-04-10': 2,
        #          '2021-04-11': 3,
        #          '2021-04-13': 7,
        #          '2021-04-21': 6,
        #          '2021-04-22': 5,
        #          '2021-04-25': 4,
        #          '2021-05-04': 4,
        #          '2021-05-05': 4,
        #          '2021-05-07': 5,
        #          '2021-05-09': 3,
        #          '2021-05-16': 6,
        #          '2021-05-23': 3,
        #          '2021-05-25': 3,
        #          '2021-05-26': 2,
        #          '2021-05-31': 3,
        #          '2021-06-20': 5,
        #          '2021-06-23': 4,
        #          '2021-06-26': 3,
        #          '2021-06-27': 3,
        #          '2021-06-28': 4,
        #          '2021-06-29': 7,
        #          '2021-07-04': 4,
        #          '2021-07-09': 3,
        #          '2021-07-12': 6,
        #          '2021-07-15': 5,
        #          '2021-07-18': 3,
        #          '2021-07-20': 9,
        #          '2021-07-23': 4,
        #          '2021-08-03': 5,
        #          '2021-08-07': 3,
        #          '2021-08-08': 3,
        #          '2021-08-10': 7,
        #          '2021-08-14': 3,
        #          '2021-08-20': 7,
        #          '2021-08-25': 6,
        #          '2021-09-02': 7,
        #          '2021-09-05': 4,
        #          '2021-09-15': 6,
        #          '2021-09-29': 5,
        #          '2021-10-01': 4,
        #          '2021-10-02': 3,
        #          '2021-10-05': 4,
        #          '2021-10-22': 11,
        #          '2021-11-01': 5,
        #          '2021-11-03': 13,
        #          '2021-11-05': 7,
        #          '2021-11-07': 3,
        #          '2021-11-09': 7,
        #          '2021-11-27': 6,
        #          '2021-12-02': 13,
        #          '2021-12-03': 10,
        #          '2021-12-11': 6,
        #          '2021-12-19': 7,
        #          '2021-12-22': 9}
        # # now using extreme days
        # cluster = {'2021-01-13': 8,
        #          '2021-01-27': 17,
        #          '2021-02-05': 10,
        #          '2021-02-20': 8,
        #          '2021-02-25': 9,
        #          '2021-03-02': 5,
        #          '2021-03-05': 6,
        #          '2021-03-07': 4,
        #          '2021-03-09': 3,
        #          '2021-03-17': 5,
        #          '2021-03-20': 1,
        #          '2021-03-23': 5,
        #          '2021-03-27': 3,
        #          '2021-03-31': 4,
        #          '2021-04-04': 3,
        #          '2021-04-13': 7,
        #          '2021-04-17': 1,
        #          '2021-04-18': 1,
        #          '2021-04-21': 6,
        #          '2021-04-22': 5,
        #          '2021-05-02': 2,
        #          '2021-05-04': 4,
        #          '2021-05-05': 4,
        #          '2021-05-07': 5,
        #          '2021-05-16': 6,
        #          '2021-05-23': 5,
        #          '2021-05-25': 3,
        #          '2021-05-30': 1,
        #          '2021-05-31': 3,
        #          '2021-06-06': 1,
        #          '2021-06-12': 1,
        #          '2021-06-23': 7,
        #          '2021-06-27': 2,
        #          '2021-06-29': 9,
        #          '2021-07-04': 4,
        #          '2021-07-09': 3,
        #          '2021-07-11': 7,
        #          '2021-07-12': 6,
        #          '2021-07-15': 5,
        #          '2021-07-20': 9,
        #          '2021-07-23': 4,
        #          '2021-07-24': 3,
        #          '2021-08-03': 5,
        #          '2021-08-07': 3,
        #          '2021-08-08': 4,
        #          '2021-08-10': 7,
        #          '2021-08-14': 4,
        #          '2021-08-20': 8,
        #          '2021-08-25': 6,
        #          '2021-09-02': 7,
        #          '2021-09-05': 5,
        #          '2021-09-15': 6,
        #          '2021-09-25': 4,
        #          '2021-09-29': 5,
        #          '2021-10-01': 5,
        #          '2021-10-05': 4,
        #          '2021-10-13': 12,
        #          '2021-11-01': 6,
        #          '2021-11-07': 3,
        #          '2021-11-11': 9,
        #          '2021-11-16': 8,
        #          '2021-12-01': 10,
        #          '2021-12-03': 11,
        #          '2021-12-11': 6,
        #          '2021-12-19': 9,
        #          '2021-12-25': 13}
        cul = pd.date_range(start="2021-01-01", freq="D", periods=365)
        cluster = {str(c).replace(' 00:00:00', ''): 1 for c in cul}
        for i in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
            del cluster[f'2021-{i}-01']
            cluster[f'2021-{i}-02'] = 2
            #del cluster[f'2021-{i}-05']
            #cluster[f'2021-{i}-06'] = 2
            del cluster[f'2021-{i}-11']
            cluster[f'2021-{i}-12'] = 2
            #del cluster[f'2021-{i}-15']
            #cluster[f'2021-{i}-16'] = 2
            del cluster[f'2021-{i}-21']
            cluster[f'2021-{i}-22'] = 2
            #del cluster[f'2021-{i}-25']
            #cluster[f'2021-{i}-26'] = 2
        print('cluster', cluster)

        print(np.sum(np.array(list(cluster.values())).astype(int)))
        print('nb clusters', len(cluster))
        # set a time period for the optimization problem according to the size of clusers
        #timePeriod = pd.date_range("2021-01-01 00:00:00", "2021-01-12 23:00:00", freq="60min")
        timePeriod = pd.date_range(start="2021-01-01 00:00:00", freq="60min", periods=24*len(cluster)) #wtf
        print('timePeriod', timePeriod)
    else:
        cluster = {}
        # set a time period for the optimization problem
        timePeriod = pd.date_range("2021-01-01 00:00:00", "2021-12-31 23:00:00", freq="60min")

    # define paths for input and result files
    inputFilePath = './'
    inputfileName = "./scenario_Annual_1_costs_0_SH35.xls"

    resultFilePath = "../results"
    resultFileName = "results2_yes_mine2.xlsx" if clustering else "results2_no.xlsx"

    # initialize parameters
    numberOfBuildings = 1
    optimizationType = "costs"  # set as "env" for environmental optimization
    mergeLinkBuses = False
    dispatchMode = False
    scenario = "Annual_{}_{}_E-3_2_SH60".format(numberOfBuildings, optimizationType)

    # logfile

    now = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")

    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open(f"../results/log/logfile_{now}.log",
                            "a")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

        def close(self):
            self.log.close()

    sys.stdout = Logger()

    # create an energy network and set the network parameters from an excel file
    network = EnergyNetwork(timePeriod)
    network.setFromExcel(os.path.join(inputFilePath, inputfileName),
                         numberOfBuildings,
                         opt=optimizationType,
                         mergeLinkBuses=mergeLinkBuses,
                         clusterSize=cluster,
                         dispatchMode=dispatchMode)

    # solver specific command line options
    optimizationOptions = {
        "gurobi": {
            "BarConvTol": 0.5,
            # The barrier solver terminates when the relative difference between the primal and dual objective values is less than the specified tolerance (with a GRB_OPTIMAL status)
            "OptimalityTol": 1e-4,
            # Reduced costs must all be smaller than OptimalityTol in the improving direction in order for a model to be declared optimal
            "MIPGap": 1e-2,
            # Relative Tolerance between the best integer objective and the objective of the best node remaining
            "MIPFocus": 1
            # 1 feasible solution quickly. 2 proving optimality. 3 if the best objective bound is moving very slowly/focus on the bound
            # "Cutoff": #Indicates that you aren't interested in solutions whose objective values are worse than the specified value., could be dynamically be used in moo
        }
        # add the command line options of the solver here, For example to use CBC add a new item to the dictionary
        # "cbc": {"tee": False}
    }

    # optimize the energy network
    limit, capacitiesTransformers, capacitiesStorages = network.optimize(
        solver='gurobi',
        numberOfBuildings=numberOfBuildings,
        options=optimizationOptions,
        optConstraints=["roof area"],
        mergeLinkBuses=mergeLinkBuses,
        clusterSize=cluster,
        envImpactlimit=1000000000000000)

    # print optimization outputs i.e. costs, environmental impact and capacities selected for different components (with investment optimization)
    network.printInvestedCapacities(capacitiesTransformers, capacitiesStorages)
    network.printCosts()
    network.printEnvImpacts()
    network.printMetaresults()

    meta = network.printMetaresults()

    # save results
    if not os.path.exists(resultFilePath):
        os.makedirs(resultFilePath)
    print(os.path.join(resultFilePath, resultFileName))
    network.exportToExcel(os.path.join(resultFilePath, resultFileName), mergeLinkBuses=mergeLinkBuses)

    # plot sankey diagram
    UseLabelDict = True  # a dictionary defining the labels to be used for different flows
    figureFilePath = "../figures"
    sankeyFileName = f"Sankey_{scenario}.html"

    snk.plot(os.path.join(resultFilePath, resultFileName), os.path.join(figureFilePath, sankeyFileName),
             numberOfBuildings, UseLabelDict, labels='default', optimType='indiv')

    # plot detailed energy flow
    plotLevel = "allMonths"  # permissible values (for energy balance plot): "allMonths" {for all months}
    # or specific month {"Jan", "Feb", "Mar", etc. three letter abbreviation of the month name}
    # or specific date {format: YYYY-MM-DD}
    plotType = "bokeh"  # permissible values: "energy balance", "bokeh"
    flowType = "electricity"  # permissible values: "all", "electricity", "space heat", "domestic hot water"

    fnc.plot(os.path.join(resultFilePath, resultFileName), figureFilePath, numberOfBuildings, plotLevel, plotType,
             flowType)

    sys.stdout = sys.__stdout__
