import pandas as pd
import os
import sys
from datetime import datetime
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
from optihood.weather_cluster import weather_cluster as meteo
from optihood.energy_network import EnergyNetworkIndiv as EnergyNetwork
import optihood.plot_sankey as snk
import optihood.plot_functions as fnc


if __name__ == '__main__':

    # In this example we consider 12 clusters
    # 12 representative days for the whole year
    # The number of days belonging to each cluster is defined in the dictionary 'cluster'
    # cluster = {"2021-07-30": 26,
    #            "2021-02-03": 44,
    #            "2021-07-23": 32,
    #            "2021-09-18": 28,
    #            "2021-04-15": 22,
    #            "2021-10-01": 32,
    #            "2021-11-04": 32,
    #            "2021-10-11": 37,
    #            "2021-01-24": 15,
    #            "2021-08-18": 26,
    #            "2021-05-28": 23,
    #            "2021-02-06": 48}

    # set a time period for the optimization problem according to the size of clusers
    timePeriod = pd.date_range("2021-01-01 00:00:00", "2021-01-12 23:00:00", freq="60min")

    # define paths for input and result files
    inputFilePath = r"..\excels\clustering"
    inputfileName = "scenario_Annual_4_costs_100%_SH35_last.xls"
    
    resultFilePath =r"..\results"
    resultFileName ="results_Yea_SPT.xlsx"
    
    #create weather file based on coordinates and PVGIS or supplying file to read
    addr_source=os.path.join(inputFilePath, inputfileName)
    
    """ initialize parameters"""
    numberOfBuildings = 4
    optimizationType = "costs"  # set as "env" for environmental optimization
    mergeLinkBuses_bool=False
    N_cl=12 # number of meteo day clusters
    
    # logfile

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            if not os.path.exists("../results/log"):
                os.mkdir("../results/log")
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
    
    
    
    """ Create meteo file and compute clusters if needed."""
    meteo_data=meteo(source=addr_source,
                     n_clusters=N_cl,
                     cluster=True,
                     clustering_vars=[],
                     save_file=False,
                     load_file=True)
    meteo_data.results.index=meteo_data.results.index.strftime('%Y-%m-%d')
    clusterBook=meteo_data.code_BK
    cluster=meteo_data.results['count'].to_dict()
    
    """declare custom cluster vector & code Book here to bypass 
    cluster computation by weather class"""
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
    
    # cluster={}
    
    """ create an energy network and set the network parameters from an excel file"""
    network = EnergyNetwork(timePeriod)
    network.setFromExcel(os.path.join(inputFilePath, inputfileName), 
                         numberOfBuildings, clusterSize=cluster, 
                         opt=optimizationType,dispatchMode=False,mergeLinkBuses=mergeLinkBuses_bool)
    
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
        },
        # add the command line options of the solver here, For example to use CBC add a new item to the dictionary
        "cbc": {"tee": False,
                "thread":4}
        }
    
    """ optimize the energy network
    """
    limit, capacitiesTransformers, capacitiesStorages = network.optimize(
        solver='gurobi', 
        numberOfBuildings=numberOfBuildings, 
        clusterSize=cluster,
        # clusterBook=clusterBook,
        mergeLinkBuses=mergeLinkBuses_bool,
        optConstraints=["roof area"],
        options=optimizationOptions,
        envImpactlimit=1000000000000000)
        

    # print optimization outputs i.e. costs, environmental impact and capacities 
    # selected for different components (with investment optimization)
    
    network.printInvestedCapacities(capacitiesTransformers, capacitiesStorages)
    network.printCosts()
    network.printEnvImpacts()
    network.printMetaresults()

    meta = network.printMetaresults()

    # save results
    if not os.path.exists(resultFilePath):
        os.makedirs(resultFilePath)
    
    network.exportToExcel(os.path.join(resultFilePath, resultFileName),mergeLinkBuses=True)
    print(os.path.join(resultFilePath, resultFileName))
    plot_bool=False
    if plot_bool==True:
        # plot sankey diagram
        UseLabelDict = True     # a dictionary defining the labels to be used for different flows
        figureFilePath = r"..\figures"
        sankeyFileName = f"Sankey_{numberOfBuildings}_{optimizationType}_ \
            {resultFileName.split(sep='.')[0].split('_')[1]}_{resultFileName.split(sep='.')[0].split('_')[2]}.html"
    
        snk.plot(os.path.join(resultFilePath, resultFileName), os.path.join(figureFilePath, sankeyFileName),
                       numberOfBuildings, UseLabelDict, labels='default', optimType='indiv')
    
        # plot detailed energy flow
        plotLevel = "allMonths"  # permissible values (for energy balance plot): "allMonths" {for all months}
        # or specific month {"Jan", "Feb", "Mar", etc. three letter abbreviation of the month name}
        # or specific date {format: YYYY-MM-DD}
        plotType = "bokeh"  # permissible values: "energy balance", "bokeh"
        flowType = "electricity"  # permissible values: "all", "electricity", "space heat", "domestic hot water"
    
        fnc.plot(os.path.join(resultFilePath, resultFileName), figureFilePath, numberOfBuildings, plotLevel, plotType, flowType)
        
        sys.stdout = sys.__stdout__