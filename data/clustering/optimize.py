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
    # optimization scenarios to run, with different clustering settings
    optims = [
        # No clustering at all
        # {
        #     'clustering': False,
        #     'name': 'results_no_invest_fixed' # name of the result file
        # },
        # 365 clusters, one per day
        # {
        #     'clustering': True,
        #     'cut': '1/1',
        #     'name': 'results_c_full_invest_fixed_no_constraint',
        #     'cluster_constraint': False
        # },
        # 365 clusters, one per day, with cluster constraint (2-days storage limit)
        # {
        #     'clustering': True,
        #     'cut': '1/1',
        #     'name': 'results_c_full_invest_fixed_with_constraint',
        #     'cluster_constraint': True
        # },
        # 182 clusters, one per 2 days
        {
            'clustering': True,
            'cut': '1/2',
            'name': 'results_c_half_invest_fixed_no_constraint',
            'cluster_constraint': False
        },
        # {
        #     'clustering': True,
        #     'cut': '1/2',
        #     'name': 'results_c_half_invest_fixed_with_constraint',
        #     'cluster_constraint': True
        # },
        # 365-36 clusters : one per day except for the 01st, 11th and 21th of each month
        # {
        #     'clustering': True,
        #     'cut': '1m',
        #     'name': 'results_c_1m'
        # },
        # 365-72 clusters : one per day except for the 01st, 06th, 11th, 16th, 21th and 26th of each month
        # {
        #     'clustering': True,
        #     'cut': '2m',
        #     'name': 'results_c_2m'
        # },
        # 365-12 clusters : one per day except for the 01st of each month
        # {
        #     'clustering': True,
        #     'cut': '1X',
        #     'name': 'results_c_1X'
        # }
    ]
    for optim_params in optims:
        # ====================
        # Define the clusters
        # ====================
        print('-=-=-=' * 20)
        print('Simulating', optim_params)
        print('-=-=-=' * 20)
        clustering = optim_params['clustering']
        cluster_constraint = optim_params['cluster_constraint'] if 'cluster_constraint' in optim_params else False
        if clustering:
            if optim_params['cut'] == '1/1':
                print('full')
                days = pd.date_range(start="2021-01-01", freq="D", periods=365)
                cluster = {str(c).replace(' 00:00:00', ''): 1 for c in days}
            elif optim_params['cut'] == '1/2':
                print('half')
                days = pd.date_range(start="2021-01-01", freq="2D", periods=182)
                cluster = {str(c).replace(' 00:00:00', ''): 2 for c in days}
                cluster['2021-01-01'] = 3
            else:
                print(optim_params['cut'])
                days = pd.date_range(start="2021-01-01", freq="D", periods=365)
                cluster = {str(c).replace(' 00:00:00', ''): 1 for c in days}
                for i in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
                    if optim_params['cut'] == '1X_15':
                        del cluster[f'2021-{i}-11']
                        cluster[f'2021-{i}-12'] = 2
                        continue
                    del cluster[f'2021-{i}-01']
                    cluster[f'2021-{i}-02'] = 2
                    if optim_params['cut'] == '1X':
                        continue
                    del cluster[f'2021-{i}-11']
                    cluster[f'2021-{i}-12'] = 2
                    del cluster[f'2021-{i}-21']
                    cluster[f'2021-{i}-22'] = 2
                    if optim_params['cut'] in ['2m', '3m']:
                        del cluster[f'2021-{i}-05']
                        cluster[f'2021-{i}-06'] = 2
                        del cluster[f'2021-{i}-15']
                        cluster[f'2021-{i}-16'] = 2
                        del cluster[f'2021-{i}-25']
                        cluster[f'2021-{i}-26'] = 2
            print('cluster', cluster)

            print('number of days', np.sum(np.array(list(cluster.values())).astype(int))) # should be 365
            print('nb clusters', len(cluster))

            # set a time period for the optimization problem according to the size of clusers
            timePeriod = pd.date_range(start="2021-01-01 00:00:00", freq="60min", periods=24*len(cluster))
            print('timePeriod', timePeriod)

            # Print the real day to cluster day mapping (for debug)
            #print('Date map:')
            #data_day = pd.date_range(start="2021-01-01", freq="D", periods=len(cluster))
            #idx = data_day
            #print('idx', idx)
            #lst = list(cluster.keys())
            #for i in range(len(lst)):
            #    print(idx[i], '->', lst[i])
        else: # no clustering
            cluster = {}
            # set a time period for the optimization problem
            timePeriod = pd.date_range("2021-01-01 00:00:00", "2021-12-31 23:00:00", freq="60min")

        # =====================
        # Run the optimization
        # =====================

        # define paths for input and result files
        inputFilePath = './'
        inputfileName = "./scenario_Annual_1_costs_0_SH35.xls"

        resultFilePath = "../results"
        resultFileName = f'{optim_params["name"]}.xlsx' # "results2_yes_mine2.xlsx" if clustering else "results2_no.xlsx"
        print('Save in', resultFileName)

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
            envImpactlimit=1000000000000000,
            clusterContraint=cluster_constraint)

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
                 flowType, cluster=cluster)

    sys.stdout = sys.__stdout__
