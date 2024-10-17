import pandas as pd
import os
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np
import pathlib as _pl
## import energy network class
# EnergyNetworkIndiv for individual optimization
# EnergyNetworkGroup for grouped optimization

from optihood.energy_network import EnergyNetworkGroup as EnergyNetwork
from optihood.weather import weather as meteo

def plotParetoFront(filePath, costsList, envList):
    plt.figure()
    plt.plot(costsList, envList, 'o-.')
    plt.xlabel('Costs (CHF)')
    plt.ylabel('Emissions (kgCO2eq)')
    plt.title('Pareto-front')
    plt.grid(True)
    plt.savefig(filePath)
    print("Costs : (CHF)")
    print(costsList)
    print("Emissions : (kgCO2)")
    print(envList)

def calcul_steps(min_env, max_env, number_of_steps):

    steps_H=[max_env]
    steps_L=[min_env]
    mid_step_origin=(max_env+min_env)/2
    mid_step_L=mid_step_origin
    mid_step_H=mid_step_origin
    mid_p=number_of_steps//2
    for i in range(1,int(mid_p)):
        step_val_H=(max_env+mid_step_H)/2
        steps_H.insert(1,step_val_H)
        mid_step_H=step_val_H
        
        step_val_L=(min_env+mid_step_L)/2
        steps_L.insert(1,step_val_L)
        mid_step_L=step_val_L
        
    print(steps_L)
    print(steps_H)
    steps_L.reverse()
    steps_L.insert(0,mid_step_origin)
    steps_H.extend(steps_L)
    steps=steps_H
    
        
    return steps

if __name__ == '__main__':

    # initialize parameters
    numberOfOptimizations = 7 # ODD NUMBER number of optimizations in multi objective optimization pareto front
    if numberOfOptimizations%2==0:
        numberOfOptimizations=numberOfOptimizations+1
    numberOfBuildings = 10
    cluster_N = [0]
    merge_opt = [True]
    con_opt = ["Con"]  # ["Con","noCon"]
    clst_opt = [True]
    clN=0
    if clN==0:
        cl=False
    else:
        cl=True
    # for clN in cluster_N:
    # In this example we consider 12 clusters
    # 12 representative days for the whole year
    # The number of days belonging to each cluster is defined in the dictionary 'cluster'

    """ file management"""
    curDir = _pl.Path(__file__).resolve().parent
    # define paths for input and result files
    # inputFilePath = r"..\excels\clustering"
    # inputfileName = "scenario_Annual_1_costs_100%_SH35_cluster_HPOnly.xls"
    # inputfileName = "scenario_Annual_2_costs_100%_SH35_cluster_HPOnly.xls"
    # inputfileName = "scenario_Annual_10_costs_TES_Final.xls"
    # inputfileName = "scenario_Annual_2_costs_TES.xls"
    inputFilePath = curDir / ".." / "excels" / "IamLenz"
    # inputFilePath = curDir / ".." / "excels" / "pvt_example"
    # inputfileName = "scenario_IamLenz_2.xls"
    # inputfileName = "scenario_IamLenz_10_costs_075_TES_CostlyBiCAD.xls"
    inputfileName = "scenario_IamLenz_10_075_TES_GSHP_PV_ST_PVT_mergeON.xls"
    # inputfileName = "scenario_IamLenz_2_costs_075_TES_GSHP_PV_ST_PVT-CAD.xls"

    resultFilePath = r"..\results"
    # resultFileName = "results_TES_Final_10bld_allHP_PV_PVT_ST_TES.xlsx"
    resultFileName = "results_IamLenz_TES_mrgON_10.xlsx"
    # resultFileName = "results_TES_env_" + str(clN) + "_TES_allHP_10bld.xlsx"

    # create weather file based on coordinates and PVGIS or supplying file to read
    addr_source = os.path.join(inputFilePath, inputfileName)

    """ initialize parameters"""
    # set a time period for the optimization problem according to the size of clusers
    timePeriod = pd.date_range("2021-01-01 00:00:00", "2021-12-31 23:00:00", freq="60min")
    optimizationType = "costs"  # set as "env" for environmental optimization
    mergeLinkBuses_bool = True
    tL_bool = True  # temperature levels flag
    """ if tL_bool==False -> single dT and Tinlet for solar technologies
     and if True and stratified storage is interesting then mergeBuses
     points to heat_buses
    """
    # mergeBuses = ["electricity",
    #               "space_heat",
    #               "domestic_hot_water",
    #               # "heat_buses"
    #               ]

    mergeBuses = ["electricity",
                  # "space_heat",
                  # "domestic_hot_water",
                  "heat_buses"
                  ]
    constraints_opt = ["roof area"]
    clusterBool = cl
    if clusterBool == True:
        MIPGap_val = 0.001
    else:
        MIPGap_val = 0.01
    N_cl = clN  # number of meteo day clusters
    plot_bool = False  # specify if sankey plots are required

    """ Create meteo file and compute clusters if needed."""
    meteo_data=meteo(source=addr_source,
                     n_clusters=N_cl,
                     cluster=clusterBool,
                     clustering_vars=[],
                     save_file=True,
                     load_file=False,
                     set_scenario=True,
                     single_scenario=False)
    # create electricity profile based on Romande Energie tarif
    # or spot profile in electricity_spot.csv
    # options are : "Tarif" or "Spot"
    meteo_data.elec(profile_elec="Tarif")
    """declare custom cluster vector & code Book here to bypass 
    cluster computation by weather class"""
    # {"2021-07-30": 26,
    #           "2021-02-03": 44,
    #           "2021-07-23": 32,
    #           "2021-09-18": 28,
    #           "2021-04-15": 22,
    #           "2021-10-01": 32,
    #           "2021-11-04": 32,
    #           "2021-10-11": 37,
    #           "2021-01-24": 15,
    #           "2021-08-18": 26,
    #           "2021-05-28": 23,
    #           "2021-02-06": 48}
    # clusterBook=pd.DataFrame(data=range(1,13),
    #                          columns=["day_index"],
    #                          index=pd.date_range(start='01-01-2018 00:00',periods=12,freq="D"))

    if clusterBool==True:
        meteo_data.results.index=meteo_data.results.index.strftime('%Y-%m-%d')
        clusterBook=pd.DataFrame(meteo_data.code_BK)
        cluster=meteo_data.results['count'].to_dict()
    elif clusterBool==False: #if no cluster is required, cluster={}
        cluster={}
        clusterBook=pd.DataFrame()



    # solver specific command line options
    optimizationOptions = {
        "gurobi": {
            "BarConvTol": 0.5,
            # The barrier solver terminates when the relative difference between the primal and dual objective values is less than the specified tolerance (with a GRB_OPTIMAL status)
            "OptimalityTol": 1e-4,
            # Reduced costs must all be smaller than OptimalityTol in the improving direction in order for a model to be declared optimal
            "MIPGap": MIPGap_val,
            # Relative Tolerance between the best integer objective and the objective of the best node remaining
            "MIPFocus": 2
            # 1 feasible solution quickly. 2 proving optimality. 3 if the best objective bound is moving very slowly/focus on the bound
            # "Cutoff": #Indicates that you aren't interested in solutions whose objective values are worse than the specified value., could be dynamically be used in moo
        }
        # add the command line options of the solver here, For example to use CBC add a new item to the dictionary
        #"cbc": {"tee": False}
    }

    # lists of costs and environmental impacts for different optimizations
    costsList = []
    envList = []
    steps=[]
    for opt in range(1, numberOfOptimizations+1):

        # First optimization is by Cost alone
        # Second optimization is by Environmental impact alone
        # Third optimization onwards are the steps in between Cost-Optimized and Env-Optimized (by epsilon constraint method)
        
        if opt == 2:        # Environmental optimization
            optimizationType = "env"
            envImpactlimit = max_env
            print(envImpactlimit)
        else:
            optimizationType = "costs"
            if opt == 1:    # Cost optimization
                envImpactlimit = 1000000000000                
                print(envImpactlimit)
            else:           # Constrained optimization for multi-objective analysis (steps between Cost-Optimized and Env-Optimized)
                # envImpactlimit = steps[opt - 3]
                envImpactlimit = steps[opt - 2]
                # envImpactlimit=(envImpactlimit_high+envImpactlimit_low)/2
                print(envImpactlimit)
        
        print("******************\nOPTIMIZATION " + str(opt) + "\n******************")

        """ create an energy network and set the network parameters from an excel file"""
        network = EnergyNetwork(timePeriod, cluster, temperatureLevels=tL_bool)
        network.setFromExcel(os.path.join(inputFilePath, inputfileName),
                             numberOfBuildings, clusterSize=cluster,
                             opt=optimizationType, dispatchMode=False,
                             mergeLinkBuses=mergeLinkBuses_bool,
                             mergeHeatSourceSink=False,
                             mergeBuses=mergeBuses)

        limit, capacitiesTransformers, capacitiesStorages = network.optimize(
            solver='gurobi',
            numberOfBuildings=numberOfBuildings,
            clusterSize=cluster,
            clusterBook=clusterBook,
            mergeLinkBuses=mergeLinkBuses_bool,
            optConstraints=constraints_opt,
            options=optimizationOptions,
            envImpactlimit=envImpactlimit)

        print("voici limit :", limit)
        """
        # create an energy network and set the network parameters from an excel file
        network = EnergyNetwork(timePeriod)
        network.setFromExcel(os.path.join(inputFilePath, inputfileName), numberOfBuildings, opt=optimizationType)
        
        # optimize the energy network
        env, capacitiesTransformers, capacitiesStorages = network.optimize(solver='gurobi',
                                                                           envImpactlimit=envImpactlimit,
                                                                           numberOfBuildings=numberOfBuildings,
                                                                           options=optimizationOptions)
        """

        # print optimization outputs i.e. costs, environmental impact and capacities selected for different components (with investment optimization)
        network.printInvestedCapacities(capacitiesTransformers, capacitiesStorages)
        network.printCosts()
        network.printEnvImpacts()

        # save results
        resultFileName = "results_pareto_mrgON_IamLenz_2_oldCAD_GSHP_2ndRun_pvtOnly" + str(numberOfBuildings) + '_' + str(opt) + '.xlsx'    # result filename for each optimization

        if not os.path.exists(resultFilePath):
            os.makedirs(resultFilePath)

        network.exportToExcel(os.path.join(resultFilePath, resultFileName),mergeLinkBuses=True)

        costs = network.getTotalCosts()
        env_impact=limit #network.getTotalEnvImpacts()
        meta = network.printMetaresults()

        if opt == 1:  # Cost optimization
            # costsListLast = meta['objective']
            # envListLast = limit
            costsList=[costs]
            envList=[env_impact]
            max_env = env_impact
            envImpactlimit_high=max_env
            print("max_env", max_env)
        else:
            if opt == 2:  # Environmental optimization
                min_env = env_impact
                print("min_env", min_env)
                costsListLast=costs
                envListLast=env_impact
                envImpactlimit_low=min_env
                # Define steps for multi-objective optimization
                steps = calcul_steps(min_env , max_env ,numberOfOptimizations)
                print(steps)
            else:  # Constrained optimization for multi-objective analysis (steps between Cost-Optimized and Env-Optimized)
                envImpactlimit_high=env_impact
                print("run > 3, envLimit:", env_impact)
                costsList.append(costs)
                envList.append(env_impact)
                print(envList)
                print(steps)

    costsList.append(costsListLast)
    envList.append(envListLast)
    print(costsList)
    print(envList)
    print(steps)
    # plot pareto front to visualize multi objective optimization results
    figureFilePath = r"..\figures"
    if not os.path.exists(figureFilePath):
        os.makedirs(figureFilePath)

    figureFileName = f"Pareto_IamLenz_10_GSHP_PV_PVT_ST_075TES_mrgON.png"

    plotParetoFront(os.path.join(figureFilePath, figureFileName), costsList, envList)


