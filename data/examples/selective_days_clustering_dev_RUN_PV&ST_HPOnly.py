import pandas as pd
import os
import sys
from datetime import datetime
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
from optihood.weather import weather as meteo
from optihood.energy_network import EnergyNetworkGroup as EnergyNetwork
import optihood.plot_sankey as snk
import optihood.plot_functions as fnc



if __name__ == '__main__':

    cluster_N=[0]#72
    merge_opt=[True]
    con_opt=["Con"] #["Con","noCon"]
    clst_opt=[True]
    
    for cc in con_opt:
         for mm in merge_opt:
              for clN in cluster_N:
                    if clN==0:
                        cl=False
                    else:
                        cl=True
                # clN=48
                    # for clN in cluster_N:
                    # In this example we consider 12 clusters
                    # 12 representative days for the whole year
                    # The number of days belonging to each cluster is defined in the dictionary 'cluster'
                
                    """ file management"""
                    # define paths for input and result files
                    inputFilePath = r"..\excels\clustering"
                    inputfileName = "scenario_Annual_10_costs_100%_SH35_cluster_HPOnly.xls"
                    # inputfileName = "scenario_Annual_10_costs_100%_SH35_cluster_GASOnly.xls"
                    # inputfileName = "scenario_Annual_10_costs_100%_SH35_cluster_GWHPOnly.xls"
                    # inputfileName = "scenario_Annual_9_costs_100%_SH35_cluster_GASOnly.xls"
                    # inputfileName = "scenario_Annual_8_costs_HP.xls"
                    # inputfileName = "scenario_Annual_2_costs_TES.xls"
                    # inputfileName = "scenario.xls"
                    
                    resultFilePath =r"..\results"
                    resultFileName ="results_SH_DHW_10bld_GWHP+ST.xlsx"
                    
                    #create weather file based on coordinates and PVGIS or supplying file to read
                    addr_source=os.path.join(inputFilePath, inputfileName)
                    
                    """ initialize parameters"""
                    # set a time period for the optimization problem according to the size of clusers
                    timePeriod = pd.date_range("2021-01-01 00:00:00", "2021-12-31 23:00:00", freq="60min")
                    numberOfBuildings = 10
                    optimizationType = "costs"  # set as "env" for environmental optimization
                    mergeLinkBuses_bool=True 
                    tL_bool=False #temperature levels flag
                    """ if tL_bool==False -> single dT and Tinlet for solar technologies
                     and if True and stratified storage is interesting then mergeBuses
                     points to heat_buses
                    """
                    mergeBuses=["electricity",
                                "space_heat",
                                "domestic_hot_water",
                                # "heat_buses"
                                ]
                    constraints_opt=["roof area"]
                    clusterBool=cl
                    if clusterBool==True:
                        MIPGap_val=0.001
                    else:
                        MIPGap_val=0.01 
                    N_cl=clN # number of meteo day clusters
                    plot_bool=False # specify if sankey plots are required
                    
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
                    
                    """ create an energy network and set the network parameters from an excel file"""
                    network = EnergyNetwork(timePeriod,cluster,temperatureLevels=tL_bool)
                    network.setFromExcel(os.path.join(inputFilePath, inputfileName), 
                                         numberOfBuildings, clusterSize=cluster, 
                                         opt=optimizationType,dispatchMode=False,
                                         mergeLinkBuses=mergeLinkBuses_bool,
                                          mergeHeatSourceSink=False,                                         
                                           mergeBuses=mergeBuses)
                    
                    optimizationOptions = {
                        "gurobi": {
                            #"BarConvTol": 0.5,
                            # The barrier solver terminates when the relative difference between the primal and dual objective values is less than the specified tolerance (with a GRB_OPTIMAL status)
                            #"OptimalityTol": 1e-4,
                            # Reduced costs must all be smaller than OptimalityTol in the improving direction in order for a model to be declared optimal
                            "MIPGap": MIPGap_val,
                            # Relative Tolerance between the best integer objective and the objective of the best node remaining
                            "MIPFocus": 1
                            # 1 feasible solution quickly. 2 proving optimality. 3 if the best objective bound is moving very slowly/focus on the bound
                            # "Cutoff": #Indicates that you aren't interested in solutions whose objective values are worse than the specified value., could be dynamically be used in moo
                        },
                        # add the command line options of the solver here, For example to use CBC add a new item to the dictionary
                        "cbc": {"tee": False,
                                } #"thread":4
                        }
                    
                    """ optimize the energy network
                    """
                    limit, capacitiesTransformers, capacitiesStorages = network.optimize(
                        solver='gurobi', 
                        numberOfBuildings=numberOfBuildings, 
                        clusterSize=cluster,
                        clusterBook=clusterBook,
                        mergeLinkBuses=mergeLinkBuses_bool,
                        optConstraints=constraints_opt,
                        options=optimizationOptions,
                        envImpactlimit=100000000000000000)
                
                    # print optimization outputs i.e. costs, environmental impact and capacities 
                    # selected for different components (with investment optimization)
                    
                    network.printInvestedCapacities(capacitiesTransformers, capacitiesStorages)
                    network.printCosts()
                    network.printEnvImpacts()
                
                    # save results
                    if not os.path.exists(resultFilePath):
                        os.makedirs(resultFilePath)
                    
                    network.exportToExcel(os.path.join(resultFilePath, resultFileName),mergeLinkBuses=True)
                    
                    if plot_bool==True:
                        # plot sankey diagram
                        UseLabelDict = False     # a dictionary defining the labels to be used for different flows
                        figureFilePath = r"..\figures"
                        if not os.path.exists(figureFilePath):
                            os.makedirs(figureFilePath)
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
                        
                        plotLevel = "allMonths"  # permissible values (for energy balance plot): "allMonths" {for all months}
                        # or specific month {"Jan", "Feb", "Mar", etc. three letter abbreviation of the month name}
                        # or specific date {format: YYYY-MM-DD}
                        plotType = "bokeh"  # permissible values: "energy balance", "bokeh"
                        flowType = "domestic hot water"  # permissible values: "all", "electricity", "space heat", "domestic hot water"
                    
                        fnc.plot(os.path.join(resultFilePath, resultFileName), figureFilePath, numberOfBuildings, plotLevel, plotType, flowType)
                        