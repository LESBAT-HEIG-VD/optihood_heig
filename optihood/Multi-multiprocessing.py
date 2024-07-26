# import jupyter_server.services.config.handlers
import pandas as pd
import numpy as np
import os
try:
    import loadProfilesResidential as Resi
except ImportError:
    Resi = None
try:
    import shoppingmall as Shop
except ImportError:
    Shop = None
from multiprocessing import Process, Queue
import concurrent.futures
import sys
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import email_notification

# Parameters
optMode = "group"  # choice between "group" and "indiv"
createProfiles = False
cluster = False
mergeLinkBuses = True
numberOfBuildings = 4
numberOfOptimizations = 5

inputFilePath = '.\data\\'
if len(sys.argv) >= 2:
    resultFilePath = f'./data/results{sys.argv[1]}'
    inputfileName = f'scenario_Annual_{numberOfBuildings}_costs_{sys.argv[1]}_SH35.xls'
else:
    resultFilePath = "./data/results"
    inputfileName = f'scenario_Annual_{numberOfBuildings}_costs.xls'
print('Arguments:', resultFilePath, '//', inputfileName)

if optMode == "indiv":
    from optihood.energy_network import EnergyNetworkIndiv as EnergyNetwork
elif optMode == "group":
    from optihood.energy_network import EnergyNetworkGroup as EnergyNetwork

if createProfiles:
    residentialBuildings = pd.read_excel(os.path.join(inputFilePath, inputfileName), sheet_name="residential")
    for i in range(len(residentialBuildings)):
        res = residentialBuildings.iloc[i]
        building = Resi.Residential(res)
        building.create_profile()
    shoppingMalls = pd.read_excel(os.path.join(inputFilePath, inputfileName), sheet_name="mall")
    for i in range(len(shoppingMalls)):
        mall = shoppingMalls.iloc[i]
        building = Shop.Shopping(mall)
        building.create_profile()

if cluster:  # at the moment, we have 12 clusters (12 days in the analysis)
    clusterSize = {"2021-07-30": 26,
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
    timePeriod = ["2021-01-01 00:00:00",
                  "2021-01-12 23:00:00"]  # 1 Jan is a specific case (for elec_impact), so we start from 2
else:
    clusterSize = {}
    timePeriod = ["2021-01-01 00:00:00", "2021-12-31 23:00:00"]

optimizationOptions = {
    "gurobi": {
        "BarConvTol": 0.5,
        # The barrier solver terminates when the relative difference between the primal and dual objective values is less than the specified tolerance (with a GRB_OPTIMAL status)
        #"NonConvex":2, # when 0 error is being sent when non-convex, 1 when non-convex funktion could not be linearized, 2 bilinear form and spacial branching for non-convex
        "OptimalityTol": 1e-2,
        # Reduced costs must all be smaller than OptimalityTol in the improving direction in order for a model to be declared optimal
        # "PoolGap":1  #Determines how large a (relative) gap to tolerate in stored solutions. When this parameter is set to a non-default value, solutions whose objective values exceed that of the best known solution by more than the specified (relative) gap are discarded.
        "MIPGap": 1e-2,
        # Relative Tolerance between the best integer objective and the objective of the best node remaining
        "MIPFocus": 3
        # 1 feasible solution quickly. 2 proving optimality. 3 if the best objective bound is moving very slowly/focus on the bound
        # "Cutoff": #Indicates that you aren't interested in solutions whose objective values are worse than the specified value., could be dynamically be used in moo
    },
    "CBC ": {

    },
    "GLPK": {

    }}

if clusterSize:
    optimizationOptions['gurobi'][
        'MIPGap'] = 1e-4  # If clusterSize is set, reduce the MIP Gap parameter in optimizationOptions to 1e-4 (else 1% is acceptable)


def optimizeNetwork(network, instance, envImpactlimit, mergeLinkBuses=False):
    """
    lauch optimization of the network
    save individual building results files (xlsx) to resultFilePath
    :param network:
    :param instance:
    :param envImpactlimit:
    :return:
    """

    limit, capacitiesTransformers, capacitiesStorages = network.optimize(
        solver='gurobi',
        numberOfBuildings=numberOfBuildings,
        envImpactlimit=envImpactlimit,
        clusterSize=clusterSize,
        optConstraints=['roof area'],
        options=optimizationOptions,
        mergeLinkBuses=mergeLinkBuses,
    )

    # print global results to log file
    network.printInvestedCapacities(capacitiesTransformers, capacitiesStorages)
    network.printCosts()
    network.printEnvImpacts()

    # get Total Costs, TotalEnvImpacts and meta restults
    costs = network.getTotalCosts()
    envImpacts = network.getTotalEnvImpacts()
    meta = network.printMetaresults()

    # save individual buildings resutls to xlx files
    if not os.path.exists(resultFilePath):
        os.makedirs(resultFilePath)
    network.exportToExcel(
        resultFilePath + "\\results" + str(numberOfBuildings) + '_' + str(instance) + '_' + optMode + '.xlsx',
        mergeLinkBuses=mergeLinkBuses,
    )


    return (limit, envImpacts, costs, meta)

def plotParetoFront(costsList, envList, resultFilePath):
    plt.figure()
    plt.plot(costsList, envList, 'o-.')
    plt.xlabel('Costs (CHF)')
    plt.ylabel('Emissions (kgCO2eq)')
    plt.title('Pareto-front')
    plt.grid(True)
    plt.savefig(os.path.join(resultFilePath, f"ParetoFront.png"))
    print("Costs : (CHF)")
    print(costsList)
    print("Emissions : (kgCO2)")
    print(envList)

def f1(q):
    # cost optimum
    old_stdout = sys.stdout
    if not os.path.exists(os.path.join(resultFilePath, "log_files")):
        os.makedirs(os.path.join(resultFilePath, "log_files"))

    log_file = open(os.path.join(resultFilePath, "log_files\\optimization1.log"), "w")
    sys.stdout = log_file
    print("******************\nOPTIMIZATION " + str(1) + "\n******************")
    network = EnergyNetwork(pd.date_range(timePeriod[0], timePeriod[1], freq="60min"))
    network.setFromExcel(os.path.join(inputFilePath, inputfileName), numberOfBuildings, clusterSize, opt="costs",
                         mergeLinkBuses=mergeLinkBuses)
    (limit, max_env, min_costs, meta) = optimizeNetwork(network, 1, 1000000,
                                                        mergeLinkBuses=mergeLinkBuses)
    # costsListLast = meta['objective']
    sys.stdout = old_stdout
    # print("comparison of meta results with processed results : {} ?= {}".format(meta['objective'], min_costs))
    log_file.close()
    q.put((limit, max_env, min_costs, meta['objective']))

def f2(q):
    # environmental optimum
    old_stdout = sys.stdout
    if not os.path.exists(os.path.join(resultFilePath, "log_files")):
        os.makedirs(os.path.join(resultFilePath, "log_files"))

    log_file = open(os.path.join(resultFilePath, "log_files\\optimization2.log"), "w")
    sys.stdout = log_file
    print("******************\nOPTIMIZATION " + str(2) + "\n******************")
    network = EnergyNetwork(pd.date_range(timePeriod[0], timePeriod[1], freq="60min"))
    network.setFromExcel(os.path.join(inputFilePath, inputfileName), numberOfBuildings, clusterSize, opt="env",
                         mergeLinkBuses=mergeLinkBuses)
    (limit, min_env, max_costs, meta) = optimizeNetwork(
        network, 2, 1000000,
        mergeLinkBuses=mergeLinkBuses
    )
    # costsList.append(costs)
    # envList.append(min_env)
    sys.stdout = old_stdout
    # print("comparison of meta results with processed results : {} ?= {}".format(meta['objective'], max_costs))
    log_file.close()
    q.put((limit, min_env, max_costs, meta['objective']))

def fi(instance, envCost, q):
    old_stdout = sys.stdout
    if not os.path.exists(os.path.join(resultFilePath, "log_files")):
        os.makedirs(os.path.join(resultFilePath, "log_files"))

    log_file = open(os.path.join(resultFilePath, f"log_files\\optimization{instance}.log"), "w")
    sys.stdout = log_file
    print("******************\nOPTIMIZATION " + str(instance) + "\n******************")
    network = EnergyNetwork(pd.date_range(timePeriod[0], timePeriod[1], freq="60min"))
    network.setFromExcel(os.path.join(inputFilePath, inputfileName), numberOfBuildings, clusterSize, opt="costs",
                         mergeLinkBuses=mergeLinkBuses
                         )
    (limit, envImpacts, costs, meta) = optimizeNetwork(network, instance, envCost + 1, mergeLinkBuses=mergeLinkBuses)
    #costsList.append(meta['objective'])
    #envList.append(limit)
    sys.stdout = old_stdout
    log_file.close()
    q.put((limit, envImpacts, costs, meta['objective'], instance))


if __name__ == '__main__':
    # initiate email notification
    email = email_notification.Email()
    if not email.receiver_email:
        email.config_email()

    costsList = numberOfOptimizations * [0]
    envList = numberOfOptimizations * [0]
    limitList = numberOfOptimizations * [0]
    metaList = numberOfOptimizations * [0]

    q1 = Queue()
    q2 = Queue()
    p1 = Process(target=f1, args=(q1,))
    p2 = Process(target=f2, args=(q2,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    (limitList[-1], envList[-1], costsList[-1], metaList[-1]) = q1.get()   # cost optimum
    (limitList[0], envList[0], costsList[0], metaList[0]) = q2.get()    # env optimum
    max_limit = limitList[-1]
    min_limit = limitList[0]


    print(
        'Each iteration will keep emissions lower than some values between femissions_min and femissions_max, so [' + str(
            min_limit) + ', ' + str(max_limit) + ']')
    #steps = list(range(int(min_env), int(max_env), int((max_env - min_env) / (numberOfOptimizations - 1))))
    steps = list(np.geomspace(int(min_limit), int(max_limit), numberOfOptimizations))
    steps = steps[1:numberOfOptimizations-1]
    print(steps)

    qi = Queue()
    processes = []
    for i, limit in enumerate(steps):
        instance = i + 3
        p = Process(target=fi, args=(instance, limit, qi,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    while not qi.empty():
        (limit, envImpacts, costs, meta, instance) = qi.get()
        costsList[instance-2] = costs
        envList[instance-2] = envImpacts
        limitList[instance-2] = limit
        metaList[instance-2] = meta

    # -----------------------------------------------------------------------------#
    ## Plot Paretofront ##
    # -----------------------------------------------------------------------------#
    #insert env optimum at the begging
    # costsList.insert(0, costsListFirst)
    # envList.insert(0, min_env)


    # costsList.append(costsListLast)
    # envList.append(max_env)

    plotParetoFront(costsList, envList, inputFilePath)

    # output values to text file
    with open(os.path.join(resultFilePath, 'pareto_values.txt'), 'w') as f:
        f.write(f"mean annual cost [CHF]\t")
        for item in costsList:
            f.write(f"{item}\t")

        f.write(f"\n ghg emissions [kgCo2eq]\t")
        for item in envList:
            f.write(f"{item}\t")

        f.write(f"\n ghg emissions limit  [kgCo2eq]\t")
        for item in limitList:
            f.write(f"{item}\t")

        f.write(f"\n meta  \t")
        for item in metaList:
            f.write(f"{item}\t")

    email.send_email(type="optFin")
