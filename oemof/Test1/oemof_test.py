
import pandas as pd
import oemof.solph as solph
from oemof.tools import logger
import logging
import os
import pprint as pp

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

if __name__ == '__main__':

    ############################
    # Define The Energy System #
    ############################

    #timeAndDateIndex = pd.date_range('1/6/2021', periods=24, freq='H')

    #basicEnergySystem = solph.EnergySystem(timeindex=timeAndDateIndex)

    # read data from excel file

    def nodes_from_excel(filePath):
        """Read node data from Excel sheet
        Parameters
        ----------
        filePath : :obj:`str`
            Path to excel file
        Returns
        -------
        :obj:`dict`
            Imported nodes data
        """

        # does Excel file exist?
        if not filePath or not os.path.isfile(filePath):
            raise FileNotFoundError(
                "Excel data file {} not found.".format(filePath)
            )
        data = pd.ExcelFile(filePath)
        nodesData = {
            "buses": data.parse("buses"),
            "commodity_sources": data.parse("commodity_sources"),
            "transformers": data.parse("transformers"),
            "renewables": data.parse("renewables"),
            "demand": data.parse("demand"),
            "storages": data.parse("storages"),
            "powerlines": data.parse("powerlines"),
            "timeseries": data.parse("time_series"),
        }

        # set datetime index
        nodesData["timeseries"].set_index("timestamp", inplace=True)
        nodesData["timeseries"].index = pd.to_datetime(
            nodes_data["timeseries"].index
        )

        print("Data from Excel file {} imported.".format(filePath))

        return nodes_data

    def create_nodes(nd=None):
        """Create nodes (oemof objects) from node dict
            Parameters
            ----------
            nd : :obj:`dict`
                Nodes data
            Returns
            -------
            nodes : `obj`:dict of :class:`nodes <oemof.network.Node>`
            """

        if not nd:
            raise ValueError("No nodes data provided.")

        nodes = []
        # Create Bus objects from buses table
        busd = {}

        for i, b in nd["buses"].iterrows():
            if b["active"]:
                bus = solph.Bus(label=b["label"])
                nodes.append(bus)

                busd[b["label"]] = bus
                if b["excess"]:
                    nodes.append(
                        solph.Sink(
                            label=b["label"] + "_excess",
                            inputs={
                                busd[b["label"]]: solph.Flow(
                                    variable_costs=b["excess costs"]
                                )
                            },
                        )
                    )
                if b["shortage"]:
                    nodes.append(
                        solph.Source(
                            label=b["label"] + "_shortage",
                            outputs={
                                busd[b["label"]]: solph.Flow(
                                    variable_costs=b["shortage costs"]
                                )
                            },
                        )
                    )

        # Create Source objects from table 'commodity sources'
        for i, cs in nd["commodity_sources"].iterrows():
            if cs["active"]:
                nodes.append(
                    solph.Source(
                        label=cs["label"],
                        outputs={
                            busd[cs["to"]]: solph.Flow(
                                variable_costs=cs["variable costs"]
                            )
                        },
                    )
                )

        # Create Source objects with fixed time series from 'renewables' table
        for i, re in nd["renewables"].iterrows():
            if re["active"]:
                # set static outflow values
                outflow_args = {"nominal_value": re["capacity"]}
                # get time series for node and parameter
                for col in nd["timeseries"].columns.values:
                    if col.split(".")[0] == re["label"]:
                        outflow_args[col.split(".")[1]] = nd["timeseries"][col]

                # create
                nodes.append(
                    solph.Source(
                        label=re["label"],
                        outputs={busd[re["to"]]: solph.Flow(**outflow_args)},
                    )
                )

        # Create Sink objects with fixed time series from 'demand' table
        for i, de in nd["demand"].iterrows():
            if de["active"]:
                # set static inflow values
                inflow_args = {"nominal_value": de["nominal value"]}
                # get time series for node and parameter
                for col in nd["timeseries"].columns.values:
                    if col.split(".")[0] == de["label"]:
                        inflow_args[col.split(".")[1]] = nd["timeseries"][col]

                # create
                nodes.append(
                    solph.Sink(
                        label=de["label"],
                        inputs={busd[de["from"]]: solph.Flow(**inflow_args)},
                    )
                )

            # Create Transformer objects from 'transformers' table
        for i, t in nd["transformers"].iterrows():
            if t["active"]:
                # set static inflow values
                inflow_args = {"variable_costs": t["variable input costs"]}
                # get time series for inflow of transformer
                for col in nd["timeseries"].columns.values:
                    if col.split(".")[0] == t["label"]:
                        inflow_args[col.split(".")[1]] = nd["timeseries"][col]
                # create
                nodes.append(
                    solph.Transformer(
                        label=t["label"],
                        inputs={busd[t["from"]]: solph.Flow(**inflow_args)},
                        outputs={
                            busd[t["to"]]: solph.Flow(nominal_value=t["capacity"])
                        },
                        conversion_factors={busd[t["to"]]: t["efficiency"]},
                    )
                )

        for i, s in nd["storages"].iterrows():
            if s["active"]:
                nodes.append(
                    solph.components.GenericStorage(
                        label=s["label"],
                        inputs={
                            busd[s["bus"]]: solph.Flow(
                                nominal_value=s["capacity inflow"],
                                variable_costs=s["variable input costs"],
                            )
                        },
                        outputs={
                            busd[s["bus"]]: solph.Flow(
                                nominal_value=s["capacity outflow"],
                                variable_costs=s["variable output costs"],
                            )
                        },
                        nominal_storage_capacity=s["nominal capacity"],
                        loss_rate=s["capacity loss"],
                        initial_storage_level=s["initial capacity"],
                        max_storage_level=s["capacity max"],
                        min_storage_level=s["capacity min"],
                        inflow_conversion_factor=s["efficiency inflow"],
                        outflow_conversion_factor=s["efficiency outflow"],
                    )
                )

        for i, p in nd["powerlines"].iterrows():
            if p["active"]:
                bus1 = busd[p["bus_1"]]
                bus2 = busd[p["bus_2"]]
                nodes.append(
                    solph.custom.Link(
                        label="powerline" + "_" + p["bus_1"] + "_" + p["bus_2"],
                        inputs={bus1: solph.Flow(), bus2: solph.Flow()},
                        outputs={
                            bus1: solph.Flow(nominal_value=p["capacity"]),
                            bus2: solph.Flow(nominal_value=p["capacity"]),
                        },
                        conversion_factors={
                            (bus1, bus2): p["efficiency"],
                            (bus2, bus1): p["efficiency"],
                        },
                    )
                )
        return nodes


    def draw_graph(
            grph,
            edge_labels=True,
            node_color="#AFAFAF",
            edge_color="#CFCFCF",
            plot=True,
            node_size=2000,
            with_labels=True,
            arrows=True,
            layout="neato",
    ):
        """
        Parameters
        ----------
        grph : networkxGraph
            A graph to draw.
        edge_labels : boolean
            Use nominal values of flow as edge label
        node_color : dict or string
            Hex color code oder matplotlib color for each node. If string, all
            colors are the same.
        edge_color : string
            Hex color code oder matplotlib color for edge color.
        plot : boolean
            Show matplotlib plot.
        node_size : integer
            Size of nodes.
        with_labels : boolean
            Draw node labels.
        arrows : boolean
            Draw arrows on directed edges. Works only if an optimization_model has
            been passed.
        layout : string
            networkx graph layout, one of: neato, dot, twopi, circo, fdp, sfdp.
        """
        if type(node_color) is dict:
            node_color = [node_color.get(g, "#AFAFAF") for g in grph.nodes()]

        # set drawing options
        options = {
            "prog": "dot",
            "with_labels": with_labels,
            "node_color": node_color,
            "edge_color": edge_color,
            "node_size": node_size,
            "arrows": arrows,
        }

        # try to use pygraphviz for graph layout
        try:
            import pygraphviz

            pos = nx.drawing.nx_agraph.graphviz_layout(grph, prog=layout)
        except ImportError:
            logging.error("Module pygraphviz not found, I won't plot the graph.")
            return

        # draw graph
        nx.draw(grph, pos=pos, **options)

        # add edge labels for all edges
        if edge_labels is True and plt:
            labels = nx.get_edge_attributes(grph, "weight")
            nx.draw_networkx_edge_labels(grph, pos=pos, edge_labels=labels)

        # show output
        if plot is True:
            plt.show()


    logger.define_logging()
    datetime_index = pd.date_range(
        "2016-01-01 00:00:00", "2016-01-01 23:00:00", freq="60min"
    )

    # model creation and solving
    logging.info("Starting optimization")

    # initialisation of the energy system
    esys = solph.EnergySystem(timeindex=datetime_index)

    # read node data from Excel sheet
    excel_nodes = nodes_from_excel(os.path.join(os.getcwd(), "scenario.xlsx", ))

    # create nodes from Excel sheet data
    my_nodes = create_nodes(nd=excel_nodes)

    # add nodes and flows to energy system
    esys.add(*my_nodes)

    print("*********************************************************")
    print("The following objects have been created from excel sheet:")
    for n in esys.nodes:
        oobj = str(type(n)).replace("<class 'oemof.solph.", "").replace("'>", "")
        print(oobj + ":", n.label)
    print("*********************************************************")

    # creation of a least cost model from the energy system
    om = solph.Model(esys)
    om.receive_duals()

    # solving the linear problem using the given solver
    om.solve(solver="gurobi")

    # create graph of esys
    # You can use argument filename='/home/somebody/my_graph.graphml'
    # to dump your graph to disc. You can open it using e.g. yEd or gephi
    graph = create_nx_graph(esys)

    # plot esys graph
    draw_graph(
        grph=graph,
        plot=True,
        layout="neato",
        node_size=1000,
        node_color={"R1_bus_el": "#cd3333", "R2_bus_el": "#cd3333"},
    )

    # print and plot some results
    results = solph.processing.results(om)

    region2 = solph.views.node(results, "R2_bus_el")
    region1 = solph.views.node(results, "R1_bus_el")

    print(region2["sequences"].sum())
    print(region1["sequences"].sum())

    fig, ax = plt.subplots(figsize=(10, 5))
    region1["sequences"].plot(ax=ax)
    ax.legend(
        loc="upper center", prop={"size": 8}, bbox_to_anchor=(0.5, 1.4), ncol=3
    )
    fig.subplots_adjust(top=0.7)
    plt.show()
    logging.info("Done!")

    #filePath = os.path.join(os.getcwd(), "data.xlsx")
    #naturalGasBus = solph.Bus(label='natural gas')
    #ambientTemperatureBus = solph.Bus(label='ambient temperature')
    #electricityBus = solph.Bus(label='electricity')
    #shBus = solph.Bus(label='space heating')
    #dhwBus = solph.Bus(label='domestic hot water')

    #busList = [naturalGasBus, ambientTemperatureBus, electricityBus, shBus, dhwBus]

    #basicEnergySystem.add(*busList)





