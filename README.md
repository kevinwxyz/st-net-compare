# st-net-compare - Network Comparison App

## Objective

To compare biologically relevant co-occurrence network properties between two graphs using NetworkX in python, which aids traditional properties and plots produced via NetCoMi in R.

Link to webapp: [Network Comparison App](https://st-net-compare.streamlit.app/)

## Upload

Users are guided to upload two association graphs, first for the pooled samples network and next for the unpooled samples network, as GraphML files

## Network Properties Comparison:

Users receive a comparison of global network properties of the two networks, namely a comparison of the number of nodes, number of edges, density, global clustering coefficient, modularity, average degree, average edge weight.

Additionally, users can compare plots of degree distribution, edge weight distribution, degree vs clustering coefficient, positive vs negative edge associations for both networks.

Module statistics for the modules found by the Clauset-Newman-Moore greedy modularity maximization are also compared and an interactive visualisation of the modules is also produced.

## Saving plots and tables

- To download tables as CSVs, users can click the "Download as CSV" option which shows in the top right corner of the table when hovering anywhere over the table. 
- Plots too can be saved by right-clicking the image and choosing the "Save image as..." option.
