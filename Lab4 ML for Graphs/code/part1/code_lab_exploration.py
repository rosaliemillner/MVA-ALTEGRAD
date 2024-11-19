"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1

G = nx.read_edgelist("datasets/CA-HepTh.txt", delimiter='\t', comments='#', create_using=nx.Graph())

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

print("Number of nodes =", num_nodes)
print("Number of edges =", num_edges)


############## Task 2

q = nx.is_connected(G)

if q:
    print("The graph is connected, it thus has 1 connected component.")
else:
    print("The graph is not connected.")

    list_components = list(nx.connected_components(G))
    print("The number of connected components is:", len(list_components))

    #print(list_components[0])

    G_largest_nodes = max(list_components, key=len)
    G_largest = G.subgraph(G_largest_nodes).copy()

    num_nodes_G_largest = G_largest.number_of_nodes()
    num_edges_G_largest = G_largest.number_of_edges()
    print("The number of nodes in the largest connected component is:", num_nodes_G_largest)
    print("The number of edges in the largest connected component is:", num_edges_G_largest)

    print("The fraction of nodes in the largest component is:", num_nodes_G_largest / num_nodes)
    print("The fraction of edges in the largest component is:", num_edges_G_largest / num_edges)

    #the largest connected component represents a large part of the graph!
