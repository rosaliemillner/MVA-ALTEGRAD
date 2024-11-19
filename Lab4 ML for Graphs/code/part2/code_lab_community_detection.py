"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans



############## Task 3
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    
    adjacency_matrix = nx.adjacency_matrix(G)

    deg = np.array([G.degree(node) for node in G.nodes()])
    degree_matrix_inverted = diags(1/deg)

    laplacian = eye(G.number_of_nodes()) - degree_matrix_inverted @ adjacency_matrix

    _, eigenvectors = eigs(laplacian, k=k, which='SM')

    U = eigenvectors.real
    #print(U.shape)
    
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(U)

    cluster_assignments = kmeans.labels_
    #clustering = {node: int(cluster) for node, cluster in enumerate(cluster_assignments)}
    clustering = {node: int(cluster) for node, cluster in zip(G.nodes(), cluster_assignments)}
    
    return clustering




############## Task 4

#Applying the Spectral Clustering to the largest connected component from graph associated to the CA-HepTh dataset
G = nx.read_edgelist("datasets/CA-HepTh.txt", delimiter='\t', comments='#', create_using=nx.Graph())

#we reuse some code from Part 1
list_components = list(nx.connected_components(G))
G_largest_nodes = max(list_components, key=len)
G_largest = G.subgraph(G_largest_nodes).copy()

G_largest_clustering = spectral_clustering(G_largest, 50)

#print(G_largest_clustering)



############## Task 5
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    m = G.number_of_edges()
    community_labels = list(clustering.values())
    Nc = len(np.unique(community_labels))
    
    community_ld = {}
    for node, group in clustering.items():
        if group not in community_ld:
            community_ld[group] = {'l' : 0, 'd' : 0}
        
       # community_ld[group]['nodes'].add(node)
        community_ld[group]['d'] += G.degree[node]
    
    for edge in G.edges():
        n1, n2 = edge
        grp1 = clustering[n1]
        grp2 = clustering[n2]
        
        if grp1 == grp2:
            community_ld[grp1]['l'] += 1
            
    modularity = 0
    for group, ld in community_ld.items():
        modularity += (ld['l']/m) - (ld['d']/(2*m))**2

    return modularity



############## Task 6


k=50
random_clustering = {node: randint(0, k-1) for node in G_largest.nodes()}

##faire clustering ici sur tout G ou sur juste largest_G ?
G_clustering = spectral_clustering(G_largest, k)
modularity_spectral_clustering = modularity(G_largest, G_clustering)
modularity_random = modularity(G_largest, random_clustering)

print("The modularity for random clustering on the largest connected component of G is: ", modularity_random)
print("The modularity for spectral clustering on the largest connected component of G is: ", modularity_spectral_clustering)


