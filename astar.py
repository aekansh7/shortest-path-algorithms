import networkx as nx
import matplotlib.pyplot as plt

#my graph for A Star Search using networkx
def get_dist_metric(node1,node2):
    edge_weights = {(0,1): 0.8,
                    (0,2): 0.95,
                    (0,3): 0.65,
                    (1,2): 1.75,
                    (1,3): 1.30,
                    (2,3): 0.32,
                    (2,4): 0.32,
                    (2,5): 0.32,
                    (3,4): 0.12,
                    (3,5): 0.32,
                    }
    return edge_weights[(node1, node2)] 


# Sets attributes for nodes, including names
def get_info_dict(node):
    info_dict = {0: {'name': 'Subway'},
                 1: {'name': 'Starbucks'},
                 2: {'name': 'Taco bell'},
                 3: {'name': 'Hassyampa'},
                 4: {'name': 'Adelphi'},
                 5: {'name': 'Vista Del Sol'},
                 }                 
    return info_dict[node]

#code for heuristic attribute if required 
def get_heuristic(node_x, node_y):
    pass

# Code from networkx that creates the same graph
G = nx.Graph()
G.add_nodes_from([(0, get_info_dict(0)), 
                  (1, get_info_dict(1)), 
                  (2, get_info_dict(2)), 
                  (3, get_info_dict(3)),
                  (4, get_info_dict(4)),
                  (5, get_info_dict(5)),
                  ])

e = [(0, 1, get_dist_metric(0,1)), 
     (0, 2, get_dist_metric(0,2)), 
     (0, 3, get_dist_metric(0,3)), 
     (1, 2, get_dist_metric(1,2)), 
     (1, 3, get_dist_metric(1,3)), 
     (2,3, get_dist_metric(2,3)), 
     (2,4, get_dist_metric(2,4)), 
     (2,5,get_dist_metric(2,5)), 
     (3,4,get_dist_metric(3,4)), 
     (3,5,get_dist_metric(3,5))]
G.add_weighted_edges_from(e)

#show edge weights in graph when drawn

pos = nx.spring_layout(G)
# nx.draw_networkx_edge_labels(G,pos)
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='pink')
nx.draw_networkx_edges(G, pos, edgelist=e)
nx.draw_networkx_edge_labels(G, pos, edge_labels={(x,y): weight for x,y,weight in G.edges.data("weight")}, font_color='red')
nx.draw_networkx_labels(G, pos, labels=dict(G.nodes(data='name')))
plt.show()
