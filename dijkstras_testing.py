#Aekansh IEE 598 Project

#import the required libraries
import networkx as nx
import matplotlib.pyplot as plt

#my graph implementation using networkx
def get_dist_metric(node1,node2):
    edge_weights = {(0,1): 3,
                    (0,2): 5,
                    (1,2): 4,
                    (2,3): 6,
                    (0,5): 2,
                    (2,4): 4,
                    (2,6): 7,
                    (4,6): 4,
                    (1,9): 5,
                    (3,17): 7,
                    (5,8): 4,
                    (5,10): 6.5,
                    (5,7): 3,
                    (8,13): 2,
                    (8,10): 4,
                    (7,11): 1,
                    (13,14): 4,
                    (11,14): 5,
                    (9,16): 2,
                    (6,12): 5,
                    (12,16): 4,
                    (11,15): 8,
                    (14,15): 6
                    }
    return edge_weights[(node1, node2)] 


# Sets attributes for nodes, including names
def get_info_dict(node):
    info_dict = {0: {'name': 'Hassyampa'},
                 1: {'name': 'Subway'},
                 2: {'name': 'R1'},
                 3: {'name': 'Adelphi Commons'},
                 4: {'name': 'Starbucks'},
                 5: {'name': 'R2'},
                 6: {'name': 'Vista Del Sol'},
                 7: {'name': 'Papa Johns'},
                 8: {'name': 'R6'},
                 9: {'name': 'R3'},
                 10: {'name': 'Barrett'},
                 11: {'name': 'R7'},
                 12: {'name': 'R5'},
                 13: {'name': 'McDonalds'},
                 14: {'name': 'R4'},
                 15: {'name': 'Taco Bell'},
                 16: {'name': 'Tooker House'},
                 17: {'name': 'Sonora'}, 
                 }                 
    return info_dict[node]

#code for heuristic attribute if required 
def get_heuristic(node_x, node_y):
    pass

# Code from networkx that creates the graph
G = nx.Graph()
G.add_nodes_from([(0, get_info_dict(0)), 
                  (1, get_info_dict(1)), 
                  (2, get_info_dict(2)), 
                  (3, get_info_dict(3)),
                  (4, get_info_dict(4)),
                  (5, get_info_dict(5)),
                  (6, get_info_dict(6)),
                  (7, get_info_dict(7)),
                  (8, get_info_dict(8)),
                  (9, get_info_dict(9)),
                  (10, get_info_dict(10)),
                  (11, get_info_dict(11)),
                  (12, get_info_dict(12)),
                  (13, get_info_dict(13)),
                  (14, get_info_dict(14)),
                  (15, get_info_dict(15)),
                  (16, get_info_dict(16)),
                  (17, get_info_dict(17)),
                  ])

e = [(0, 1, get_dist_metric(0,1)), 
     (0, 2, get_dist_metric(0,2)), 
     (0, 5, get_dist_metric(0,5)), 
     (2, 3, get_dist_metric(2,3)), 
     (2, 4, get_dist_metric(2,4)), 
     (2,6, get_dist_metric(2,6)), 
     (1,2, get_dist_metric(1,2)), 
     (4,6,get_dist_metric(4,6)), 
     (1,9,get_dist_metric(1,9)), 
     (3,17,get_dist_metric(3,17)), 
     (5,8,get_dist_metric(5,8)), 
     (5,10,get_dist_metric(5,10)), 
     (5,7,get_dist_metric(5,7)), 
     (8,13,get_dist_metric(8,13)), 
     (8,10,get_dist_metric(8,10)), 
     (7,11,get_dist_metric(7,11)), 
     (13,14,get_dist_metric(13,14)), 
     (11,14,get_dist_metric(11,14)), 
     (9,16,get_dist_metric(9,16)), 
     (6,12,get_dist_metric(6,12)), 
     (12,16,get_dist_metric(12,16)), 
     (11,15,get_dist_metric(11,15)), 
     (14,15,get_dist_metric(14,15))]
G.add_weighted_edges_from(e)

#show edge weights in graph when drawn


               
# This is my Dijkstra code
"""
G: networkx graph
node_x: source node
node_y: destination node
"""
def dijkstra_my(G, node_x, node_y=None):
    # distance is list of lists containing smallest from source node
    # to each node in the network
    dist = [None] * len(G.nodes())
    for i in range(len(G.nodes())):
        dist[i] = [float("inf")]
        dist[i].append([node_x])
    # distance from source node to itself
    dist[node_x][0] = 0
    # Queue data structure to keep track of nodes
    queue = [i for i in range(len(G.nodes()))]
    # Set of numbers seen so far
    seen = set()
    # iterate over queue to find the minimum distance
    while len(queue) > 0:
        # if goal node is specified then stop 
        # when min distance to goal node has been found
        if node_y in seen:
            break

        # determine node with the smallest distance from the current node
        min_dist = float("inf")
        min_node = None
        for n in queue: 
            if dist[n][0] < min_dist and n not in seen:
                min_dist = dist[n][0]
                min_node = n
        
        # pop the node with min distance from queue
        # and add it to seen set()
        queue.remove(min_node)
        seen.add(min_node)

        # get nodes connected to min node
        connections = [(n, G[min_node][n]["weight"]) for n in G.neighbors(min_node)]
        # update the distance and the path
        for (node, weight) in connections: 
            tot_dist = weight + min_dist
            if tot_dist < dist[node][0]:
                dist[node][0] = tot_dist
                dist[node][1] = list(dist[min_node][1])
                dist[node][1].append(node)
    return dist  

#input source and destination nodes here

node_x = 0  # source node
node_y = 6  # destination node
dist = dijkstra_my(G, node_x=node_x)

# print the distance from source to every other node
print('The distance returned by Dijkstras is:', dist)

dist = dijkstra_my(G, node_x=node_x, node_y=node_y)
# print the distance from source to destination node
print('The distance returned by Dijkstras is:', dist[node_y])


pos = nx.circular_layout(G)
# nx.draw_networkx_edge_labels(G,pos)
nx.draw_networkx_nodes(G, pos, node_size=300, node_color='orange')
nx.draw_networkx_edges(G, pos, edgelist=e, arrows=False, edge_color= 'grey')
nx.draw_networkx_edges(G, pos, edgelist=[(dist[node_y][1][idx], dist[node_y][1][idx+1], get_dist_metric(idx, idx+1)) 
                                         for idx in range(len(dist[node_y][1]) - 1)], edge_color='red', arrows=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels={(x,y): weight for x,y,weight in G.edges.data("weight")}, font_color='red')
nx.draw_networkx_labels(G, pos, labels=dict(G.nodes(data='name')), font_size=9)
plt.show()
