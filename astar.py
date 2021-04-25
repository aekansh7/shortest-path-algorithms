import networkx as nx
import matplotlib.pyplot as plt
from queue import PriorityQueue

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
                    (4,5): 0.32,
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
def get_heuristic(node, node_y):
    if node == node_y:
        return 0
    values = {(0,1): 0.525,
                 (0,2): 0.77,
                 (0,3): 0.48,
                 (0,4): 0.675,
                 (0,5): 0.56,
                 (1,2): 1.29,
                 (1,3): 1.02,
                 (1,4): 1.21,
                 (1,5): 1.13,
                 (2,3): 0.335,
                 (2,4): 0.235,
                 (2,5): 0.368,
                 (3,4): 0.16,
                 (3,5): 0.238,
                 (4,5): 0.2,
                 }
    heuristic = values.get((node, node_y))
    if not heuristic:
        heuristic = values.get((node_y, node))
    return heuristic

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
     (4,5,get_dist_metric(4,5))]
G.add_weighted_edges_from(e)

# A* search algo
def a_star_search(G, node_x, node_y):
 # distance is list of lists containing smallest from source node
 # to each node in the network   
    dist = [None] * len(G.nodes())
    for i in range(len(G.nodes())):
        dist[i] = [float("inf")]
        dist[i].append([node_x])
    # distance from source node to itself
    dist[node_x][0] = 0
    
# PriorityQueue data structure to keep track of nodes
    frontier = PriorityQueue()
    frontier.put(node_x, 0)
    # Set of numbers seen so far
    seen = set()

    # iterate over queue to find the minimum distance
    while not frontier.empty():
        # if goal node is specified then stop 
        # when min distance to goal node has been found
        if node_y in seen:
            break
        # determine node with the smallest distance from the current node
        min_node = frontier.get()
        min_dist = dist[min_node][0]
        print(f"Min node: {min_node}")
        # add min node to the seen set
        seen.add(min_node)
        # Get nodes connected to min node 
        connections = [(n, G[min_node][n]["weight"]) for n in G.neighbors(min_node)]
        # update the distance and the path using the heuristic
        for (node, weight) in connections: 
            tot_dist = weight + min_dist
            if tot_dist < dist[node][0] and node not in seen:
                dist[node][0] = tot_dist
                dist[node][1] = list(dist[min_node][1])
                dist[node][1].append(node)
                frontier.put(node, tot_dist + get_heuristic(node, node_y))
    return dist  

#input source and destination nodes here
node_x = 0 #source node
node_y = 4 #destination node

dist = a_star_search(G, node_x=node_x, node_y=node_y)
# print the distance from source to destination node
print('The distance returned by A* is:', dist[node_y])

pos = nx.circular_layout(G)
# nx.draw_networkx_edge_labels(G,pos)
nx.draw_networkx_nodes(G, pos, node_size=400, node_color='pink')
nx.draw_networkx_edges(G, pos, edgelist=e, edge_color='grey')
nx.draw_networkx_edges(G, pos, edgelist=[(dist[node_y][1][idx], dist[node_y][1][idx+1], get_dist_metric(idx, idx+1)) 
                                         for idx in range(len(dist[node_y][1]) - 1)], edge_color='red', arrows=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels={(x,y): weight for x,y,weight in G.edges.data("weight")}, font_color='red')
nx.draw_networkx_labels(G, pos, labels=dict(G.nodes(data='name')), font_size=9)
plt.show()
