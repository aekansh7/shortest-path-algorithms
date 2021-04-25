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
    edge_weight = edge_weights.get((node1, node2))
    if not edge_weight:
        edge_weight = edge_weights.get((node2,node1))
    return edge_weight


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
    return 0

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


def a_star_search(G, node_x, node_y):
    
    dist = [None] * len(G.nodes())
    for i in range(len(G.nodes())):
        dist[i] = [float("inf")]
        dist[i].append([node_x])
    
    dist[node_x][0] = 0
    
    frontier = PriorityQueue()
    frontier.put(node_x, 0)
    # Set of numbers seen so far
    seen = set()
    while not frontier.empty():
        if node_y in seen:
            break
       
        
        min_node = frontier.get()
        min_dist = dist[min_node][0]
        print(f"Min node: {min_node}")

        seen.add(min_node)
        
        connections = [(n, G[min_node][n]["weight"]) for n in G.neighbors(min_node)]
    
        for (node, weight) in connections: 
            tot_dist = weight + min_dist
            if tot_dist < dist[node][0] and node not in seen:
                dist[node][0] = tot_dist
                dist[node][1] = list(dist[min_node][1])
                dist[node][1].append(node)
                frontier.put(node, tot_dist + get_heuristic(node, node_y))
    return dist  

node_x = 4
node_y = 0

dist = a_star_search(G, node_x=node_x, node_y=node_y)
print('The distance returned by A* is:', dist[node_y])

pos = nx.circular_layout(G)
# nx.draw_networkx_edge_labels(G,pos)
nx.draw_networkx_nodes(G, pos, node_size=400, node_color='pink')
nx.draw_networkx_edges(G, pos, edgelist=e, edge_color='grey')
nx.draw_networkx_edges(G, pos, edgelist=[(dist[node_y][1][idx], dist[node_y][1][idx+1], get_dist_metric(dist[node_y][1][idx], dist[node_y][1][idx+1])) 
                                         for idx in range(len(dist[node_y][1]) - 1)], edge_color='red', arrows=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels={(x,y): weight for x,y,weight in G.edges.data("weight")}, font_color='red')
nx.draw_networkx_labels(G, pos, labels=dict(G.nodes(data='name')), font_size=9)
plt.show()
