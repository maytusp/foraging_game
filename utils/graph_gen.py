import random
from collections import deque
import numpy as np
import math
from itertools import combinations
import random

def compute_all_pairs_shortest_paths(num_nodes, edges):
    """
    Compute shortest path lengths (social distances) between all pairs of nodes.
    Returns a matrix distances[i][j] = shortest path length from i to j.
    """
    # Build adjacency list
    graph = {i: [] for i in range(num_nodes)}
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)  # assuming undirected graph

    # Initialize distance matrix
    distances = [[float('inf')] * num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes):
        distances[i][i] = 0

    # BFS for each node to compute shortest paths
    for start in range(num_nodes):
        queue = deque([start])
        visited = {start}
        while queue:
            current = queue.popleft()
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[start][neighbor] = distances[start][current] + 1
                    queue.append(neighbor)

    return distances

def watts_strogatz_pairs(num_nodes, k=4, p=0.2):
    """
    Generate edges for a small-world network (Watts-Strogatz-like).
    
    num_nodes: number of nodes (agents)
    k: each node is connected to k neighbors on each side (k must be even)
    p: probability of rewiring an edge to a random node
    """
    edges = []
    
    # Start with ring lattice: connect k neighbors on each side
    for i in range(num_nodes):
        for j in range(1, k//2 + 1):
            edges.append([i, (i + j) % num_nodes])
    
    # Rewire edges with probability p
    for edge in edges.copy():
        if random.random() < p:
            u = edge[0]
            v = random.choice([n for n in range(num_nodes) if n != u])
            edges.remove(edge)
            edges.append([u, v])
    
    return edges


def optimized_small_world(num_nodes, k=None, max_edges=None):
    """
    Greedy small-world optimization: reduce path length under cost constraint.
    """
    
    # Initial ring lattice
    edges = [[i, (i+j) % num_nodes] for i in range(num_nodes) for j in range(1, k//2 + 1)]
    
    if max_edges is None:
        max_edges = len(edges) + num_nodes  # allow extra shortcuts
    
    # Distance metric (assuming nodes on a circle)
    def distance(u, v):
        return min(abs(u-v), num_nodes - abs(u-v))
    
    while len(edges) < max_edges:
        best_edge = None
        best_gain = 0
        
        # Try adding edges that give best ratio: path length reduction / distance cost
        for u, v in combinations(range(num_nodes), 2):
            if [u, v] in edges or [v, u] in edges or u == v:
                continue
            # Approximate gain: prefer distant nodes
            gain = distance(u, v)  # heuristic: prefer longer shortcuts
            cost = 1.0 / (1 + gain)
            ratio = gain / cost
            if ratio > best_gain:
                best_gain = ratio
                best_edge = (u, v)
        
        if best_edge:
            edges.append(list(best_edge))
        else:
            break
    
    return edges

def get_adjacency_matrix(connected_nodes, num_nodes):
    '''
    Input: A list of connected nodes in the graph
    Output: Adjacency Matrix
    '''
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for pairs in connected_nodes:
        i, j = pairs
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1
    return adj_matrix

# possible_pairs_ws = watts_strogatz_pairs(15, k=4, p=0.2)
possible_pairs_opt = optimized_small_world(15, k=2, max_edges=30)
distances = compute_all_pairs_shortest_paths(15, possible_pairs_opt)

# print(possible_pairs_ws)
print(possible_pairs_opt)
# # Print the distance matrix
for row in distances:
    print(row)

WS_PAIRS = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [3, 5], [4, 5], [4, 6], [5, 7], [6, 7], [7, 8], [8, 9], [8, 10], [9, 10], [10, 12], [11, 12], [11, 13], [12, 13], [14, 0], [14, 1], [5, 13], [6, 7], [7, 10], [9, 6], [10, 8], [12, 5], [13, 6], [13, 5]]
OPT_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 0], [0, 7], [0, 8], [1, 8], [1, 9], [2, 9], [2, 10], [3, 10], [3, 11], [4, 11], [4, 12], [5, 12], [5, 13], [6, 13], [6, 14], [7, 14]]

WS_PAIRS = set([f"{pair[0]}-{pair[1]}" for pair in WS_PAIRS])
OPT_PAIRS = set([f"{pair[0]}-{pair[1]}" for pair in OPT_PAIRS])
print(f"Watt-Strogatz: {WS_PAIRS}")
print(f"Greedy: {OPT_PAIRS}")
# print(f"Watt-Strogatz: \n {get_adjacency_matrix(WS_PAIRS, 15)}")
# print(f"Greedy: \n {get_adjacency_matrix(OPT_PAIRS, 15)}")
