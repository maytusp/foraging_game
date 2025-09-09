import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# # Number of nodes
# n_nodes = 15

# # --- Ring Structured Network (Light Red Edges) ---
# ring_graph = nx.Graph()
# ring_graph.add_nodes_from(range(1, n_nodes + 1))

# # Connect nodes in a ring
# for i in range(1, n_nodes):
#     ring_graph.add_edge(i, i + 1)
# ring_graph.add_edge(n_nodes, 1)

# # Draw the ring network with light red edges
# plt.figure(figsize=(6, 6))
# angle_ring = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
# pos_octagon_ring = {i + 1: (np.cos(angle_ring[i]), np.sin(angle_ring[i])) for i in range(n_nodes)}
# nx.draw(ring_graph, pos_octagon_ring, with_labels=False, node_size=500, node_color='lightcoral', font_size=10, font_weight='bold', edge_color='lightcoral', width=2)
# plt.title(f"Ring Network ({n_nodes} Nodes, Light Red Edges)")
# plt.axis('equal')
# plt.savefig("social_ring.pdf")
# plt.show()

# # --- Fully Connected Network (Light Red Edges) ---
# fully_connected_graph = nx.Graph()
# fully_connected_graph.add_nodes_from(range(1, n_nodes + 1))

# # Connect all nodes
# for i in range(1, n_nodes + 1):
#     for j in range(i + 1, n_nodes + 1):
#         fully_connected_graph.add_edge(i, j)

# # Draw the fully connected network with light red edges
# plt.figure(figsize=(6, 6))
# angle_full = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
# pos_octagon_full = {i + 1: (np.cos(angle_full[i]), np.sin(angle_full[i])) for i in range(n_nodes)}
# nx.draw(fully_connected_graph, pos_octagon_full, with_labels=False, node_size=500, node_color='lightcoral', font_size=10, font_weight='bold', edge_color='lightcoral', width=1.5)
# plt.title(f"Fully Connected Network ({n_nodes} Nodes, Light Red Edges)")
# plt.axis('equal')
# plt.savefig("social_fc.pdf")
# plt.show()


# Watt-Strogatz Network (k=4 p=0.2)

# --- Parameters ---
n_nodes = 15
possible_pairs_ws = [[0, 1], [0, 2], [0, 13], [0, 14], [1, 2], [1, 3], [1, 14], [2, 3], [2, 4], [3, 4], [3, 5], [4, 5], [4, 6], [5, 6], [5, 7], [6, 7], [6, 8], [7, 8], [7, 9], [8, 9], [8, 10], [9, 10], [9, 11], [10, 11], [10, 12], [11, 12], [11, 13], [12, 13], [12, 14], [13, 14]]
# --- Create Graph from possible_pairs_ws ---
small_world_graph = nx.Graph()
small_world_graph.add_nodes_from(range(1, n_nodes + 1))  # nodes indexed from 1
for u, v in possible_pairs_ws:
    # add 1 to match node indices if your WS code is 0-based
    small_world_graph.add_edge(u + 1, v + 1)

# --- Position nodes in a circle (same as fully connected layout) ---
angle_ws = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
pos_octagon_ws = {i + 1: (np.cos(angle_ws[i]), np.sin(angle_ws[i])) for i in range(n_nodes)}

# --- Draw the Small-World Network ---
plt.figure(figsize=(6, 6))
nx.draw(small_world_graph, pos_octagon_ws, with_labels=False, node_size=500, node_color='lightcoral', font_size=10, font_weight='bold', edge_color='lightcoral', width=1.5)
plt.title(f"Watts–Strogatz Small-World Network ({n_nodes} Nodes)")
plt.axis('equal')
plt.savefig("ws_network_k4p0.png")
plt.show()


# # Optimized Network (k=2, edges=30)
# # --- Parameters ---
# n_nodes = 15
# possible_pairs_ws = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 0], [0, 7], [0, 8], [1, 8], [1, 9], [2, 9], [2, 10], [3, 10], [3, 11], [4, 11], [4, 12], [5, 12], [5, 13], [6, 13], [6, 14], [7, 14]]
# # --- Create Graph from possible_pairs_ws ---
# small_world_graph = nx.Graph()
# small_world_graph.add_nodes_from(range(1, n_nodes + 1))  # nodes indexed from 1
# for u, v in possible_pairs_ws:
#     # add 1 to match node indices if your WS code is 0-based
#     small_world_graph.add_edge(u + 1, v + 1)

# # --- Position nodes in a circle (same as fully connected layout) ---
# angle_ws = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
# pos_octagon_ws = {i + 1: (np.cos(angle_ws[i]), np.sin(angle_ws[i])) for i in range(n_nodes)}

# # --- Draw the Small-World Network ---
# plt.figure(figsize=(6, 6))
# nx.draw(small_world_graph, pos_octagon_ws, with_labels=False, node_size=500, node_color='lightcoral', font_size=10, font_weight='bold', edge_color='lightcoral', width=1.5)
# plt.title(f"Watts–Strogatz Small-World Network ({n_nodes} Nodes)")
# plt.axis('equal')
# plt.savefig("opt_network.pdf")
# plt.show()


# # Circular Clique Network
# # --- Parameters ---
# n_nodes = 15
# possible_pairs_cc = [(3, 4), (12, 13), (0, 2), (8, 9), (9, 11), (0, 14), (13, 14), (6, 8), (4, 5), (5, 6), (0, 1), (9, 10), (1, 2), (10, 11), (6, 7), (3, 5), (12, 14), (2, 3), (11, 12), (7, 8)]
# # --- Create Graph from possible_pairs_ws ---
# small_world_graph = nx.Graph()
# small_world_graph.add_nodes_from(range(1, n_nodes + 1))  # nodes indexed from 1
# for u, v in possible_pairs_cc:
#     # add 1 to match node indices if your WS code is 0-based
#     small_world_graph.add_edge(u + 1, v + 1)

# # --- Position nodes in a circle (same as fully connected layout) ---
# angle_ws = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
# pos_octagon_ws = {i + 1: (np.cos(angle_ws[i]), np.sin(angle_ws[i])) for i in range(n_nodes)}

# # --- Draw the Small-World Network ---
# plt.figure(figsize=(6, 6))
# nx.draw(small_world_graph, pos_octagon_ws, with_labels=False, node_size=500, node_color='lightcoral', font_size=10, font_weight='bold', edge_color='lightcoral', width=1.5)
# plt.title(f"Watts–Strogatz Small-World Network ({n_nodes} Nodes)")
# plt.axis('equal')
# plt.savefig("circular_clique.pdf")
# plt.show()