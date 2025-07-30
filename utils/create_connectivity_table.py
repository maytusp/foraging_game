ws_edges = {'1-3', '9-6', '2-4', '7-10', '10-8', '2-3', '3-4', '3-5', '9-10', '8-10', '0-1', '7-8', '4-6', '11-13', '10-12', '6-7', '12-13', '8-9', '13-6', '5-13', '5-7', '1-2', '12-5', '4-5', '13-5', '0-2', '11-12', '14-1', '14-0'}

opt_edges = {'13-14', '0-7', '6-13', '2-3', '3-4', '3-11', '7-14', '9-10', '2-9', '0-1', '5-6', '0-8', '7-8', '3-10', '2-10', '6-14', '1-9', '4-12', '6-7', '4-11', '12-13', '5-12', '8-9', '1-8', '5-13', '1-2', '4-5', '11-12', '10-11', '14-0'}


# Parse edges into adjacency list
from collections import defaultdict

def create_table(edges):
    adj = defaultdict(set)
    for e in edges:
        u, v = map(int, e.split('-'))
        adj[u].add(v)
        adj[v].add(u)  # undirected

    # Convert adjacency list to a sorted Markdown table
    nodes = sorted(adj.keys())
    print("| Node | Connected To |")
    print("|------|--------------|")
    for node in nodes:
        neighbors = ", ".join(map(str, sorted(adj[node])))
        print(f"| {node:<4} | {neighbors} |")

create_table(ws_edges)
create_table(opt_edges)
