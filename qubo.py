import dimod
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, EmbeddingComposite
import pickle
import json

# Import "map.py" as a module
import map

# Function to convert graph to QUBO
def map_to_qubo(graph: nx.DiGraph):
    safety_zones = []

    for node_id, data in graph.nodes(data=True):
        if 'safe_zone' in data:
            if data['safe_zone'] == 'yes':
                safety_zones.append(node_id)

    # Define binary decision variables, one for each edge
    edge_variables = {}
    edge_strings = {}
    for u, v, data in graph.edges(data=True):
        edge_variables[(u, v)] = f"e_{u}_{v}"
        edge_strings[edge_variables[(u, v)]] = (u, v)
        
    # Define throughput for each variable
    throughput = {}
    for u, v, data in graph.edges(data=True):
        throughput[(u, v)] = data['capacity']

    # Define quadratic coefficients for each pair of variables
    congestion = {}
    for u, v, data in graph.edges(data=True):
        for w, z, data2 in graph.edges(data=True):
            if u == w and v != z: # If the edges start at the same node (less congestion)
                if u in safety_zones:
                    continue
                congestion[(u, v), (w, z)] = - throughput[(u, v)] - throughput[(w, z)]
            elif u != w and v == z: # If the edges end at the same node (more congestion)
                if z in safety_zones:
                    continue
                congestion[(u, v), (w, z)] = throughput[(u, v)] + throughput[(w, z)]
            elif u == z and v != w: # If the edges are connected
                if u in safety_zones:
                    continue
                congestion[(u, v), (w, z)] = - throughput[(u, v)] + throughput[(w, z)]
            elif u != z and v == w: # If the edges are connected
                if v in safety_zones:
                    continue
                congestion[(u, v), (w, z)] = throughput[(u, v)] - throughput[(w, z)]
            else:
                pass
            
    ## Define the QUBO for the main objective function which is to minimize congestion and maximize throughput
    qubo_main = dimod.BinaryQuadraticModel(dimod.BINARY)

    # Throughput terms for each edge
    for (u, v), t in throughput.items():
        qubo_main.set_linear(edge_variables[(u, v)], -t)

    # Congestion terms for each pair of edges
    for ((u, v), (w, z)), c in congestion.items():
        qubo_main.add_interaction(edge_variables[(u, v)], edge_variables[(w, z)], c)

    ## Define the QUBO for the constraint that the solution is exactly V-1 edges
    qubo_spanning_tree = dimod.BinaryQuadraticModel(dimod.BINARY)

    # Constraint term for the number of edges (sum_e(x_e) - (V-1))^2
    # linear term
    for var1 in edge_variables.values():
        qubo_spanning_tree.set_linear(var1, -2 * (len(graph.nodes) - 1) + 1)
        
        for var2 in edge_variables.values():
            if var1 != var2:
                if var1 < var2:
                    qubo_spanning_tree.add_interaction(var1, var2, 1)
                else:
                    qubo_spanning_tree.add_interaction(var2, var1, 1)

    ## Define the QUBO for avoiding selecting edges that have the same nodes
    qubo_no_cycle = dimod.BinaryQuadraticModel(dimod.BINARY)

    # Constraint term (e_i + e_j - 1)^2
    for (u, v), var1 in edge_variables.items():
        for (w, z), var2 in edge_variables.items():
            if u == z and v == w:
                qubo_no_cycle.add_interaction(var1, var2, 1)

    ## Define the QUBO for selecting 1 leaving edge from each node
    qubo_path = dimod.BinaryQuadraticModel(dimod.BINARY)

    # For each node, sum the edges that leave the node
    for node in graph.nodes:
        leaving_edges = [edge_variables[(node, v)] for u, v in graph.out_edges(node)]
        incoming_edges = [edge_variables[(u, node)] for u, v in graph.in_edges(node)]
        
        if node in safety_zones:
            continue

        # if degree of node is 2, then we can skip the rest of the constraints
        if len(incoming_edges) <= 1:
            qubo_path.add_interaction(incoming_edges[0], leaving_edges[0], 1)
            continue


        for edge1 in incoming_edges:
            qubo_path.set_linear(edge1, -1)
            for edge2 in incoming_edges:
                if edge1 != edge2:
                    if edge1 < edge2:
                        qubo_path.add_interaction(edge1, edge2, 1)
                    else:
                        qubo_path.add_interaction(edge2, edge1, 1)

        for edge1 in leaving_edges:
            qubo_path.set_linear(edge1, -1)
            for edge2 in leaving_edges:
                if edge1 != edge2:
                    if edge1 < edge2:
                        qubo_path.add_interaction(edge1, edge2, 1)
                    else:
                        qubo_path.add_interaction(edge2, edge1, 1)

    # Combine the QUBOs

    # qubo_complete = dimod.quicksum([qubo_main, 17 * qubo_path, 17 * qubo_spanning_tree, 17 * qubo_no_cycle])
    qubo_complete = dimod.quicksum([qubo_main, 68 * qubo_path, 136 * qubo_no_cycle])

    # qubo_complete = dimod.quicksum([qubo_path, 2 * qubo_no_cycle])

    return qubo_complete, edge_strings

def edges_from_solution(sample, edge_strings):
    edges = []
    for edge, value in sample.items():
        if value == 1:
            edges.append(edge_strings[edge])

    return edges
    

# Main function
if __name__ == "__main__":
    location_name = "Rafah, Gaza Strip, Palestinian Territories"
    # location_name = "Abu Dhabi, UAE"

    # Load the graph
    graph = map.analyze_and_plot_road_network(location_name)

    # Convert the graph to a QUBO
    qubo, edge_strings = map_to_qubo(graph)

    # Solve the QUBO
    # sampler = SimulatedAnnealingSampler()
    # solver_type = "simulated_annealing"
    sampler = EmbeddingComposite(DWaveSampler())
    solver_type = "quantum"
    response = sampler.sample(qubo, num_reads=100)

    # Print sum(e_i) for the solution
    print("Number of nodes:", len(graph.nodes))
    print("Number of edges selected:", sum(response.first.sample.values()))
    print("Energy:", response.first.energy)


    edges = edges_from_solution(response.first.sample, edge_strings)
    # Print the results
    print(edges)

    # Create graph from edges
    graph_solution = nx.DiGraph()
    graph_solution.add_edges_from(edges)
    graph_solution = graph_solution.to_undirected()

    print(graph_solution.nodes())

    safety_zones = []

    for node_id, data in graph.nodes(data=True):
        if 'safe_zone' in data:
            if data['safe_zone'] == 'yes':
                safety_zones.append(node_id)

    dict_evacuation = {}
    for node in graph_solution.nodes:
        shortest_length = None
        shortest_path = None

        # find path to nearest safe zone using shortest path with "length" as weight
        for safe_node in safety_zones:
            ## Check if a path exists
            if not nx.has_path(graph_solution, node, safe_node):
                # print(f"No path from {node} to {safe_node}")
                continue
            shortest_path_test = nx.shortest_path(graph_solution, source=node, target=safe_node, weight='length')

            # Calculate the length of the path
            length =  nx.shortest_path_length(graph_solution, source=node, target=safe_node, weight='length')
            if (shortest_length is None) or (length < shortest_length):
                shortest_length = length
                shortest_path = shortest_path_test

        if shortest_path is None:
            ## Use graph to find the shortest path
            for safe_node in safety_zones:
                ## Check if a path exists
                if not nx.has_path(graph, node, safe_node):
                    # print(f"No path from {node} to {safe_node}")
                    continue
                shortest_path_test = nx.shortest_path(graph, source=node, target=safe_node)

                # Calculate the length of the path
                length =  nx.shortest_path_length(graph, source=node, target=safe_node)
                if (shortest_length is None) or (length < shortest_length):
                    shortest_length = length
                    shortest_path = shortest_path_test

        # store
        dict_evacuation[node] = shortest_path

    print(dict_evacuation)

    # Store the results in a file using pickle
    with open(f"solution_{solver_type}.pkl", "wb") as f:
        pickle.dump(graph_solution, f)
        pickle.dump(graph, f)
        pickle.dump(dict_evacuation, f)
        
    # Store the coordinates of paths and safety zones in a json file
    dict_coordinates = {}
    for node in graph_solution.nodes:
        path = dict_evacuation[node]
        coordinates = [] # list of tuples (x, y)
        for node_id in path:
            coordinates.append((graph.nodes[node_id]['x'], graph.nodes[node_id]['y']))

        dict_node = {
            'coordinates': coordinates,
            'is_safe': 'yes!' if node in safety_zones else 'no, get out please!'
        }

        dict_coordinates[f"{node}"] = dict_node

    print(dict_coordinates)

    with open(f"{location_name}_coordinates_{solver_type}.json", "w") as f:
        json.dump(dict_coordinates, f)

    fig = map.plot_graph(graph, location_name)

    pos = {node: (data['x'], data['y']) for node, data in graph.nodes(data=True)}
    edge_x = []
    edge_y = []
    for edge in graph_solution.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='red'),
        hoverinfo='none',
        mode='lines')
    
    fig.add_trace(edge_trace)

    fig.write_html(f"{location_name}_{solver_type}.html", include_plotlyjs='cdn', full_html=False)

    # Repeat but for MST
    undirected_graph = graph.to_undirected()

    # Compute the minimum spanning tree of the undirected graph
    mst_graph = nx.minimum_spanning_tree(undirected_graph, weight='capacity')

    edges = mst_graph.edges(data=True)

    fig = map.plot_graph(graph, location_name)

    pos = {node: (data['x'], data['y']) for node, data in graph.nodes(data=True)}
    edge_x = []
    edge_y = []
    for edge in mst_graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='red'),
        hoverinfo='none',
        mode='lines')
    
    fig.add_trace(edge_trace)

    fig.write_html(f"{location_name}_mst.html", include_plotlyjs='cdn', full_html=False)