import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import json

# Specify the location name
place_name = "Musaffah, Abu Dhabi, United Arab Emirates"

# Fetch OSM street network from the location
# Customize filter to include only highways
custom_filter = '["highway"]["area"!~"yes"]["highway"~"motorway|trunk|primary|secondary|tertiary"]'
graph = ox.graph_from_place(place_name, network_type='drive', custom_filter=custom_filter)

# Simplify graph to merge nodes
#graph = ox.simplify_graph(graph)

# Plot the street network and save to a file
fig, ax = ox.plot_graph(ox.project_graph(graph))
plt.savefig('highway_network_plot.png')



##plt.close(fig)

# Save the graph to a GraphML file
ox.save_graphml(graph, filepath='highway_network.graphml')

# Read the graph from GraphML
g = nx.read_graphml('highway_network.graphml')

# Optionally, convert the graph to JSON format
data_json = json.dumps(nx.readwrite.json_graph.node_link_data(g))
with open('highway_network.json', 'w') as f:
    f.write(data_json)

# To prepare for max flow analysis:
# Add capacity based on number of lanes or other suitable properties if available
for u, v, d in g.edges(data=True):
    # You might need to adjust this depending on what data is actually available in your GraphML
    d['capacity'] = d.get('lanes', 1)  # Default to 1 if lane data is not available

# Use networkx to find maximum flow between two nodes (example):
# You need to specify source and target nodes
## source, target = list(g.nodes())[0], list(g.nodes())[-1]
##flow_value, flow_dict = nx.maximum_flow(g, source, target, capacity='capacity')
##print("Max flow from node {} to node {}: {}".format(source, target, flow_value))
