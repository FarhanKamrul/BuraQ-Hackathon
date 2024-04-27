import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

def analyze_and_plot_road_network(place_name):
    # Customize filter to include only certain types of highways
    custom_filter = '["highway"]["area"!~"yes"]["highway"~"motorway|trunk|primary|secondary|tertiary"]'
    graph = ox.graph_from_place(place_name, network_type='drive', custom_filter=custom_filter)

    # Simplify graph to merge nodes and consolidate intersections
    graph = ox.consolidate_intersections(ox.project_graph(graph), tolerance=100, rebuild_graph=True, dead_ends=False)

    # Define safety values and calculate capacities
    safety_values = {'motorway': 0.9, 'trunk': 0.8, 'primary': 0.6, 'secondary': 0.4, 'tertiary': 0.2}

    for u, v, d in graph.edges(data=True):
        road_type = d.get('highway', 'unknown')
        if isinstance(road_type, list):
            road_type = road_type[0]
        safety_value = safety_values.get(road_type, 0.5)

        lanes = d.get('lanes', '1')
        if isinstance(lanes, list):
            lanes = lanes[0]
        if isinstance(lanes, str):
            lanes = lanes.split(";")[0]
        lanes = int(lanes)

        length = float(d['length'])
        d['capacity'] = (safety_value * 50) + (10 * lanes) - 2 * (length / 1000)

    # Plotting with background map and increased node size
    ec = ox.plot.get_edge_colors_by_attr(graph, attr='capacity', cmap='viridis', num_bins=20)
    fig, ax = ox.plot_graph(graph, edge_color=ec, edge_linewidth=2, node_size=30, node_color='#66ccff', bgcolor='k')
    plt.show()

    # Convert to a directed graph
    digraph = nx.DiGraph(graph)

    

    # Save the graph to an OSM file
    create_osm_file(graph, f"{place_name.replace(' ', '_')}_graph.osm")

    # Return the directed graph
    return digraph

def create_osm_file(graph, filename):
    osm = ET.Element("osm", version="0.6", generator="OSMnx")
    for node_id, data in graph.nodes(data=True):
        node = ET.SubElement(osm, "node", id=str(node_id), visible="true", lat=str(data['y']), lon=str(data['x']))
        for key, value in data.items():
            if key not in ['x', 'y']:  # Skip coordinates which are attributes of the node element
                tag = ET.SubElement(node, "tag", k=key, v=str(value))

    for u, v, data in graph.edges(data=True):
        way = ET.SubElement(osm, "way", id=f"{u}_{v}", visible="true")
        nd1 = ET.SubElement(way, "nd", ref=str(u))
        nd2 = ET.SubElement(way, "nd", ref=str(v))
        for key, value in data.items():
            tag = ET.SubElement(way, "tag", k=key, v=str(value))

    tree = ET.ElementTree(osm)
    tree.write(filename, encoding='utf-8', xml_declaration=True)

# Example usage:
graph = analyze_and_plot_road_network("Rafah, Gaza Strip, Palestinian Territories")
# Now 'graph' holds the directed graph which can be used for further analysis or manipulation.
