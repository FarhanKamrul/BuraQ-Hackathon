import osmnx as ox
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import xml.etree.ElementTree as ET

def analyze_and_plot_road_network(place_name):
    custom_filter = '["highway"]["area"!~"yes"]["highway"~"motorway|trunk|primary|secondary|tertiary"]'
    graph_raw = ox.graph_from_place(place_name, network_type='drive', custom_filter=custom_filter)
    graph = ox.consolidate_intersections(ox.project_graph(graph_raw), tolerance=100, rebuild_graph=True, dead_ends=False)
    for u, data in graph.nodes(data=True):
        nodes = graph_raw.nodes(data=True)

        # parse 'osmid_original' as a list of integers
        if type(data['osmid_original']) != int:
            data['osmid_original'] = [int(osmid) for osmid in data['osmid_original'].replace("[", "").replace("]", "").replace(" ", "").split(",")]
        else:
            data['osmid_original'] = [data['osmid_original']]

        ## Replace x and y with average x and y from osmid_original nodes in graph_raw
        graph.nodes[u]['x'] = np.mean([ nodes[osmid]['x'] for osmid in data['osmid_original'] ])
        graph.nodes[u]['y'] = np.mean([ nodes[osmid]['y'] for osmid in data['osmid_original'] ])

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

    # Select multiple random nodes to mark as 'safe zones'
    safe_nodes = np.random.choice(list(graph.nodes()), size=4, replace=False)
    for node in graph.nodes():
        graph.nodes[node]['safe_zone'] = 'no'
    for node in safe_nodes:
        graph.nodes[node]['safe_zone'] = 'yes'

    # Plot with Plotly
    fig = plot_graph(graph, place_name)
    html_filename = f"{place_name.replace(' ', '_')}_graph.html"
    fig.write_html(html_filename, include_plotlyjs='cdn', full_html=False)
    # Save to PNG for reporting
    # fig.write_image(f"{place_name.replace(' ', '_')}_graph.png")

    # Convert to a directed graph
    digraph = nx.DiGraph(graph)

    # For each edge, check if the reverse edge exists and if not, add it
    for u, v, d in list(digraph.edges(data=True)):
        if u == v:
            print(f"Self-loop detected: {u}")
            ## Remove self-loops
            digraph.remove_edge(u, v)
            continue
        if not digraph.has_edge(v, u):
            digraph.add_edge(v, u, **d)

    create_osm_file(graph, f"{place_name.replace(' ', '_')}_graph.osm")
    return digraph

def plot_graph(graph, place_name):
    pos = {node: (data['x'], data['y']) for node, data in graph.nodes(data=True)}
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_size = []
    node_color = []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        if graph.nodes[node]['safe_zone'] == 'yes':
            node_size.append(20)  # Larger size for safe zones
            node_color.append('red')
        else:
            node_size.append(10)
            node_color.append('blue')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line=dict(color='black', width=2)
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f"Network Graph of {place_name}",
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

def create_osm_file(graph, filename):
    osm = ET.Element("osm", version="0.6", generator="OSMnx")
    for node_id, data in graph.nodes(data=True):
        node = ET.SubElement(osm, "node", id=str(node_id), visible="true", lat=str(data['y']), lon=str(data['x']))
        for key, value in data.items():
            if key not in ['x', 'y']:
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
graph = analyze_and_plot_road_network("Rafah, Palestinian Territories")
