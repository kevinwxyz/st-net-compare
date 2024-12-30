import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from io import StringIO

# Helper Function: Compute Network Metrics
def compute_metrics(G):
    degrees = [val for (_, val) in G.degree()]
    weights = [attr['weight'] for _, _, attr in G.edges(data=True) if 'weight' in attr]
    clustering_coeffs = nx.clustering(G, weight='weight')
    density = nx.density(G)
    global_clustering = nx.average_clustering(G, weight='weight')
    
    # Community detection using greedy modularity
    from networkx.algorithms.community import greedy_modularity_communities
    communities = list(greedy_modularity_communities(G, weight='weight'))
    modularity = nx.algorithms.community.modularity(G, communities, weight='weight')
    
    metrics = {
        "Number of Nodes": G.number_of_nodes(),
        "Number of Edges": G.number_of_edges(),
        "Density": density,
        "Global Clustering Coefficient": global_clustering,
        "Modularity": modularity,
        "Average Degree": sum(degrees) / len(degrees) if degrees else 0,
        "Average Edge Weight": sum(weights) / len(weights) if weights else 0
    }
    
    return metrics, degrees, weights, clustering_coeffs, communities

# Helper Functions: Plot Metrics
def plot_degree_vs_clustering(degrees, clustering_coeffs):
    fig, ax = plt.subplots()
    ax.scatter(degrees, clustering_coeffs.values(), color='blue', alpha=0.7)
    ax.set_title("Degree vs Clustering Coefficient")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Clustering Coefficient")
    return fig

def plot_positive_vs_negative(weights):
    positive_weights = [w for w in weights if w > 0]
    negative_weights = [w for w in weights if w < 0]
    
    fig, ax = plt.subplots()
    sns.kdeplot(positive_weights, label="Positive Weights", color="green", ax=ax)
    sns.kdeplot(negative_weights, label="Negative Weights", color="red", ax=ax)
    ax.set_title("Positive vs Negative Edge Weight Distribution")
    ax.set_xlabel("Edge Weight")
    ax.set_ylabel("Density")
    ax.legend()
    return fig

def plot_metric_distribution(degrees, weights):
    # Degree Distribution Plot
    fig1, ax1 = plt.subplots()
    sns.histplot(degrees, kde=True, bins=15, ax=ax1, color='skyblue')
    ax1.set_title("Degree Distribution")
    ax1.set_xlabel("Degree")
    ax1.set_ylabel("Count")
    
    # Edge Weight Distribution Plot
    fig2, ax2 = plt.subplots()
    sns.histplot(weights, kde=True, bins=15, ax=ax2, color='lightcoral')
    ax2.set_title("Edge Weight Distribution")
    ax2.set_xlabel("Edge Weight")
    ax2.set_ylabel("Count")
    
    return fig1, fig2

# Streamlit App
st.title("Network Comparison App")

# Upload GraphML files
uploaded_file_1 = st.file_uploader("Upload the first GraphML file", type=["graphml"])
uploaded_file_2 = st.file_uploader("Upload the second GraphML file", type=["graphml"])

if uploaded_file_1 and uploaded_file_2:
    # Load the graphs
    G1 = nx.read_graphml(uploaded_file_1)
    G2 = nx.read_graphml(uploaded_file_2)
    
    # Compute metrics for both networks
    metrics1, degrees1, weights1, clustering_coeffs1, communities1 = compute_metrics(G1)
    metrics2, degrees2, weights2, clustering_coeffs2, communities2 = compute_metrics(G2)
    
    # Display Metrics
    st.header("Network Metrics Comparison")
    metrics_df = pd.DataFrame([metrics1, metrics2], index=["Network 1", "Network 2"])
    st.dataframe(metrics_df)
    
    # Generate Plots
    st.header("Plots")
    
    st.subheader("Degree Distribution")
    fig1a, fig2a = plot_metric_distribution(degrees1, weights1)
    fig1b, fig2b = plot_metric_distribution(degrees2, weights2)
    
    st.write("Network 1")
    st.pyplot(fig1a)
    st.pyplot(fig2a)
    
    st.write("Network 2")
    st.pyplot(fig1b)
    st.pyplot(fig2b)
    
    st.subheader("Degree vs Clustering Coefficient")
    fig_deg_clust_1 = plot_degree_vs_clustering(degrees1, clustering_coeffs1)
    fig_deg_clust_2 = plot_degree_vs_clustering(degrees2, clustering_coeffs2)
    st.write("Network 1")
    st.pyplot(fig_deg_clust_1)
    st.write("Network 2")
    st.pyplot(fig_deg_clust_2)
    
    st.subheader("Positive vs Negative Edge Weight Distribution")
    fig_pos_neg_1 = plot_positive_vs_negative(weights1)
    fig_pos_neg_2 = plot_positive_vs_negative(weights2)
    st.write("Network 1")
    st.pyplot(fig_pos_neg_1)
    st.write("Network 2")
    st.pyplot(fig_pos_neg_2)
    
    # Communities visualization
    st.subheader("Community Structure")
    
    def get_node_colors_from_communities(communities, G):
        # Create a dictionary mapping each node to its community index
        community_dict = {}
        for i, community in enumerate(communities):
            for node in community:
                community_dict[node] = i
        
        # Assign a color index to each node based on its community
        node_colors = [community_dict[node] for node in G.nodes()]
        return node_colors
    
    # Get node colors for both networks
    node_colors1 = get_node_colors_from_communities(communities1, G1)
    node_colors2 = get_node_colors_from_communities(communities2, G2)
    
    fig_com, ax_com = plt.subplots(1, 2, figsize=(14, 7))
    pos1 = nx.spring_layout(G1)  # Spring layout for G1
    pos2 = nx.spring_layout(G2)  # Spring layout for G2
    
    # Draw Network 1 with community-based coloring
    nx.draw(G1, pos1, ax=ax_com[0], node_color=node_colors1, cmap='viridis', node_size=50)
    ax_com[0].set_title("Network 1 Communities")
    
    # Draw Network 2 with community-based coloring
    nx.draw(G2, pos2, ax=ax_com[1], node_color=node_colors2, cmap='viridis', node_size=50)
    ax_com[1].set_title("Network 2 Communities")
    
    st.pyplot(fig_com)

    from collections import Counter
    import matplotlib.pyplot as plt
    import random
    
    # Function to compute module-specific metrics
    def compute_module_metrics(graph, community):
        metrics = []
        for i, module in enumerate(community):
            subgraph = graph.subgraph(module)
            num_nodes = subgraph.number_of_nodes()
            num_edges = subgraph.number_of_edges()
            density = nx.density(subgraph)
            avg_clustering = nx.average_clustering(subgraph)
            
            # Modularity for the module as a subgraph
            if num_nodes > 1:  # Modularity isn't defined for a single node
                partition = {node: 0 for node in subgraph.nodes()}
                modularity = nx.algorithms.community.quality.modularity(subgraph, [set(partition.keys())])
            else:
                modularity = None  # Not defined
    
            metrics.append({
                "Module ID": i + 1,
                "Nodes": num_nodes,
                "Edges": num_edges,
                "Density": density,
                "Avg Clustering": avg_clustering,
                "Modularity": modularity
            })
        return metrics
    
    # Function to visualize a network with modules
    def visualize_network_with_modules(graph, communities, module_selection=None):
        pos = nx.spring_layout(graph, seed=42)
        colors = plt.cm.tab10.colors
        color_mapping = {i: colors[i % len(colors)] for i in range(len(communities))}
    
        plt.figure(figsize=(10, 7))
        for i, module in enumerate(communities):
            if module_selection is None or i + 1 in module_selection:
                nx.draw_networkx_nodes(
                    graph,
                    pos,
                    nodelist=module,
                    node_color=[color_mapping[i]],
                    label=f"Module {i + 1}",
                    node_size=50
                )
    
        nx.draw_networkx_edges(graph, pos, alpha=0.5)
        plt.legend(loc="best")
        plt.title("Network Visualization with Modules")
        plt.axis("off")
        st.pyplot(plt)

    # def visualize_network_with_modules(graph, module_attribute="module", selected_modules=None):
    #     """
    #     Visualizes a network, allowing for module filtering and interactive taxa name display.
    
    #     Parameters:
    #     - graph (networkx.Graph): The input graph.
    #     - module_attribute (str): The node attribute representing the module assignment.
    #     - selected_modules (list of int): The list of module IDs to include. If None, include all modules.
    
    #     Returns:
    #     - Pyvis network HTML file path
    #     """
    #     from pyvis.network import Network
        
    #     # Initialize Pyvis network
    #     net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="black")
    #     net.toggle_physics(True)  # Enable physics
        
    #     # Filter nodes and edges based on selected modules
    #     if selected_modules is not None:
    #         filtered_nodes = [node for node, data in graph.nodes(data=True) if data.get(module_attribute) in selected_modules]
    #         filtered_edges = [
    #             (u, v, data) for u, v, data in graph.edges(data=True) if u in filtered_nodes and v in filtered_nodes
    #         ]
    #     else:
    #         filtered_nodes = list(graph.nodes)
    #         filtered_edges = list(graph.edges(data=True))
        
    #     # Add filtered nodes to the Pyvis network
    #     for node in filtered_nodes:
    #         data = graph.nodes[node]
    #         taxa_name = data.get("name", str(node))  # Get node name or fallback to node ID
    #         module_id = data.get(module_attribute, "Unknown")  # Get module ID
    #         net.add_node(node, label=taxa_name, title=f"Taxa: {taxa_name}<br>Module: {module_id}")
        
    #     # Add filtered edges to the Pyvis network
    #     for u, v, data in filtered_edges:
    #         weight = data.get("weight", 0)  # Default to 0 if weight attribute not available
    #         net.add_edge(u, v, value=weight)
        
    #     # Save visualization as an HTML file
    #     html_file = "/tmp/module_filtered_network.html"
    #     net.save_graph(html_file)
    #     return html_file

    # # Function to visualize a network with selected modules using the communities list
    # def visualize_network_with_modules(graph, communities, selected_modules=None):
    #     net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="black")
    #     net.toggle_physics(True)
    
    #     # Determine nodes and edges for selected modules
    #     if selected_modules:
    #         selected_nodes = set()
    #         for i in selected_modules:
    #             selected_nodes.update(communities[i - 1])  # Subtracting 1 since selected_modules is 1-indexed
            
    #         filtered_nodes = [node for node in graph.nodes if node in selected_nodes]
    #         filtered_edges = [
    #             (u, v, data) for u, v, data in graph.edges(data=True) if u in selected_nodes and v in selected_nodes
    #         ]
    #     else:
    #         # If no module is selected, include all nodes and edges
    #         filtered_nodes = list(graph.nodes)
    #         filtered_edges = list(graph.edges(data=True))
        
    #     # Add nodes and edges to the Pyvis network
    #     for node in filtered_nodes:
    #         data = graph.nodes[node]
    #         taxa_name = data.get("name", str(node))
    #         net.add_node(node, label=taxa_name, title=f"Taxa: {taxa_name}")
        
    #     for u, v, data in filtered_edges:
    #         weight = data.get("weight", 0)
    #         net.add_edge(u, v, value=weight)
    
    #     # Save and return the HTML file
    #     html_file = "/tmp/module_filtered_network.html"
    #     net.save_graph(html_file)
    #     return html_file




    
    # Module statistics and visualizations for Network 1
    st.subheader("Module Statistics and Visualization for Network 1")
    module_metrics1 = compute_module_metrics(G1, communities1)
    num_modules1 = len(communities1)
    st.write(f"Number of modules in Network 1: {num_modules1}")
    
    module_metrics1_sorted = sorted(module_metrics1, key=lambda x: -x["Nodes"])[:5]
    # Convert the metrics to a DataFrame for better visualization
    module_metrics1_df = pd.DataFrame(module_metrics1_sorted)
    st.write("Module Metrics (Top 5 Largest Modules in Network 1):")
    st.dataframe(module_metrics1_df)
    # st.write(module_metrics1_sorted)
    
    # Interactive selection for module visualization
    selected_modules1 = st.multiselect(
        "Select modules to visualize (Network 1):",
        options=range(1, num_modules1 + 1),
        default=[1]  # Default to the largest module
    )
    # selected_modules1 = st.sidebar.multiselect(
    #     "Select Modules to Visualize:",
    #     options=list(set(nx.get_node_attributes(G1, "module").values())),
    #     default=[]
    # )


    
    visualize_network_with_modules(G1, communities1, module_selection=selected_modules1)
    # if selected_modules1:
    #     html_file1 = visualize_network_with_modules(G1, communities1, selected_modules=selected_modules1)
    #     st.components.v1.html(open(html_file1, "r").read(), height=750)
    # else:
    #     st.write("Please select at least one module to visualize.")



    
    # Module statistics and visualizations for Network 2
    st.subheader("Module Statistics and Visualization for Network 2")
    module_metrics2 = compute_module_metrics(G2, communities2)
    num_modules2 = len(communities2)
    st.write(f"Number of modules in Network 2: {num_modules2}")
    
    module_metrics2_sorted = sorted(module_metrics2, key=lambda x: -x["Nodes"])[:5]
    # Convert the metrics to a DataFrame for Network 2
    module_metrics2_df = pd.DataFrame(module_metrics2_sorted)
    st.write("Module Metrics (Top 5 Largest Modules in Network 2):")
    st.dataframe(module_metrics2_df)
    # st.write(module_metrics2_sorted)
    
    # Interactive selection for module visualization
    selected_modules2 = st.multiselect(
        "Select modules to visualize (Network 2):",
        options=range(1, num_modules2 + 1),
        default=[1]  # Default to the largest module
    )
    # selected_modules2 = st.sidebar.multiselect(
    #     "Select Modules to Visualize:",
    #     options=list(set(nx.get_node_attributes(G2, "module").values())),
    #     default=[]
    # )



    
    visualize_network_with_modules(G2, communities2, module_selection=selected_modules2)
    # if selected_modules:
    #     html_file1 = visualize_network_with_modules(G2, communities2, selected_modules=selected_modules2)
    #     st.components.v1.html(open(html_file1, "r").read(), height=750)
    # else:
    #     st.write("Please select at least one module to visualize.")
