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

# Helper Function: Plot Metrics
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
    
    # Communities visualization
    st.subheader("Community Structure")
    fig_com, ax_com = plt.subplots(1, 2, figsize=(14, 7))
    pos1 = nx.spring_layout(G1)  # Spring layout for G1
    pos2 = nx.spring_layout(G2)  # Spring layout for G2
    
    nx.draw(G1, pos1, ax=ax_com[0], node_color=[len(c) for c in communities1], cmap='viridis', node_size=50)
    ax_com[0].set_title("Network 1 Communities")
    
    nx.draw(G2, pos2, ax=ax_com[1], node_color=[len(c) for c in communities2], cmap='viridis', node_size=50)
    ax_com[1].set_title("Network 2 Communities")
    
    st.pyplot(fig_com)
    
    st.subheader("Key Insights")
    st.write("""
    - Use these plots and metrics to identify key differences between networks.
    - Consider factors like clustering coefficients, modularity, and edge weight distributions for detailed analysis.
    """)

