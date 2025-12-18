import torch 
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def plot2graphs(G1: nx.Graph, G2: nx.Graph) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    nx.draw(G1, with_labels=True, node_color='skyblue', edge_color='gray', ax=ax1)
    ax1.set_title("G1")
    nx.draw(G2, with_labels=True, node_color='salmon', edge_color='gray', ax=ax2)
    ax2.set_title("G2")
    plt.tight_layout()
    return fig