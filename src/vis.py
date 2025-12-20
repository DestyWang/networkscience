import matplotlib.pyplot as plt
import networkx as nx

from src.utils import tensor_to_numpy


def plot2graphs(G1: nx.Graph, G2: nx.Graph) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    nx.draw(G1, with_labels=True, node_color='skyblue', edge_color='gray', ax=ax1)
    ax1.set_title("G1")
    nx.draw(G2, with_labels=True, node_color='salmon', edge_color='gray', ax=ax2)
    ax2.set_title("G2")
    plt.tight_layout()
    return fig


def plot_degree_dist(graph: nx.Graph, *, bins: int = 50, title: str | None = None) -> plt.Figure:
    """
    Histogram of node degrees for quick inspection.
    """

    degrees = [degree for _, degree in graph.degree()]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(degrees, bins=bins, color="#377eb8", alpha=0.8)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Node count")
    ax.set_title(title or graph.graph.get("name", "Degree distribution"))
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    return fig


def plot_heatmap(
    matrix,
    *,
    title: str = "Heatmap",
    cmap: str = "viridis",
    xlabel: str | None = None,
    ylabel: str | None = None,
    xticks: list[str] | None = None,
    yticks: list[str] | None = None,
) -> plt.Figure:
    """
    Generic helper to visualize matrices such as distance / coupling / similarity.
    """

    array = tensor_to_numpy(matrix)
    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap = ax.imshow(array, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(xlabel or "Target")
    ax.set_ylabel(ylabel or "Source")
    if xticks is not None:
        ax.set_xticks(range(len(xticks)))
        ax.set_xticklabels(xticks, rotation=90, fontsize=6)
    if yticks is not None:
        ax.set_yticks(range(len(yticks)))
        ax.set_yticklabels(yticks, fontsize=6)
    fig.colorbar(heatmap, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    return fig