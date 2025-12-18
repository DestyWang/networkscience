from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, MutableMapping, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

DATASET_ROOT = Path(
    "/home/bcl/wanghongyu/other/networkscience/data/"
    "NAPAbench/NAPAbenchVer2/benchmarkDataset"
).resolve()


@dataclass
class FamilyData:
    """
    Aggregated container for all files that belong to a single network family.

    Attributes
    ----------
    graphs : Dict[str, nx.Graph]
        Mapping where each key is the network identifier (e.g. 'A') and the
        value is an undirected graph containing |V| nodes and |E| edges.
    annotations : Dict[str, Dict[str, List[str]]]
        Mapping where annotations[network][node] stores a list of functional
        orthology (FO) terms associated with that node.
    similarities : Dict[Tuple[str, str], pd.DataFrame]
        The dataframe for a pair (u, v) has shape (m, 3) and contains the
        columns ['source_node', 'target_node', 'score'] describing m fuzzy
        similarity scores between nodes of the two networks.
    """

    graphs: Dict[str, nx.Graph]
    annotations: Dict[str, Dict[str, List[str]]]
    similarities: Dict[Tuple[str, str], pd.DataFrame]


def discover_dataset_structure(dataset_root: Path = DATASET_ROOT,) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    """
    Build a nested dictionary that mirrors the on-disk hierarchy.

    Parameters
    ----------
    dataset_root : Path
        Absolute directory that contains the top-level benchmark folders.

    Returns
    -------
    Dict[str, Dict[str, Dict[str, List[str]]]]
        structure[suite][category][family] -> sorted list of file names
        contained in that family directory.
    """

    structure: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    for suite_dir in sorted(dataset_root.iterdir()):
        if not suite_dir.is_dir():
            continue
        categories: Dict[str, Dict[str, List[str]]] = {}
        for category_dir in sorted(suite_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            families: Dict[str, List[str]] = {}
            for family_dir in sorted(category_dir.iterdir()):
                if not family_dir.is_dir():
                    continue
                file_names = sorted(
                    path.name for path in family_dir.iterdir() if path.is_file()
                )
                families[family_dir.name] = file_names
            categories[category_dir.name] = families
        structure[suite_dir.name] = categories
    return structure


def summarize_structure(structure: Dict[str, Dict[str, Dict[str, List[str]]]]) -> str:
    """
    Convert the nested structure dictionary into a human-readable summary.

    Parameters
    ----------
    structure : Dict[str, Dict[str, Dict[str, List[str]]]]
        Output of :func:`discover_dataset_structure`.

    Returns
    -------
    str
        Multi-line description of counts for suites, categories, and families.
    """

    lines: List[str] = []
    for suite_name, categories in sorted(structure.items()):
        family_count = sum(len(families) for families in categories.values())
        lines.append(
            f"{suite_name}: {len(categories)} categories, {family_count} families total"
        )
        for category_name, families in sorted(categories.items()):
            lines.append(
                f"  - {category_name}: {len(families)} families "
                f"({', '.join(sorted(families.keys()))})"
            )
    return "\n".join(lines)


def read_network_file(file_path: Path) -> nx.Graph:
    """
    Load a .net file into a NetworkX graph.

    Parameters
    ----------
    file_path : Path
        Absolute path to a '.net' file that stores an edge list.

    Returns
    -------
    nx.Graph
        Undirected graph with |V| nodes and |E| edges represented in the file.
    """

    graph = nx.Graph(name=file_path.stem)
    with file_path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            stripped = line.strip()
            if not stripped:
                continue
            node_u, node_v = stripped.split()
            graph.add_edge(node_u, node_v)
    graph.graph["source_file"] = str(file_path)
    return graph


def read_functional_annotations(file_path: Path) -> Dict[str, List[str]]:
    """
    Load a .fo file into a node -> FO terms mapping.

    Parameters
    ----------
    file_path : Path
        Absolute path to a '.fo' file (two-tab-separated columns).

    Returns
    -------
    Dict[str, List[str]]
        Dictionary where each value is a list of FO terms (shape (k_i,)) that
        belongs to node i. The list length varies per node.
    """

    annotations: MutableMapping[str, List[str]] = defaultdict(list)
    with file_path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            stripped = line.strip()
            if not stripped:
                continue
            node_id, fo_term = stripped.split()
            annotations[node_id].append(fo_term)
    return dict(annotations)


def read_similarity_file(file_path: Path) -> pd.DataFrame:
    """
    Load a .sim file into a pandas DataFrame.

    Parameters
    ----------
    file_path : Path
        Absolute path to a '.sim' file containing node pairs and scores.

    Returns
    -------
    pd.DataFrame
        DataFrame with shape (m, 3) and columns
        ['source_node', 'target_node', 'score'] where m is the number of
        cross-network node comparisons present in the file.
    """

    dataframe = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=["source_node", "target_node", "score"],
        engine="python",
    )
    dataframe.attrs["source_file"] = str(file_path)
    return dataframe


def load_family_data(family_dir: Path) -> FamilyData:
    """
    Read every .net, .fo, and .sim file inside a family directory.

    Parameters
    ----------
    family_dir : Path
        Absolute directory that contains files for a specific family.

    Returns
    -------
    FamilyData
        Container populated with networks, annotations, and similarities for
        the requested family.
    """

    graphs: Dict[str, nx.Graph] = {}
    for net_file in sorted(family_dir.glob("*.net")):
        network_name = net_file.stem.upper()
        graphs[network_name] = read_network_file(net_file)
        LOGGER.info(
            "Loaded %s (%d nodes, %d edges)",
            network_name,
            graphs[network_name].number_of_nodes(),
            graphs[network_name].number_of_edges(),
        )

    annotations: Dict[str, Dict[str, List[str]]] = {}
    for fo_file in sorted(family_dir.glob("*.fo")):
        network_name = fo_file.stem.upper()
        annotations[network_name] = read_functional_annotations(fo_file)
        LOGGER.info(
            "Loaded %s annotations (%d nodes with FO labels)",
            network_name,
            len(annotations[network_name]),
        )

    similarities: Dict[Tuple[str, str], pd.DataFrame] = {}
    for sim_file in sorted(family_dir.glob("*.sim")):
        file_key = tuple(part.upper() for part in sim_file.stem.split("-"))
        similarities[file_key] = read_similarity_file(sim_file)
        reverse_key = tuple(reversed(file_key))
        similarities[reverse_key] = similarities[file_key]
        LOGGER.info(
            "Loaded similarity file %s (%d pairs)",
            "-".join(file_key),
            similarities[file_key].shape[0],
        )

    return FamilyData(graphs=graphs, annotations=annotations, similarities=similarities)


def plot_degree_distribution(graph: nx.Graph, network_name: str, bins: int = 50,) -> plt.Figure:
    """
    Create a histogram of node degrees for a given network.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph whose degrees will be summarized.
    network_name : str
        Identifier used in the plot title.
    bins : int
        Number of histogram bins (shape parameter for the histogram vector).

    Returns
    -------
    plt.Figure
        Figure object containing the histogram visualization.
    """

    degrees: List[int] = [degree for _, degree in graph.degree()]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(degrees, bins=bins, color="#377eb8", alpha=0.75)
    ax.set_title(f"{network_name}: Degree distribution")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Node count")
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    return fig


def plot_annotation_histogram(annotations: Dict[str, List[str]], network_name: str, top_k_terms: int = 20,) -> plt.Figure:
    """
    Plot the frequency of the top FO terms for a single network.

    Parameters
    ----------
    annotations : Dict[str, List[str]]
        Mapping node -> FO terms (each list has shape (k_i,)).
    network_name : str
        Identifier used in the plot title.
    top_k_terms : int
        Number of most frequent FO terms to display.

    Returns
    -------
    plt.Figure
        Figure object containing a horizontal bar chart.
    """

    counter = Counter(term for values in annotations.values() for term in values)
    most_common = counter.most_common(top_k_terms)
    labels = [item[0] for item in most_common]
    counts = [item[1] for item in most_common]

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(labels) + 1))
    ax.barh(labels, counts, color="#4daf4a", alpha=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("Occurrences")
    ax.set_title(f"{network_name}: Top {len(labels)} FO terms")
    fig.tight_layout()
    return fig


def plot_similarity_heatmap(similarity_df: pd.DataFrame, pair_label: str, top_k_pairs: int = 200,) -> plt.Figure:
    """
    Plot a heatmap for the highest-scoring subset of similarity pairs.

    Parameters
    ----------
    similarity_df : pd.DataFrame
        DataFrame shaped (m, 3) with the columns
        ['source_node', 'target_node', 'score'].
    pair_label : str
        Identifier for the pair, e.g. 'A-B'.
    top_k_pairs : int
        Number of rows to keep before pivoting into a matrix.

    Returns
    -------
    plt.Figure
        Figure object containing the similarity heatmap.
    """

    subset = similarity_df.nlargest(top_k_pairs, "score")
    pivot = subset.pivot(
        index="source_node", columns="target_node", values="score"
    ).fillna(0.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    heatmap = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_title(f"{pair_label}: top {top_k_pairs} similarity scores")
    ax.set_xlabel("Target node")
    ax.set_ylabel("Source node")
    fig.colorbar(heatmap, ax=ax, fraction=0.025, pad=0.02, label="Score")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=6)
    fig.tight_layout()
    return fig


def visualize_family(
    family_data: FamilyData,
    similarity_pair: Tuple[str, str] = ("A", "B"),
    degree_bins: int = 60,
    top_annotation_terms: int = 20,
    top_similarity_pairs: int = 250,
) -> Dict[str, plt.Figure]:
    """
    Generate diagnostic plots for the requested family.

    Parameters
    ----------
    family_data : FamilyData
        Container returned by :func:`load_family_data`.
    similarity_pair : Tuple[str, str]
        Network identifiers for the similarity heatmap.
    degree_bins : int
        Number of bins for degree histograms.
    top_annotation_terms : int
        How many FO terms to display per network.
    top_similarity_pairs : int
        Number of rows retained from the similarity dataframe.

    Returns
    -------
    Dict[str, plt.Figure]
        Mapping where the key describes the visualization and the value is the
        Matplotlib Figure object. Figures are not saved to disk to respect the
        user's workspace constraints.
    """

    figures: Dict[str, plt.Figure] = {}
    for network_name, graph in family_data.graphs.items():
        figures[f"{network_name}_degree_hist"] = plot_degree_distribution(
            graph, network_name, bins=degree_bins
        )
        if network_name in family_data.annotations:
            figures[f"{network_name}_fo_hist"] = plot_annotation_histogram(
                family_data.annotations[network_name],
                network_name,
                top_k_terms=top_annotation_terms,
            )

    pair_key = tuple(part.upper() for part in similarity_pair)
    similarity_df = family_data.similarities.get(pair_key)
    if similarity_df is None:
        pair_key = tuple(reversed(pair_key))
        similarity_df = family_data.similarities.get(pair_key)
    if similarity_df is not None:
        label = "-".join(pair_key)
        figures[f"{label}_similarity_heatmap"] = plot_similarity_heatmap(
            similarity_df,
            pair_label=label,
            top_k_pairs=top_similarity_pairs,
        )
    else:
        LOGGER.warning(
            "Similarity file for pair %s not found inside the family bundle.",
            similarity_pair,
        )
    return figures


def main() -> None:
    """
    Entry point that prints the dataset structure and loads one example family.
    """

    LOGGER.info("Scanning dataset root at %s", DATASET_ROOT)
    structure = discover_dataset_structure()
    print(summarize_structure(structure))

    sample_family = DATASET_ROOT / "2way" / "DMR" / "family1"
    if not sample_family.exists():
        LOGGER.error("Sample family directory %s does not exist.", sample_family)
        return

    LOGGER.info("Loading sample family from %s", sample_family)
    family_data = load_family_data(sample_family)
    figures = visualize_family(family_data, similarity_pair=("A", "B"))
    LOGGER.info(
        "Generated %d diagnostic plots. Call `.show()` or `.savefig()` on the "
        "figure objects that the function returned to inspect them.",
        len(figures),
    )


if __name__ == "__main__":
    main()

