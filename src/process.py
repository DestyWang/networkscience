from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, MutableMapping, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch

import math
from scipy.special import zeta as hurwitz_zeta
from scipy.optimize import minimize_scalar

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
    Container that bundles every artifact that belongs to one family.

    Attributes
    ----------
    graphs : Dict[str, nx.Graph]
        Loaded adjacency information for each network variant (A/B/C...).
    annotations : Dict[str, Dict[str, List[str]]]
        FO (functional orthology) labels keyed by network then node id.
    similarities : Dict[Tuple[str, str], pd.DataFrame]
        Cross-network similarity tables keyed by the ordered pair (src, tgt).
    """

    graphs: Dict[str, nx.Graph]
    annotations: Dict[str, Dict[str, List[str]]]
    similarities: Dict[Tuple[str, str], pd.DataFrame]


def discover_dataset_structure(
    dataset_root: Path = DATASET_ROOT,
) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    """
    Recursively enumerate the benchmark hierarchy.

    Returns
    -------
    Dict[str, Dict[str, Dict[str, List[str]]]]
        structure[suite][category][family] -> list of files.
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
    Convert the nested structure dictionary into a readable summary.
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
    Load a `.net` edge list into a NetworkX graph.
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
    Load `.fo` files (node -> FO term) into a dictionary.
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
    Load `.sim` files into a pandas DataFrame.
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
    Load every network/annotation/similarity file stored inside a family folder.
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


def plot_degree_distribution(
    graph: nx.Graph,
    network_name: str,
    bins: int = 50,
) -> plt.Figure:
    """
    Plot a histogram of degrees for diagnostic purposes.
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


def plot_annotation_histogram(
    annotations: Dict[str, List[str]],
    network_name: str,
    top_k_terms: int = 20,
) -> plt.Figure:
    """
    Visualize the most frequent FO terms within a network.
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


def visualize_family(
    family_data: FamilyData,
    similarity_pair: Tuple[str, str] = ("A", "B"),
    degree_bins: int = 60,
    top_annotation_terms: int = 20,
    top_similarity_pairs: int = 250,
) -> Dict[str, plt.Figure]:
    """
    Generate a standard set of exploratory plots for one family directory.
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


def get_dist_matrix(raph: nx.Graph,node_order: Sequence[str] | None = None) -> Tuple[torch.Tensor, List[str]]:
    """
    Compute the all-pairs shortest path distances for the requested graph.

    Parameters
    ----------
    graph : nx.Graph
        Input network (treated as undirected and unweighted).
    node_order : Sequence[str] | None
        Optional deterministic node ordering. Defaults to sorted node ids.

    Returns
    -------
    Tuple[torch.Tensor, List[str]]
        Distance matrix with shape (n, n) and the node order that was used.
    """

    if node_order is None:
        node_order = sorted(graph.nodes())
    node_order = list(node_order)
    if not node_order:
        return torch.zeros((0, 0), dtype=torch.float32), []

    adjacency = nx.to_numpy_array(
        graph,
        nodelist=node_order,
        dtype=np.float32,
        weight=None,
    )
    adjacency = torch.from_numpy(adjacency.astype(np.float32, copy=False))

    n = adjacency.shape[0]
    dist_matrix = torch.full((n, n), float("inf"), dtype=torch.float32)
    diag_idx = torch.arange(n)
    dist_matrix[diag_idx, diag_idx] = 0.0

    edge_mask = adjacency > 0.0
    if edge_mask.any():
        dist_matrix[edge_mask] = adjacency[edge_mask]

    for k in range(n):
        dist_matrix = torch.minimum(
            dist_matrix,
            dist_matrix[:, k].unsqueeze(1) + dist_matrix[k].unsqueeze(0),
        )

    if torch.isinf(dist_matrix).any():
        finite = dist_matrix[~torch.isinf(dist_matrix)]
        fill_value = float(finite.max() * 2.0) if finite.numel() else 0.0
        dist_matrix = torch.where(
            torch.isinf(dist_matrix),
            torch.full_like(dist_matrix, fill_value),
            dist_matrix,
        )
    return dist_matrix, node_order


def get_ACC(graph: nx.Graph) -> Dict[str, float]:
    """
    Per-node average clustering coefficient (local clustering).
    """

    return nx.clustering(graph)


def get_ASP(graph: nx.Graph) -> Dict[str, float]:
    """
    Per-node average shortest-path length (within each connected component).
    """

    asp: Dict[str, float] = {}
    for node in graph.nodes():
        lengths = nx.single_source_shortest_path_length(graph, node)
        if len(lengths) <= 1:
            asp[node] = 0.0
            continue
        total = sum(dist for target, dist in lengths.items() if target != node)
        asp[node] = total / max(len(lengths) - 1, 1)
    return asp


def get_BC(graph: nx.Graph, *, k: int | None = None, seed: int = 42) -> Dict[str, float]:
    """
    Betweenness centrality for every node (approximate if k is provided).
    """

    return nx.betweenness_centrality(graph, k=k, normalized=True, seed=seed)


def _largest_component(graph: nx.Graph) -> nx.Graph:
    """
    Internal helper that extracts the largest connected component copy.
    """

    if graph.number_of_nodes() == 0:
        return graph.copy()
    if nx.is_connected(graph):
        return graph.copy()
    component_nodes = max(nx.connected_components(graph), key=len)
    return graph.subgraph(component_nodes).copy()


def get_small_worldness(graph: nx.Graph, *, seed: int = 42, n_iter: int = 8,) -> Tuple[float, bool]:
    """
    Estimate small-world-ness using the sigma statistic.

    Returns
    -------
    Tuple[float, bool]
        sigma value and whether sigma > 1 (classic small-world criterion).
    """

    core = _largest_component(graph)
    n = core.number_of_nodes()
    m = core.number_of_edges()
    if n < 3 or m == 0:
        return float("nan"), False

    clustering = nx.average_clustering(core)
    path_length = nx.average_shortest_path_length(core)

    rng = np.random.default_rng(seed)
    sigma_samples: List[float] = []
    for _ in range(n_iter):
        LOGGER.info("Iteration: %s", _)
        random_graph = nx.gnm_random_graph(
            n,
            m,
            seed=int(rng.integers(0, 1_000_000)),
        )
        if random_graph.number_of_edges() == 0 or not nx.is_connected(random_graph):
            random_graph = _largest_component(random_graph)
        rand_clustering = nx.average_clustering(random_graph)
        rand_path = (
            nx.average_shortest_path_length(random_graph)
            if random_graph.number_of_edges() > 0
            else float("inf")
        )
        if rand_clustering == 0 or rand_path == float("inf"):
            continue
        sigma_samples.append((clustering / rand_clustering) / (path_length / rand_path))

    if not sigma_samples:
        return float("nan"), False
    sigma_value = float(np.mean(sigma_samples))
    return sigma_value, sigma_value > 1.0


def get_clustering(graph: nx.Graph, *, seed: int = 42, n_iter: int = 8) -> Tuple[float, bool]:
    """
    Compare clustering coefficient against Erdos-Renyi baselines.

    Returns
    -------
    Tuple[float, bool]
        Ratio C / C_rand and a boolean indicating if the network is more
        clustered than random graphs (ratio > 1).
    """

    core = _largest_component(graph)
    n = core.number_of_nodes()
    m = core.number_of_edges()
    if n < 3 or m == 0:
        return float("nan"), False

    clustering = nx.average_clustering(core)

    rng = np.random.default_rng(seed)
    ratios: List[float] = []
    for _ in range(n_iter):
        random_graph = nx.gnm_random_graph(
            n,
            m,
            seed=int(rng.integers(0, 1_000_000)),
        )
        rand_clustering = nx.average_clustering(random_graph)
        if rand_clustering == 0:
            continue
        ratios.append(clustering / rand_clustering)

    if not ratios:
        return float("nan"), False
    ratio = float(np.mean(ratios))
    return ratio, ratio > 1.0


def get_scale_free_ness(graph: nx.Graph) -> Tuple[float, bool]:
    """
    Fit the degree distribution on a log-log scale to check for power-law tails.

    Returns
    -------
    Tuple[float, bool]
        The gamma value.
        r^2 goodness-of-fit value and a boolean indicating whether the fitted
        exponent falls inside the canonical (2, 3) range with r^2 >= 0.8.
    """

    degrees = torch.tensor([deg for _, deg in graph.degree()], dtype=torch.float32)
    degrees = degrees[degrees > 0]
    if degrees.numel() < 3:
        return float("nan"), False

    unique, counts = torch.unique(degrees, return_counts=True)
    mask = unique >= 1
    unique = unique[mask]
    counts = counts[mask].float()
    if unique.numel() < 2:
        return float("nan"), False

    x = torch.log(unique)
    y = torch.log(counts)
    x_mean = x.mean()
    y_mean = y.mean()

    denom = torch.sum((x - x_mean) ** 2)
    if denom == 0:
        return float("nan"), False
    slope = torch.sum((x - x_mean) * (y - y_mean)) / denom
    intercept = y_mean - slope * x_mean
    y_pred = slope * x + intercept
    ss_res = torch.sum((y - y_pred) ** 2)
    ss_tot = torch.sum((y - y_mean) ** 2)
    if ss_tot == 0:
        return float("nan"), False
    r_squared = 1.0 - ss_res / ss_tot
    gamma = -float(slope.item())
    r_value = float(r_squared.item())
    is_scale_free = 2.0 <= gamma <= 3.0 and r_value >= 0.8
    return gamma, r_value, is_scale_free


def _degrees_positive(graph: nx.Graph) -> np.ndarray:
    degs = np.array([d for _, d in graph.degree()], dtype=np.int64)
    return degs[degs > 0]

def _neg_log_likelihood_discrete(alpha: float, data: np.ndarray, kmin: int) -> float:
    """
    Negative log-likelihood for discrete power-law P(k) = k^{-alpha} / zeta(alpha, kmin).
    We return +inf for invalid alpha values.
    """
    if alpha <= 1.0:
        return float("inf")
    try:
        z = hurwitz_zeta(alpha, kmin)
        if not np.isfinite(z) or z <= 0.0:
            return float("inf")
        n = data.size
        s = np.sum(np.log(data))
        # negative log-likelihood:
        return n * math.log(z) + alpha * float(s)
    except Exception:
        return float("inf")

def _mle_alpha_discrete(data: np.ndarray, kmin: int, alpha_max: float = 10.0) -> float:
    """
    Find MLE alpha for discrete power-law data >= kmin by numeric optimization.
    Uses bounded scalar minimization on alpha in (1+eps, alpha_max).
    Returns estimated alpha (float).
    """
    # ensure data contains only integers >= kmin
    tail = data[data >= kmin]
    if tail.size == 0:
        return float("nan")

    # objective: minimize negative log-likelihood
    res = minimize_scalar(
        lambda a: _neg_log_likelihood_discrete(a, tail, kmin),
        bounds=(1.0001, alpha_max),
        method="bounded",
        options={"xatol": 1e-6},
    )
    if not res.success:
        return float("nan")
    return float(res.x)

def _ks_statistic_discrete(data: np.ndarray, kmin: int, alpha: float, kmax: int = None) -> float:
    """
    Compute KS statistic between empirical CDF of data>=kmin and the discrete power-law CDF with given alpha.
    kmax controls truncation for the theoretical CDF sum; default is max(data).
    """
    tail = data[data >= kmin]
    if tail.size == 0:
        return float("nan")
    if kmax is None:
        kmax = int(tail.max())

    # empirical CDF at integer k: S(k) = fraction of tail values >= k
    # But Clauset uses CDF F(k) = P(X <= k) usually; KS uses sup |S_emp - S_model|
    # We'll compute empirical CDF F_emp(k) = fraction <= k for k in [kmin..kmax]
    ks = np.arange(kmin, kmax + 1)
    counts = np.array([np.sum(tail <= k) for k in ks], dtype=float)
    F_emp = counts / tail.size

    # theoretical pmf p(k) and cdf F_model(k)
    try:
        z = hurwitz_zeta(alpha, kmin)
        pk = ks.astype(np.float64) ** (-alpha) / float(z)
    except Exception:
        return float("inf")
    F_model = np.cumsum(pk)
    # Make sure model cdf has same support; if we truncated at kmax, renormalize if needed:
    # but we want the model CDF on ks; so it's fine.

    # KS statistic (sup |F_emp - F_model|)
    ks_extended = min(kmax, ks[-1])
    # match lengths
    if F_model.size != F_emp.size:
        # fallback: interpolate model onto ks positions (shouldn't happen)
        F_model = F_model[:F_emp.size]
    D = np.max(np.abs(F_emp - F_model))
    return float(D)

def _sample_discrete_powerlaw(alpha: float, kmin: int, size: int, kmax: int) -> np.ndarray:
    """
    Sample 'size' integers from discrete power-law P(k) ~ k^{-alpha} for k in [kmin..kmax].
    Uses inverse transform with precomputed cdf (truncated at kmax).
    """
    ks = np.arange(kmin, kmax + 1)
    z = np.sum(ks.astype(np.float64) ** (-alpha))
    pmf = ks.astype(np.float64) ** (-alpha) / z
    cdf = np.cumsum(pmf)
    u = np.random.rand(size)
    # searchsorted returns indices in [0..len(ks)-1]
    inds = np.searchsorted(cdf, u, side="right")
    samples = ks[np.clip(inds, 0, len(ks) - 1)]
    return samples


def get_scale_free_ness_clauset(
    graph: nx.Graph, *,
    n_sims: int = 500,
    alpha_max: float = 10.0,
    min_tail: int = 50,
    rng_seed: int = 42
) -> Tuple[float, int, float, bool]:
    """
    Clauset et al. (2009) - rigorous discrete power-law fit for degree distribution.

    Returns
    -------
    Tuple[gamma, kmin, p_value, is_plausible]
      gamma: fitted power-law exponent alpha (so degree PDF ~ k^{-gamma} for k >= kmin)
      kmin: selected minimum degree for tail
      p_value: Monte-Carlo p-value for goodness-of-fit (higher means power-law is plausible)
      is_plausible: boolean (p_value > 0.1) per Clauset's recommendation
    Notes
    -----
    - This implements Clauset's algorithm: for each candidate kmin, find MLE for alpha,
      compute KS statistic between empirical tail and fitted discrete power-law,
      choose kmin minimizing KS. Then perform Monte-Carlo to get p-value.
    - n_sims controls Monte-Carlo repetitions (costly for large graphs).
    - min_tail: minimal number of points in tail to consider a candidate kmin.
    """
    rng = np.random.default_rng(rng_seed)

    degs = _degrees_positive(graph)
    if degs.size < 3:
        return float("nan"), -1, float("nan"), False

    unique_ks = np.unique(degs)
    # candidate kmin values: unique degrees, but need enough tail samples
    candidates = []
    for k in unique_ks:
        tail_count = np.sum(degs >= k)
        if tail_count >= min_tail:
            candidates.append(int(k))
    if not candidates:
        # try relaxing min_tail
        candidates = [int(k) for k in unique_ks if np.sum(degs >= k) >= 3]
        if not candidates:
            return float("nan"), -1, float("nan"), False

    best = {"kmin": None, "alpha": None, "ks": np.inf, "n_tail": 0}

    kmax_obs = int(degs.max())

    # 1) scan kmin
    for kmin in candidates:
        alpha_hat = _mle_alpha_discrete(degs, kmin, alpha_max=alpha_max)
        if not np.isfinite(alpha_hat):
            continue
        ks_stat = _ks_statistic_discrete(degs, kmin, alpha_hat, kmax=kmax_obs)
        n_tail = int(np.sum(degs >= kmin))
        if ks_stat < best["ks"]:
            best.update({"kmin": int(kmin), "alpha": float(alpha_hat), "ks": float(ks_stat), "n_tail": n_tail})

    if best["kmin"] is None:
        return float("nan"), -1, float("nan"), False

    # 2) Monte-Carlo to estimate p-value
    kmin_star = best["kmin"]
    alpha_star = best["alpha"]
    ks_data = best["ks"]
    n_tail = best["n_tail"]

    # To sample synthetic data, we truncate distribution at observed kmax
    kmax = kmax_obs

    larger_count = 0
    # For reproducibility:
    np.random.seed(rng_seed)

    for i in range(n_sims):
        # sample synthetic tail of size n_tail from fitted discrete power-law
        synth_tail = _sample_discrete_powerlaw(alpha_star, kmin_star, n_tail, kmax)
        # combine with empirical lower-degree values? Clauset uses only tail for KS and simulation
        # Fit alpha_s to the synthetic tail (re-estimate)
        alpha_s = _mle_alpha_discrete(synth_tail, kmin_star, alpha_max=alpha_max)
        if not np.isfinite(alpha_s):
            # treat as large KS (skip)
            continue
        ks_s = _ks_statistic_discrete(synth_tail, kmin_star, alpha_s, kmax=kmax)
        if ks_s > ks_data:
            larger_count += 1

    # p-value = fraction of synthetic KS greater than empirical KS
    p_value = larger_count / max(1, n_sims)
    is_plausible = p_value > 0.1  # Clauset recommended threshold

    gamma = float(alpha_star)
    return gamma, kmin_star, float(p_value), bool(is_plausible)


def main() -> None:
    """
    Entry-point for quick manual inspection.
    """

    LOGGER.info("Scanning dataset root at %s", DATASET_ROOT)
    structure = discover_dataset_structure()
    print(summarize_structure(structure))


if __name__ == "__main__":
    main()
