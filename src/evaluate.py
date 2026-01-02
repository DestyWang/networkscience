from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Sequence, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch

from src.utils import tensor_to_numpy


def _ensure_matrix(matrix: np.ndarray | torch.Tensor) -> np.ndarray:
    """
    Shared helper that normalizes the coupling representation.
    """

    array = tensor_to_numpy(matrix).astype(np.float64, copy=False)
    if array.ndim != 2:
        raise ValueError("Coupling matrix必须是二维数组。")
    return array


def _similarity_matrix_from_file(
    sim_path: Path | str,
    source_nodes: Sequence[str],
    target_nodes: Sequence[str],
) -> np.ndarray:
    """
    Convert `.sim` triplets into a dense matrix aligned with the supplied nodes.
    """

    df = pd.read_csv(
        Path(sim_path),
        sep=r"\s+",
        header=None,
        names=["source_node", "target_node", "score"],
        engine="python",
    )
    source_index = {node: idx for idx, node in enumerate(source_nodes)}
    target_index = {node: idx for idx, node in enumerate(target_nodes)}
    matrix = np.zeros((len(source_nodes), len(target_nodes)), dtype=np.float64)
    for row in df.itertuples(index=False):
        i = source_index.get(row.source_node)
        j = target_index.get(row.target_node)
        if i is None or j is None:
            continue
        matrix[i, j] = float(row.score)
    return matrix


def uniform_coupling_from_clusters(
    cluster_path: Path | str,
    source_nodes: Sequence[str],
    target_nodes: Sequence[str],
) -> np.ndarray:
    """
    Convert IsoRankN `final_cluster` output into a uniform coupling matrix.

    Each valid cluster (containing at least one source node and one target node)
    contributes the same total mass, which is distributed uniformly across all
    cross-network pairs inside that cluster. The resulting matrix sums to 1.0 and
    can be evaluated with the same metrics used for FUGW couplings.
    """

    source_index = {node: idx for idx, node in enumerate(source_nodes)}
    target_index = {node: idx for idx, node in enumerate(target_nodes)}
    clusters: List[Tuple[List[int], List[int]]] = []

    with Path(cluster_path).open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            tokens = line.strip().split()
            if not tokens:
                continue
            src_idx: List[int] = []
            tgt_idx: List[int] = []
            for token in tokens:
                if token in source_index:
                    src_idx.append(source_index[token])
                elif token in target_index:
                    tgt_idx.append(target_index[token])
            if src_idx and tgt_idx:
                clusters.append((src_idx, tgt_idx))

    matrix = np.zeros((len(source_nodes), len(target_nodes)), dtype=np.float64)
    if not clusters:
        return matrix

    cluster_mass = 1.0 / len(clusters)
    for src_idx, tgt_idx in clusters:
        weight = cluster_mass / (len(src_idx) * len(tgt_idx))
        for i in src_idx:
            for j in tgt_idx:
                matrix[i, j] += weight

    return matrix


def sim_score(
    sim_path: Path | str,
    coupling: np.ndarray | torch.Tensor,
    source_nodes: Sequence[str],
    target_nodes: Sequence[str],
) -> float:
    """
    基于 .sim 文件和 FUGW 耦合矩阵计算 alignment 相似分数。
    """

    P = _ensure_matrix(coupling)
    if P.shape != (len(source_nodes), len(target_nodes)):
        raise ValueError("耦合矩阵尺寸与节点数量不一致。")
    S = _similarity_matrix_from_file(sim_path, source_nodes, target_nodes)
    return float(np.sum(P * S))


def FO_con_soft(
    coupling: np.ndarray | torch.Tensor,
    source_nodes: Sequence[str],
    target_nodes: Sequence[str],
    source_annotations: Mapping[str, Sequence[str]],
    target_annotations: Mapping[str, Sequence[str]],
) -> float:
    """
    FO-consistency：耦合矩阵加权的功能同源一致性。
    """

    P = _ensure_matrix(coupling)
    src_lookup = {node: set(source_annotations.get(node, [])) for node in source_nodes}
    tgt_lookup = {node: set(target_annotations.get(node, [])) for node in target_nodes}

    indicator = np.zeros_like(P, dtype=np.float64)
    for i, src in enumerate(source_nodes):
        labels_src = src_lookup[src]
        if not labels_src:
            continue
        for j, tgt in enumerate(target_nodes):
            labels_tgt = tgt_lookup[tgt]
            if labels_src & labels_tgt:
                indicator[i, j] = 1.0
    return float(np.sum(P * indicator))


def FO_con_hard(
    coupling: np.ndarray | torch.Tensor,
    source_nodes: Sequence[str],
    target_nodes: Sequence[str],
    source_annotations: Mapping[str, Sequence[str]],
    target_annotations: Mapping[str, Sequence[str]],
    *,
    axis: str = "source",
) -> float:
    """
    Hard-assign FO consistency via row/column argmax.

    axis = 'source'  -> match source->target (row-wise argmax)
    axis = 'target'  -> match target->source (column-wise argmax)
    """

    axis = axis.lower()
    if axis not in {"source", "target"}:
        raise ValueError("axis must be 'source' or 'target'")

    P = _ensure_matrix(coupling)
    if P.shape != (len(source_nodes), len(target_nodes)):
        raise ValueError("耦合矩阵尺寸与节点数量不一致。")

    src_lookup = {node: set(source_annotations.get(node, [])) for node in source_nodes}
    tgt_lookup = {node: set(target_annotations.get(node, [])) for node in target_nodes}

    if axis == "source":
        assignments = np.argmax(P, axis=1)
        hits = 0
        considered = 0
        for i, j in enumerate(assignments):
            labels_src = src_lookup.get(source_nodes[i], set())
            if not labels_src:
                continue
            considered += 1
            labels_tgt = tgt_lookup.get(target_nodes[int(j)], set())
            if labels_src & labels_tgt:
                hits += 1
        if considered == 0:
            return float("nan")
        return hits / considered

    # axis == "target"
    assignments = np.argmax(P, axis=0)
    hits = 0
    considered = 0
    for j, i in enumerate(assignments):
        labels_tgt = tgt_lookup.get(target_nodes[j], set())
        if not labels_tgt:
            continue
        considered += 1
        labels_src = src_lookup.get(source_nodes[int(i)], set())
        if labels_src & labels_tgt:
            hits += 1
    if considered == 0:
        return float("nan")
    return hits / considered


def _topk_hits(
    matrix: np.ndarray,
    label_lookup_a: Dict[str, set],
    label_lookup_b: Dict[str, set],
    nodes_a: Sequence[str],
    nodes_b: Sequence[str],
    *,
    k: int,
) -> float:
    """
    Compute the probability that each node finds a FO-matched partner within Top-k.
    """

    if matrix.size == 0:
        return float("nan")

    hits = 0
    considered = 0
    for i, node in enumerate(nodes_a):
        labels = label_lookup_a.get(node, set())
        if not labels:
            continue
        considered += 1
        topk_idx = np.argsort(matrix[i])[::-1][: min(k, matrix.shape[1])]
        if any(labels & label_lookup_b.get(nodes_b[j], set()) for j in topk_idx):
            hits += 1
    if considered == 0:
        return float("nan")
    return hits / considered


def topk_FO_recall(
    coupling: np.ndarray | torch.Tensor,
    source_nodes: Sequence[str],
    target_nodes: Sequence[str],
    source_annotations: Mapping[str, Sequence[str]],
    target_annotations: Mapping[str, Sequence[str]],
    *,
    k: int = 5,
) -> float:
    """
    Top-k Functional Recall（双向平均）。
    """

    P = _ensure_matrix(coupling)
    if P.shape != (len(source_nodes), len(target_nodes)):
        raise ValueError("耦合矩阵尺寸与节点数量不一致。")

    src_lookup = {node: set(source_annotations.get(node, [])) for node in source_nodes}
    tgt_lookup = {node: set(target_annotations.get(node, [])) for node in target_nodes}

    recall_src = _topk_hits(P, src_lookup, tgt_lookup, source_nodes, target_nodes, k=k)
    recall_tgt = _topk_hits(
        P.T,
        tgt_lookup,
        src_lookup,
        target_nodes,
        source_nodes,
        k=k,
    )
    if np.isnan(recall_src) and np.isnan(recall_tgt):
        return float("nan")
    if np.isnan(recall_src):
        return recall_tgt
    if np.isnan(recall_tgt):
        return recall_src
    return 0.5 * (recall_src + recall_tgt)


def _argmax_mapping(
    matrix: np.ndarray,
    source_nodes: Sequence[str],
    target_nodes: Sequence[str],
) -> Dict[str, str]:
    """
    Deterministic mapping derived from the maximum of each row.
    """

    if matrix.shape[0] != len(source_nodes) or matrix.shape[1] != len(target_nodes):
        raise ValueError("Matrix shape does not align with node lists.")
    indices = np.argmax(matrix, axis=1)
    return {source_nodes[i]: target_nodes[int(j)] for i, j in enumerate(indices)}


def _count_preserved_edges(
    graph_source: nx.Graph,
    graph_target: nx.Graph,
    mapping: Mapping[str, str],
) -> int:
    """
    Count how many source edges remain edges after applying the mapping.
    """

    count = 0
    for u, v in graph_source.edges():
        mu = mapping.get(u)
        mv = mapping.get(v)
        if mu is None or mv is None:
            continue
        if graph_target.has_edge(mu, mv):
            count += 1
    return count


def _s3_from_mapping(
    graph_source: nx.Graph,
    graph_target: nx.Graph,
    mapping: Mapping[str, str],
) -> float:
    conserved = _count_preserved_edges(graph_source, graph_target, mapping)
    denom = (
        graph_source.number_of_edges()
        + graph_target.number_of_edges()
        - conserved
    )
    if denom == 0:
        return float("nan")
    return conserved / denom


def S3_hard(
    graph_source: nx.Graph,
    graph_target: nx.Graph,
    coupling: np.ndarray | torch.Tensor,
    source_nodes: Sequence[str],
    target_nodes: Sequence[str],
) -> float:
    """
    S³ 指标（hard mapping 版本，双向平均）。
    """

    P = _ensure_matrix(coupling)
    mapping_src = _argmax_mapping(P, source_nodes, target_nodes)
    mapping_tgt = _argmax_mapping(P.T, target_nodes, source_nodes)

    score_src = _s3_from_mapping(graph_source, graph_target, mapping_src)
    score_tgt = _s3_from_mapping(graph_target, graph_source, mapping_tgt)

    if np.isnan(score_src) and np.isnan(score_tgt):
        return float("nan")
    if np.isnan(score_src):
        return score_tgt
    if np.isnan(score_tgt):
        return score_src
    return 0.5 * (score_src + score_tgt)


def _adjacency_matrix(graph: nx.Graph, nodes: Sequence[str]) -> np.ndarray:
    """
    Return a float64 adjacency matrix with zeroed diagonal for the provided node order.
    """

    adj = nx.to_numpy_array(graph, nodelist=nodes, dtype=np.float64)
    np.fill_diagonal(adj, 0.0)
    return adj


def S3_soft(
    graph_source: nx.Graph,
    graph_target: nx.Graph,
    coupling: np.ndarray | torch.Tensor,
    source_nodes: Sequence[str],
    target_nodes: Sequence[str],
) -> float:
    """
    S³ 指标（soft / coupling 版本，双向平均）。
    """

    P = _ensure_matrix(coupling)
    if P.shape != (len(source_nodes), len(target_nodes)):
        raise ValueError("耦合矩阵尺寸与节点数量不一致。")

    As = _adjacency_matrix(graph_source, source_nodes)
    At = _adjacency_matrix(graph_target, target_nodes)

    proj_source = P @ At @ P.T
    proj_target = P.T @ As @ P

    conserved_src = 0.5 * float(np.sum(As * proj_source))
    conserved_tgt = 0.5 * float(np.sum(At * proj_target))
    conserved = 0.5 * (conserved_src + conserved_tgt)

    edges_source = graph_source.number_of_edges()
    edges_target = graph_target.number_of_edges()
    denom = edges_source + edges_target - conserved
    if denom == 0:
        return float("nan")
    return conserved / denom

