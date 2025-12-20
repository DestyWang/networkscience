from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Sequence

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


def FO_consistency(
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
    for i, node in enumerate(nodes_a):
        labels = label_lookup_a.get(node, set())
        if not labels:
            continue
        topk_idx = np.argsort(matrix[i])[::-1][: min(k, matrix.shape[1])]
        if any(labels & label_lookup_b.get(nodes_b[j], set()) for j in topk_idx):
            hits += 1
    total = len(nodes_a)
    return hits / total if total else float("nan")


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

    indices = np.argmax(matrix, axis=1)
    return {source_nodes[i]: target_nodes[j] for i, j in enumerate(indices)}


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


def S3_hard(
    graph_source: nx.Graph,
    graph_target: nx.Graph,
    coupling: np.ndarray | torch.Tensor,
    source_nodes: Sequence[str],
    target_nodes: Sequence[str],
) -> float:
    """
    S³ 指标（hard mapping 版本）。
    """

    P = _ensure_matrix(coupling)
    mapping = _argmax_mapping(P, source_nodes, target_nodes)
    conserved = _count_preserved_edges(graph_source, graph_target, mapping)
    denom = (
        graph_source.number_of_edges()
        + graph_target.number_of_edges()
        - conserved
    )
    if denom == 0:
        return float("nan")
    return conserved / denom


def S3_soft(
    graph_source: nx.Graph,
    graph_target: nx.Graph,
    coupling: np.ndarray | torch.Tensor,
    source_nodes: Sequence[str],
    target_nodes: Sequence[str],
) -> float:
    """
    S³ 指标（soft / coupling 版本）。
    """

    P = _ensure_matrix(coupling)
    if P.shape != (len(source_nodes), len(target_nodes)):
        raise ValueError("耦合矩阵尺寸与节点数量不一致。")

    As = nx.to_numpy_array(graph_source, nodelist=source_nodes, dtype=np.float64)
    At = nx.to_numpy_array(graph_target, nodelist=target_nodes, dtype=np.float64)

    overlap_matrix = P @ At @ P.T
    conserved = 0.5 * float(np.sum(As * overlap_matrix))
    denom = (
        graph_source.number_of_edges()
        + graph_target.number_of_edges()
        - conserved
    )
    if denom == 0:
        return float("nan")
    return conserved / denom



