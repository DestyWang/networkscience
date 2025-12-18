from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np
import torch


def argmax_alignment(coupling: torch.Tensor, source_nodes: Sequence[str], target_nodes: Sequence[str],
) -> Dict[str, str]:
    """
    基于耦合矩阵行最大值生成确定性映射。

    Returns
    -------
    Dict[str, str]
        source->target 的一一映射（shape: |source_nodes| 键）。
    """

    if coupling.shape[0] != len(source_nodes):
        raise ValueError("coupling 行数与 source_nodes 不一致")
    if coupling.shape[1] != len(target_nodes):
        raise ValueError("coupling 列数与 target_nodes 不一致")

    _, indices = torch.max(coupling, dim=1)
    return {
        source_nodes[i]: target_nodes[int(j)]
        for i, j in enumerate(indices.cpu().numpy())
    }


def topk_alignment(coupling: torch.Tensor, source_nodes: Sequence[str], target_nodes: Sequence[str], k: int = 5,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    返回每个源节点的 Top-k 目标候选及其分数。
    """

    values, indices = torch.topk(coupling, k=k, dim=1)
    result: Dict[str, List[Tuple[str, float]]] = {}
    for i, src in enumerate(source_nodes):
        pairs = [
            (target_nodes[int(j)], float(values[i, pos].cpu().item()))
            for pos, j in enumerate(indices[i])
        ]
        result[src] = pairs
    return result


def node_correctness(mapping: Mapping[str, str], ground_truth: Mapping[str, str],
) -> float:
    """
    NC = 正确匹配的比例。
    """

    if not ground_truth:
        return float("nan")
    correct = sum(1 for k, v in mapping.items() if ground_truth.get(k) == v)
    return correct / max(len(ground_truth), 1)


def recall_at_k(topk: Mapping[str, List[Tuple[str, float]]], ground_truth: Mapping[str, str], k: int = 5,
) -> float:
    """
    Recall@k：真值是否出现在前 k。
    """

    if not ground_truth:
        return float("nan")
    hits = 0
    for src, true_tgt in ground_truth.items():
        preds = topk.get(src, [])[:k]
        if any(p[0] == true_tgt for p in preds):
            hits += 1
    return hits / max(len(ground_truth), 1)


def precision_at_k(topk: Mapping[str, List[Tuple[str, float]]], ground_truth: Mapping[str, str], k: int = 5,
) -> float:
    """
    Precision@k：Top-k 中正确的比例。
    """

    if not ground_truth:
        return float("nan")
    total = 0
    correct = 0
    for src, preds in topk.items():
        chosen = preds[:k]
        total += len(chosen)
        correct += sum(1 for tgt, _ in chosen if ground_truth.get(src) == tgt)
    return correct / max(total, 1)


def _count_mapped_edges(graph_source: nx.Graph, graph_target: nx.Graph, mapping: Mapping[str, str],
) -> int:
    """
    统计源图边在靶图中是否仍为边。
    """

    count = 0
    for u, v in graph_source.edges():
        mu, mv = mapping.get(u), mapping.get(v)
        if mu is None or mv is None:
            continue
        if graph_target.has_edge(mu, mv):
            count += 1
    return count


def edge_correctness(graph_source: nx.Graph, graph_target: nx.Graph, mapping: Mapping[str, str],
) -> float:
    """
    EC = 保留的源边 / 源边总数。
    """

    if graph_source.number_of_edges() == 0:
        return float("nan")
    conserved = _count_mapped_edges(graph_source, graph_target, mapping)
    return conserved / graph_source.number_of_edges()


def s3_score(graph_source: nx.Graph, graph_target: nx.Graph, mapping: Mapping[str, str],
) -> float:
    """
    S3 = conserved / (|E_s| + |E_t| - conserved)
    """

    conserved = _count_mapped_edges(graph_source, graph_target, mapping)
    denom = graph_source.number_of_edges() + graph_target.number_of_edges() - conserved
    if denom == 0:
        return float("nan")
    return conserved / denom


def drop_edges(graph: nx.Graph, drop_fraction: float, rng: np.random.Generator | None = None,
) -> nx.Graph:
    """
    随机删除一定比例的边，用于鲁棒性评估。
    """

    rng = rng or np.random.default_rng()
    g_copy = graph.copy()
    edges = list(g_copy.edges())
    if not edges or drop_fraction <= 0:
        return g_copy
    k = int(len(edges) * drop_fraction)
    to_remove = rng.choice(len(edges), size=k, replace=False)
    for idx in to_remove:
        g_copy.remove_edge(*edges[int(idx)])
    return g_copy


@dataclass
class RobustnessResult:
    fractions: List[float]
    metric: List[float]
    auc: float


def robustness_curve(
    align_fn: Callable[[nx.Graph, nx.Graph], Mapping[str, str]],
    metric_fn: Callable[[nx.Graph, nx.Graph, Mapping[str, str]], float],
    source_graph: nx.Graph,
    target_graph: nx.Graph,
    *,
    fractions: Iterable[float] = (0.0, 0.1, 0.2, 0.3),
    repeats: int = 1,
    seed: int = 42,
) -> RobustnessResult:
    """
    对不同删边比例的性能进行积分，输出 AUC。

    Parameters
    ----------
    align_fn : Callable
        输入 (source_graph, target_graph) 返回映射的函数。
    metric_fn : Callable
        输入 (source_graph, target_graph, mapping) 返回标量指标。
    source_graph : nx.Graph
        源图。
    target_graph : nx.Graph
        靶图。
    fractions : Iterable[float]
        删边比例列表。
    repeats : int
        每个比例重复次数。
    seed : int
        随机种子。

    Returns
    -------
    RobustnessResult
        fractions/metric/AUC。
    """

    rng = np.random.default_rng(seed)
    frac_list = list(fractions)
    scores: List[float] = []
    for frac in frac_list:
        rep_scores = []
        for _ in range(repeats):
            gs = drop_edges(source_graph, frac, rng)
            gt = drop_edges(target_graph, frac, rng)
            mapping = align_fn(gs, gt)
            rep_scores.append(metric_fn(gs, gt, mapping))
        scores.append(float(np.nanmean(rep_scores)))

    auc = float(np.trapz(scores, frac_list))
    return RobustnessResult(fractions=frac_list, metric=scores, auc=auc)

