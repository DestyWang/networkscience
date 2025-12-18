from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
from fugw.mappings import FUGW

from src.evaluate import (
    argmax_alignment,
    edge_correctness,
    node_correctness,
    precision_at_k,
    recall_at_k,
    s3_score,
    topk_alignment,
)
from src.features import build_fo_vocabulary, compute_node_features, standardize_pair
from src.geometry import shortest_path_distance_matrix
from src.process import DATASET_ROOT, FamilyData, load_family_data

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


@dataclass
class FeatureConfig:
    """
    特征计算相关的超参数。
    """

    fo_top_k: int = 64
    shortest_path_sample: int = 256
    betweenness_k: int = 256


@dataclass
class AlignmentConfig:
    """
    FUGW 对齐超参数。
    """

    source_id: str = "A"
    target_id: str = "B"
    alpha: float = 0.5
    rho: float = 1.0
    eps: float = 1e-4
    solver: str = "sinkhorn"
    solver_params: Dict[str, Any] = field(
        default_factory=lambda: {"tol_uot": 1e-10}
    )
    device: str = "auto"
    verbose: bool = True


def _strip_prefix(node_id: str) -> str:
    """
    去掉节点名的字母前缀，用于推断真值映射。
    """

    i = 0
    while i < len(node_id) and not node_id[i].isdigit():
        i += 1
    return node_id[i:]


def infer_ground_truth_by_suffix(
    source_nodes: Sequence[str],
    target_nodes: Sequence[str],
) -> Dict[str, str]:
    """
    假定同一数字后缀的节点为真值对应（NAPAbench 常见约定）。
    """

    tgt_lookup = {_strip_prefix(t): t for t in target_nodes}
    mapping: Dict[str, str] = {}
    for s in source_nodes:
        candidate = tgt_lookup.get(_strip_prefix(s))
        if candidate:
            mapping[s] = candidate
    return mapping


def _prepare_features_and_geometry(
    source_graph: nx.Graph,
    target_graph: nx.Graph,
    source_annotations: Dict[str, List[str]],
    target_annotations: Dict[str, List[str]],
    feature_config: FeatureConfig,
    feature_weights: Optional[np.ndarray] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    List[str],
    List[str],
    np.ndarray,
    List[str],
]:
    """
    生成特征张量、几何矩阵与节点顺序。

    Returns
    -------
    (
      source_features, target_features,
      source_geometry, target_geometry,
      source_nodes, target_nodes,
      applied_weights, feature_names
    )
    """

    fo_vocab = build_fo_vocabulary(
        [source_annotations, target_annotations],
        top_k=feature_config.fo_top_k,
    )

    src_feats, feat_names, src_nodes = compute_node_features(
        graph=source_graph,
        annotations=source_annotations,
        fo_vocab=fo_vocab,
        shortest_path_sample=feature_config.shortest_path_sample,
        betweenness_k=feature_config.betweenness_k,
    )
    tgt_feats, _, tgt_nodes = compute_node_features(
        graph=target_graph,
        annotations=target_annotations,
        fo_vocab=fo_vocab,
        shortest_path_sample=feature_config.shortest_path_sample,
        betweenness_k=feature_config.betweenness_k,
    )

    src_norm, tgt_norm, applied_weights = standardize_pair(
        source_features=src_feats,
        target_features=tgt_feats,
        weights=feature_weights,
    )

    src_geom = shortest_path_distance_matrix(
        graph=source_graph,
        node_list=src_nodes,
    )
    tgt_geom = shortest_path_distance_matrix(
        graph=target_graph,
        node_list=tgt_nodes,
    )

    source_features = torch.from_numpy(src_norm.T.copy())
    target_features = torch.from_numpy(tgt_norm.T.copy())
    source_geometry = torch.from_numpy(src_geom)
    target_geometry = torch.from_numpy(tgt_geom)

    return (
        source_features,
        target_features,
        source_geometry,
        target_geometry,
        src_nodes,
        tgt_nodes,
        applied_weights,
        feat_names,
    )


def run_fugw_alignment(
    source_graph: nx.Graph,
    target_graph: nx.Graph,
    source_annotations: Dict[str, List[str]],
    target_annotations: Dict[str, List[str]],
    feature_config: FeatureConfig,
    align_config: AlignmentConfig,
    *,
    feature_weights: Optional[np.ndarray] = None,
    ground_truth: Optional[Mapping[str, str]] = None,
    infer_truth: bool = True,
    eval_top_k: int = 5,
) -> Dict[str, Any]:
    """
    核心管线：生成特征/几何 -> 运行 FUGW -> 输出耦合与评估。

    Returns
    -------
    Dict[str, Any]
        包含 pi/mapping/topk/metrics 及中间产物。
    """

    (
        source_features,
        target_features,
        source_geometry,
        target_geometry,
        src_nodes,
        tgt_nodes,
        used_weights,
        feature_names,
    ) = _prepare_features_and_geometry(
        source_graph=source_graph,
        target_graph=target_graph,
        source_annotations=source_annotations,
        target_annotations=target_annotations,
        feature_config=feature_config,
        feature_weights=feature_weights,
    )

    ns, nt = len(src_nodes), len(tgt_nodes)
    source_weights = torch.full((ns,), 1.0 / max(ns, 1), dtype=torch.float32)
    target_weights = torch.full((nt,), 1.0 / max(nt, 1), dtype=torch.float32)

    model = FUGW(
        alpha=align_config.alpha,
        rho=align_config.rho,
        eps=align_config.eps,
    )
    model.fit(
        source_features=source_features,
        target_features=target_features,
        source_geometry=source_geometry,
        target_geometry=target_geometry,
        source_weights=source_weights,
        target_weights=target_weights,
        solver=align_config.solver,
        solver_params=align_config.solver_params,
        device=align_config.device,
        verbose=align_config.verbose,
    )

    coupling = model.pi.detach().cpu()
    mapping = argmax_alignment(coupling, src_nodes, tgt_nodes)
    topk = topk_alignment(coupling, src_nodes, tgt_nodes, k=eval_top_k)

    if ground_truth is None and infer_truth:
        ground_truth = infer_ground_truth_by_suffix(src_nodes, tgt_nodes)

    metrics = {}
    if ground_truth:
        metrics = {
            "node_correctness": node_correctness(mapping, ground_truth),
            "recall_at_k": recall_at_k(topk, ground_truth, k=eval_top_k),
            "precision_at_k": precision_at_k(topk, ground_truth, k=eval_top_k),
            "edge_correctness": edge_correctness(source_graph, target_graph, mapping),
            "s3": s3_score(source_graph, target_graph, mapping),
        }

    return {
        "pi": coupling,
        "mapping": mapping,
        "topk": topk,
        "metrics": metrics,
        "source_nodes": src_nodes,
        "target_nodes": tgt_nodes,
        "feature_names": feature_names,
        "feature_weights": used_weights,
    }


def align_family(
    family_dir: Path,
    feature_config: FeatureConfig = FeatureConfig(),
    align_config: AlignmentConfig = AlignmentConfig(),
    *,
    feature_weights: Optional[np.ndarray] = None,
    ground_truth: Optional[Mapping[str, str]] = None,
    infer_truth: bool = True,
    eval_top_k: int = 5,
) -> Dict[str, Any]:
    """
    针对单个 family（默认 A/B）运行完整对齐。
    """

    family_dir = Path(family_dir).resolve()
    family: FamilyData = load_family_data(family_dir)

    sid = align_config.source_id.upper()
    tid = align_config.target_id.upper()
    if sid not in family.graphs or tid not in family.graphs:
        raise KeyError(f"未找到 {sid}/{tid} 网络，请检查 family 目录。")

    result = run_fugw_alignment(
        source_graph=family.graphs[sid],
        target_graph=family.graphs[tid],
        source_annotations=family.annotations.get(sid, {}),
        target_annotations=family.annotations.get(tid, {}),
        feature_config=feature_config,
        align_config=align_config,
        feature_weights=feature_weights,
        ground_truth=ground_truth,
        infer_truth=infer_truth,
        eval_top_k=eval_top_k,
    )
    return result


if __name__ == "__main__":
    print("Test")

