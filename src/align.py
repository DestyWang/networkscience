from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from numbers import Number
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import torch
from fugw.mappings import FUGW

from src.features import get_node_features
from src.process import get_dist_matrix
from src.utils import (
    load_cached_features,
    prepare_output_dir,
    save_alignment_inputs,
    save_alignment_results,
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


@dataclass
class FeatureConfig:
    """
    Hyper-parameters that control the feature standardization process.
    """

    normalize: bool = True
    eps: float = 1e-8


@dataclass
class AlignmentConfig:
    """
    Configuration for the FUGW solver.
    """

    alpha: float = 0.9
    rho: float = 0.1
    eps: float = 1e-5
    solver: str = "sinkhorn"
    solver_params: Dict[str, Any] = field(
        default_factory=lambda: {"nits_bcd": 25, "nits_uot": 1000, "tol_bcd": 1e-4,"tol_uot": 1e-10}
    )
    device: str = "cuda"
    verbose: bool = True


def _standardize_pair(
    source_features: torch.Tensor | Any,
    target_features: torch.Tensor | Any,
    *,
    eps: float,
    normalize: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optionally center/scale both feature matrices together.
    """

    src = torch.as_tensor(source_features, dtype=torch.float32)
    tgt = torch.as_tensor(target_features, dtype=torch.float32)
    if not normalize:
        return src, tgt

    combined = torch.cat([src, tgt], dim=0)
    mean = combined.mean(dim=0)
    std = combined.std(dim=0, unbiased=False)
    std = torch.where(std < eps, torch.ones_like(std), std)
    src = (src - mean) / std
    tgt = (tgt - mean) / std
    return src, tgt


def _as_torch_feature_matrix(features: torch.Tensor | Any) -> torch.Tensor:
    """
    Convert feature matrix shaped (n, d) into the (d, n) tensor FUGW expects.
    """

    tensor = torch.as_tensor(features, dtype=torch.float32)
    if tensor.numel() == 0:
        return torch.empty((0, 0), dtype=torch.float32)
    return tensor.T.contiguous()


def _as_torch_geometry(matrix: torch.Tensor | Any) -> torch.Tensor:
    """
    Ensure distance matrices are torch tensors with float32 dtype.
    """

    return torch.as_tensor(matrix, dtype=torch.float32)


def _coerce_scalar(value: Any) -> float:
    """
    Safely cast scalar-like objects into float (shape=()).

    Parameters
    ----------
    value : Any
        Single numeric value or tensor（shape=()）。

    Returns
    -------
    float
        标量结果，shape=()；若无法转换，则返回 NaN。
    """

    if isinstance(value, Number):
        return float(value)
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return float("nan")
    return float("nan")


def _normalize_loss_terms(loss_payload: Any) -> Optional[Dict[str, Any]]:
    """
    Convert `model.loss` payload into JSON-safe dict of floats.

    Parameters
    ----------
    loss_payload : Any
        原始 `model.loss` 对象，通常为 Dict[str, Sequence[number]]。

    Returns
    -------
    Optional[Dict[str, Any]]
        若可解析，则返回 {str: List[float] 或 float shape=()} 的字典；
        否则返回 None。
    """

    if not isinstance(loss_payload, dict):
        return None

    normalized: Dict[str, Any] = {}
    for key, series in loss_payload.items():
        if isinstance(series, dict):
            normalized[key] = _normalize_loss_terms(series)
            continue
        if isinstance(series, (list, tuple)):
            normalized[key] = [_coerce_scalar(item) for item in series]
            continue
        if torch.is_tensor(series):
            normalized[key] = _coerce_scalar(series)
            continue
        normalized[key] = _coerce_scalar(series)
    return normalized

def sim_to_cost(
    S: torch.Tensor,
    q_low: float = 0.01,
    q_high: float = 0.99,
    eps: float = 1e-8,
    scale_median_to_1: bool = True,
) -> torch.Tensor:
    """
    将非负相似度矩阵 S 转换为 OT 可用的 cost matrix C（同形状）。

    参数
    ----------
    S : torch.Tensor, shape (n, m)
        非负相似度分数矩阵（允许 0；数值可很大，如几百/几千）。
    q_low : float
        低分位数裁剪（默认 1%），用于鲁棒归一化。
    q_high : float
        高分位数裁剪（默认 99%），用于鲁棒归一化。
    eps : float
        数值稳定项，避免 log(0)。建议 1e-12 ~ 1e-6。
    scale_median_to_1 : bool
        是否将输出 C 再缩放，使 median(C)=1，便于后续 Sinkhorn 等算法调参。

    返回
    ----
    C : torch.Tensor, shape (n, m)
        非负 cost matrix；越相似 cost 越小。
    """
    if not isinstance(S, torch.Tensor):
        S = torch.as_tensor(S)
    if S.ndim != 2:
        raise ValueError(f"S 必须是二维矩阵，当前 S.ndim={S.ndim}")
    if torch.any(S < 0):
        raise ValueError("S 必须非负（>=0）")
    
    S = S.to(dtype=torch.float64)
    # 1) 压缩动态范围：log(1+S)
    T = torch.log1p(S)
    
    # 2) 分位数裁剪 + 归一化到 [0,1]
    a = torch.quantile(T, q_low)
    b = torch.quantile(T, q_high)
    denom = max((b - a).item(), eps)
    U = (T - a) / denom
    U = U.clamp(0.0, 1.0)
    
    # 3) 转为代价：-log(U + eps)
    C = -torch.log(U + eps)
    
    # 可选：把尺度归一到 median(C)=1，方便后续算法（尤其 Sinkhorn）稳定
    if scale_median_to_1:
        med = torch.median(C)
        if med > 0:
            C = C / med
    
    return C

def fugw_align(
    source_graph: nx.Graph,
    target_graph: nx.Graph,
    *,
    feature_config: FeatureConfig = FeatureConfig(),
    align_config: AlignmentConfig = AlignmentConfig(),
    use_unit_weights: bool = True,
    feature_path: str = "",
    output_dir: str = "/home/bcl/wanghongyu/other/networkscience/data/NAPAbench/outputs",
    sim_mat: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run FUGW using structural features + geodesic distances.

    Returns
    -------
    Dict[str, Any]
        Keys: `pi`, `source_nodes`, `target_nodes`, `feature_names`,
        `source_features`, `target_features`, `source_geometry`, `target_geometry`.
    """

    output_path, current_time = prepare_output_dir(output_dir)

    cached_data = load_cached_features(feature_path) if feature_path else None
    if cached_data:
        current_src_nodes: List[str] = list(source_graph.nodes())
        current_tgt_nodes: List[str] = list(target_graph.nodes())
        if (
            cached_data.source_nodes != current_src_nodes
            or cached_data.target_nodes != current_tgt_nodes
        ):
            LOGGER.warning(
                "Cached features at %s do not match current graphs, recomputing.",
                feature_path,
            )
            cached_data = None

    if cached_data:
        LOGGER.info("Loaded cached features from %s", feature_path)
        src_features = cached_data.source_features
        tgt_features = cached_data.target_features
        feature_names = cached_data.feature_names
        src_nodes = cached_data.source_nodes
        tgt_nodes = cached_data.target_nodes
        src_geometry = cached_data.source_geometry
        tgt_geometry = cached_data.target_geometry
    else:
        if feature_path:
            LOGGER.info("Failed to load cached features from %s, recomputing.", feature_path)
        src_features_raw, feature_names, src_nodes = get_node_features(source_graph)
        tgt_features_raw, _, tgt_nodes = get_node_features(target_graph)
        src_features, tgt_features = _standardize_pair(
            src_features_raw,
            tgt_features_raw,
            eps=feature_config.eps,
            normalize=feature_config.normalize,
        )
        src_geometry, _ = get_dist_matrix(source_graph, node_order=src_nodes)
        tgt_geometry, _ = get_dist_matrix(target_graph, node_order=tgt_nodes)

    LOGGER.info(
        "Source features: %s, Target features: %s",
        tuple(src_features.shape),
        tuple(tgt_features.shape),
    )
    LOGGER.info(
        "Source geometry: %s, Target geometry: %s",
        tuple(src_geometry.shape),
        tuple(tgt_geometry.shape),
    )

    source_features_tensor = _as_torch_feature_matrix(src_features)
    target_features_tensor = _as_torch_feature_matrix(tgt_features)
    source_geometry_tensor = _as_torch_geometry(src_geometry)
    target_geometry_tensor = _as_torch_geometry(tgt_geometry)

    ns, nt = len(src_nodes), len(tgt_nodes)

    metadata = {
        "feature_names": feature_names,
        "source_nodes": src_nodes,
        "target_nodes": tgt_nodes,
    }
    params_payload = {
        "timestamp": current_time,
        "output_path": str(output_path),
        "feature_cache_path": feature_path or "",
        "feature_config": asdict(feature_config),
        "align_config": asdict(align_config),
        "source_node_count": ns,
        "target_node_count": nt,
    }
    artifact_paths = save_alignment_inputs(
        output_path,
        source_features=src_features,
        target_features=tgt_features,
        source_geometry=src_geometry,
        target_geometry=tgt_geometry,
        metadata=metadata,
        params=params_payload,
    )

    sim_tensor: Optional[torch.Tensor] = None
    if sim_mat is not None:
        sim_tensor = torch.as_tensor(sim_mat, dtype=torch.float32)
        if sim_tensor.shape != (ns, nt):
            raise ValueError(
                "sim_mat shape must match (len(source_nodes), len(target_nodes))."
            )
        cost_matrix = sim_to_cost(sim_tensor)

    if ns == 0 or nt == 0:
        LOGGER.warning("No nodes in source or target graph, returning empty coupling matrix")
        empty_coupling = torch.zeros((ns, nt), dtype=torch.float32)
        save_alignment_results(
            output_path,
            coupling=empty_coupling,
            status="empty_graph",
            artifact_paths=artifact_paths,
            extra={
                "timestamp": current_time,
                "feature_cache_path": feature_path or "",
            },
        )
        return {
            "pi": empty_coupling,
            "source_nodes": src_nodes,
            "target_nodes": tgt_nodes,
            "feature_names": feature_names,
            "source_features": src_features,
            "target_features": tgt_features,
            "source_geometry": src_geometry,
            "target_geometry": tgt_geometry,
            "output_path": str(output_path),
        }

    if use_unit_weights:
        print("Using unit weights")
        source_weights = torch.full((ns,), 1.0 / ns, dtype=torch.float32)
        target_weights = torch.full((nt,), 1.0 / nt, dtype=torch.float32)
    elif sim_tensor is not None:
        row_sums = sim_tensor.sum(dim=1)
        col_sums = sim_tensor.sum(dim=0)
        total_mass = row_sums.sum()
        if total_mass <= 0:
            LOGGER.warning("sim_mat mass is zero; falling back to uniform marginals.")
            source_weights = torch.full((ns,), 1.0 / ns, dtype=torch.float32)
            target_weights = torch.full((nt,), 1.0 / nt, dtype=torch.float32)
        else:
            source_weights = row_sums / total_mass
            target_weights = col_sums / total_mass
    else:
        source_weights = torch.full((ns,), 1.0 / ns, dtype=torch.float32)
        target_weights = torch.full((nt,), 1.0 / nt, dtype=torch.float32)

    model = FUGW(
        alpha=align_config.alpha,
        rho=align_config.rho,
        eps=align_config.eps,
    )
    start_time = perf_counter()
    model.fit(
        source_features=source_features_tensor,
        target_features=target_features_tensor,
        cost_matrix=cost_matrix,
        source_geometry=source_geometry_tensor,
        target_geometry=target_geometry_tensor,
        source_weights=source_weights,
        target_weights=target_weights,
        solver=align_config.solver,
        solver_params=align_config.solver_params,
        device=align_config.device,
        verbose=align_config.verbose,
    )
    runtime_seconds = perf_counter() - start_time
    loss_terms = _normalize_loss_terms(getattr(model, "loss", None))
    if loss_terms and isinstance(loss_terms.get("total"), list) and loss_terms["total"]:
        loss_value = float(loss_terms["total"][-1])
    elif loss_terms and isinstance(loss_terms.get("total"), (int, float)):
        loss_value = float(loss_terms["total"])
    else:
        loss_value = _coerce_scalar(getattr(model, "loss", float("nan")))
    loss_history_raw = getattr(model, "loss_steps", None)
    loss_history = (
        [float(item) for item in loss_history_raw] if loss_history_raw is not None else []
    )

    coupling = model.pi.detach().cpu()
    extra_payload: Dict[str, Any] = {
        "timestamp": current_time,
        "feature_cache_path": feature_path or "",
        "runtime_seconds": runtime_seconds,
        "loss_value": loss_value,
        "loss_history": loss_history,
    }
    if loss_terms is not None:
        extra_payload["loss_terms"] = loss_terms

    save_alignment_results(
        output_path,
        coupling=coupling,
        status="ok",
        artifact_paths=artifact_paths,
        extra=extra_payload,
    )
    return {
        "pi": coupling,
        "source_nodes": src_nodes,
        "target_nodes": tgt_nodes,
        "feature_names": feature_names,
        "source_features": src_features,
        "target_features": tgt_features,
        "source_geometry": src_geometry,
        "target_geometry": tgt_geometry,
        "output_path": str(output_path),
        "loss": loss_value,
        "loss_terms": loss_terms,
        "loss_history": loss_history,
        "runtime_seconds": runtime_seconds,
    }
