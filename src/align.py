from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

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

    alpha: float = 0.5
    rho: float = 10
    eps: float = 1e-4
    solver: str = "sinkhorn"
    solver_params: Dict[str, Any] = field(
        default_factory=lambda: {"nits_bcd": 20, "nits_uot": 1000, "tol_bcd": 1e-4,"tol_uot": 1e-10}
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


def fugw_align(
    source_graph: nx.Graph,
    target_graph: nx.Graph,
    *,
    feature_config: FeatureConfig = FeatureConfig(),
    align_config: AlignmentConfig = AlignmentConfig(),
    feature_path: str = "",
    output_dir: str = "/home/bcl/wanghongyu/other/networkscience/data/NAPAbench/outputs",
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
        current_src_nodes: List[str] = sorted(source_graph.nodes())
        current_tgt_nodes: List[str] = sorted(target_graph.nodes())
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

    source_weights = torch.full((ns,), 1.0 / ns, dtype=torch.float32)
    target_weights = torch.full((nt,), 1.0 / nt, dtype=torch.float32)

    model = FUGW(
        alpha=align_config.alpha,
        rho=align_config.rho,
        eps=align_config.eps,
    )
    model.fit(
        source_features=source_features_tensor,
        target_features=target_features_tensor,
        source_geometry=source_geometry_tensor,
        target_geometry=target_geometry_tensor,
        source_weights=source_weights,
        target_weights=target_weights,
        solver=align_config.solver,
        solver_params=align_config.solver_params,
        device=align_config.device,
        verbose=align_config.verbose,
    )

    coupling = model.pi.detach().cpu()
    save_alignment_results(
        output_path,
        coupling=coupling,
        status="ok",
        artifact_paths=artifact_paths,
        extra={
            "timestamp": current_time,
            "feature_cache_path": feature_path or "",
        },
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
    }
