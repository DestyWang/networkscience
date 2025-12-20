from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


LOGGER = logging.getLogger(__name__)

SOURCE_FEATURE_FILE = "source_features.pt"
TARGET_FEATURE_FILE = "target_features.pt"
SOURCE_GEOMETRY_FILE = "source_geometry.pt"
TARGET_GEOMETRY_FILE = "target_geometry.pt"
METADATA_FILE = "metadata.json"
PARAMS_FILE = "params.json"
RESULTS_FILE = "results.json"
COUPLING_FILE = "coupling.pt"


@dataclass
class AlignmentCache:
    source_features: torch.Tensor
    target_features: torch.Tensor
    source_geometry: torch.Tensor
    target_geometry: torch.Tensor
    feature_names: List[str]
    source_nodes: List[str]
    target_nodes: List[str]


def tensor_to_numpy(array: Any) -> np.ndarray:
    """
    Convert tensors (torch) or array-like inputs into a NumPy array.
    """

    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    if isinstance(array, np.ndarray):
        return array
    return np.asarray(array)


def prepare_output_dir(
    root: str | Path,
    timestamp: Optional[str] = None,
) -> Tuple[Path, str]:
    """
    Create (if needed) and return a timestamped output directory.
    """

    root_path = Path(root).expanduser().resolve()
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = root_path / timestamp
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path, timestamp


def _save_tensor(tensor: torch.Tensor, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu(), path)
    return str(path)


def _save_json(data: Dict[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def save_alignment_inputs(
    output_dir: str | Path,
    *,
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_geometry: torch.Tensor,
    target_geometry: torch.Tensor,
    metadata: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, str]:
    """
    Persist feature/geometry tensors plus metadata & params into output_dir.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = {
        "source_feature_file": _save_tensor(
            source_features, output_path / SOURCE_FEATURE_FILE
        ),
        "target_feature_file": _save_tensor(
            target_features, output_path / TARGET_FEATURE_FILE
        ),
        "source_geometry_file": _save_tensor(
            source_geometry, output_path / SOURCE_GEOMETRY_FILE
        ),
        "target_geometry_file": _save_tensor(
            target_geometry, output_path / TARGET_GEOMETRY_FILE
        ),
    }
    files["metadata_file"] = _save_json(metadata, output_path / METADATA_FILE)
    files["params_file"] = _save_json(params, output_path / PARAMS_FILE)
    return files


def save_alignment_results(
    output_dir: str | Path,
    *,
    coupling: torch.Tensor,
    status: str,
    artifact_paths: Dict[str, str],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Save coupling tensor and write a summary JSON that references all artifacts.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    coupling_path = output_path / COUPLING_FILE
    torch.save(coupling.detach().cpu(), coupling_path)

    payload: Dict[str, Any] = {
        "status": status,
        "coupling_file": str(coupling_path),
    }
    payload.update(artifact_paths)
    if extra:
        payload.update(extra)
    payload.setdefault("timestamp", datetime.now().isoformat())

    results_path = output_path / RESULTS_FILE
    results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "coupling_file": str(coupling_path),
        "results_file": str(results_path),
    }


def load_cached_features(cache_dir: str | Path) -> Optional[AlignmentCache]:
    """
    Attempt to load pre-computed features/geometry/metadata from cache_dir.
    """

    if not cache_dir:
        return None
    cache_path = Path(cache_dir).expanduser().resolve()
    if not cache_path.is_dir():
        return None

    try:
        metadata_path = cache_path / METADATA_FILE
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source_features = torch.load(
            cache_path / SOURCE_FEATURE_FILE, map_location="cpu"
        )
        target_features = torch.load(
            cache_path / TARGET_FEATURE_FILE, map_location="cpu"
        )
        source_geometry = torch.load(
            cache_path / SOURCE_GEOMETRY_FILE, map_location="cpu"
        )
        target_geometry = torch.load(
            cache_path / TARGET_GEOMETRY_FILE, map_location="cpu"
        )
    except FileNotFoundError:
        return None
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to load cached features from %s: %s", cache_path, exc)
        return None

    return AlignmentCache(
        source_features=source_features,
        target_features=target_features,
        source_geometry=source_geometry,
        target_geometry=target_geometry,
        feature_names=metadata.get("feature_names", []),
        source_nodes=metadata.get("source_nodes", []),
        target_nodes=metadata.get("target_nodes", []),
    )
