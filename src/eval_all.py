from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from src.evaluate import (
    FO_con_hard,
    FO_con_soft,
    S3_hard,
    sim_score,
    topk_FO_recall,
)
from src.process import load_family_data
from src.utils import (
    COUPLING_FILE,
    METADATA_FILE,
    RESULTS_FILE,
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

DATASET_ROOT = Path(
    "/home/bcl/wanghongyu/other/networkscience/data/"
    "NAPAbench/NAPAbenchVer2/benchmarkDataset"
).resolve()
OUTPUT_ROOT = Path("/home/bcl/wanghongyu/other/networkscience/outputs").resolve()
EVAL_OUTPUT = OUTPUT_ROOT / "eval_all.csv"
SUITE_NAMES = {"2way", "5way", "8way"}

METRIC_COLUMNS = [
    "sim_soft",
    "sim_hard_source",
    "sim_hard_target",
    "fo_consistency_soft",
    "fo_consistency_hard_source",
    "fo_consistency_hard_target",
    "topk_recall@3",
    "topk_recall@5",
    "topk_recall@10",
    "s3_hard",
    "total_loss",
    "runtime_seconds",
]

FamilyKey = Tuple[str, str, str]
FamilyCache = Dict[FamilyKey, any]


def _iter_run_dirs(root: Path) -> Iterator[Tuple[str, str, str, str, Path]]:
    for suite_dir in sorted(root.iterdir()):
        if suite_dir.name not in SUITE_NAMES or not suite_dir.is_dir():
            continue
        for category_dir in sorted(suite_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            for family_dir in sorted(category_dir.iterdir()):
                if not family_dir.is_dir():
                    continue
                for pair_dir in sorted(family_dir.iterdir()):
                    if not pair_dir.is_dir():
                        continue
                    for run_dir in sorted(pair_dir.iterdir()):
                        if not run_dir.is_dir():
                            continue
                        if not (run_dir / COUPLING_FILE).exists():
                            continue
                        if not (run_dir / METADATA_FILE).exists():
                            continue
                        if not (run_dir / RESULTS_FILE).exists():
                            continue
                        yield (
                            suite_dir.name,
                            category_dir.name,
                            family_dir.name,
                            pair_dir.name,
                            run_dir,
                        )
            # if suite_dir.name == "8way":
            #     break


def _load_family(
    cache: FamilyCache,
    suite: str,
    category: str,
    family: str,
) -> any:
    key: FamilyKey = (suite, category, family)
    if key not in cache:
        family_path = DATASET_ROOT / suite / category / family
        cache[key] = load_family_data(family_path)
    return cache[key]


def _load_coupling(run_dir: Path) -> np.ndarray:
    tensor = torch.load(run_dir / COUPLING_FILE, map_location="cpu").detach().float()
    total = tensor.sum()
    if float(total) > 0:
        tensor = tensor / total
    return tensor.cpu().numpy()


def _row_hard(pi: np.ndarray) -> np.ndarray:
    if pi.size == 0:
        return pi
    hard = np.zeros_like(pi)
    idx = np.argmax(pi, axis=1)
    rows = np.arange(pi.shape[0])
    hard[rows, idx] = 1.0
    denom = hard.sum()
    if denom > 0:
        hard /= denom
    return hard


def _col_hard(pi: np.ndarray) -> np.ndarray:
    if pi.size == 0:
        return pi
    hard = np.zeros_like(pi)
    idx = np.argmax(pi, axis=0)
    cols = np.arange(pi.shape[1])
    hard[idx, cols] = 1.0
    denom = hard.sum()
    if denom > 0:
        hard /= denom
    return hard


def _resolve_sim_path(family_dir: Path, source: str, target: str) -> Path:
    candidate = family_dir / f"{source}-{target}.sim"
    if candidate.exists():
        return candidate
    reverse = family_dir / f"{target}-{source}.sim"
    if reverse.exists():
        return reverse
    raise FileNotFoundError(f"Missing similarity file for {source}-{target} in {family_dir}")


def _extract_total_loss(loss_terms: any, fallback: any) -> float:
    if isinstance(loss_terms, dict):
        total_series = loss_terms.get("total")
        if isinstance(total_series, (list, tuple)) and total_series:
            try:
                return float(total_series[-1])
            except (TypeError, ValueError):
                pass
        if isinstance(total_series, (int, float)):
            return float(total_series)
    try:
        return float(fallback)
    except (TypeError, ValueError):
        return float("nan")


def _to_float(value: any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _evaluate_run(
    suite: str,
    category: str,
    family: str,
    pair: str,
    run_dir: Path,
    cache: FamilyCache,
) -> Dict[str, float]:
    family_data = _load_family(cache, suite, category, family)
    metadata = json.loads((run_dir / METADATA_FILE).read_text(encoding="utf-8"))
    results_payload = json.loads((run_dir / RESULTS_FILE).read_text(encoding="utf-8"))

    pi = _load_coupling(run_dir)
    pi_hard_row = _row_hard(pi)
    pi_hard_col = _col_hard(pi)

    source_nodes: Sequence[str] = metadata["source_nodes"]
    target_nodes: Sequence[str] = metadata["target_nodes"]
    source_name, target_name = [token.upper() for token in pair.split("_vs_")]

    family_dir = DATASET_ROOT / suite / category / family
    sim_path = _resolve_sim_path(family_dir, source_name, target_name)

    src_annotations = family_data.annotations.get(source_name, {})
    tgt_annotations = family_data.annotations.get(target_name, {})

    metrics = {
        "sim_soft": sim_score(sim_path, pi, source_nodes, target_nodes),
        "sim_hard_source": sim_score(sim_path, pi_hard_row, source_nodes, target_nodes),
        "sim_hard_target": sim_score(sim_path, pi_hard_col, source_nodes, target_nodes),
        "fo_consistency_soft": FO_con_soft(
            pi,
            source_nodes,
            target_nodes,
            src_annotations,
            tgt_annotations,
        ),
        "fo_consistency_hard_source": FO_con_hard(
            pi,
            source_nodes,
            target_nodes,
            src_annotations,
            tgt_annotations,
            axis="source",
        ),
        "fo_consistency_hard_target": FO_con_hard(
            pi,
            source_nodes,
            target_nodes,
            src_annotations,
            tgt_annotations,
            axis="target",
        ),
        "topk_recall@3": topk_FO_recall(
            pi,
            source_nodes,
            target_nodes,
            src_annotations,
            tgt_annotations,
            k=3,
        ),
        "topk_recall@5": topk_FO_recall(
            pi,
            source_nodes,
            target_nodes,
            src_annotations,
            tgt_annotations,
            k=5,
        ),
        "topk_recall@10": topk_FO_recall(
            pi,
            source_nodes,
            target_nodes,
            src_annotations,
            tgt_annotations,
            k=10,
        ),
        "s3_hard": S3_hard(
            family_data.graphs[source_name],
            family_data.graphs[target_name],
            pi,
            source_nodes,
            target_nodes,
        ),
    }

    loss_terms = results_payload.get("loss_terms")
    metrics["total_loss"] = _extract_total_loss(
        loss_terms,
        results_payload.get("loss_value"),
    )
    metrics["runtime_seconds"] = _to_float(results_payload.get("runtime_seconds"))
    return metrics


def main() -> None:
    cache: FamilyCache = {}
    records: Dict[str, Dict[str, float]] = {}

    for suite, category, family, pair, run_dir in _iter_run_dirs(OUTPUT_ROOT):
        pair_id = f"{suite}/{category}/{family}/{pair}/{run_dir.name}"
        try:
            metrics = _evaluate_run(
                suite,
                category,
                family,
                pair,
                run_dir,
                cache,
            )
            records[pair_id] = metrics
            LOGGER.info("Evaluated %s", pair_id)
        except Exception as exc:  # pragma: no cover - robustness
            LOGGER.exception("Failed to evaluate %s: %s", pair_id, exc)

    if not records:
        LOGGER.warning("No evaluation records generated.")
        return

    df = pd.DataFrame.from_dict(records, orient="index")
    missing_cols = [col for col in METRIC_COLUMNS if col not in df.columns]
    for col in missing_cols:
        df[col] = np.nan
    df = df[METRIC_COLUMNS]
    df.index.name = "pair_id"
    EVAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(EVAL_OUTPUT)
    LOGGER.info("Saved %d evaluations to %s", len(df), EVAL_OUTPUT)


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=1 nohup python -u -m src.eval_all > eval_all.log 2>&1 &
# 1875766