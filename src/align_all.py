from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from src.align import AlignmentConfig, FeatureConfig, fugw_align
from src.process import load_family_data

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


@dataclass(frozen=True)
class FamilyJob:
    """
    描述一个需要执行对齐的 family。
    """

    suite: str
    category: str
    name: str
    path: Path


def _enumerate_family_jobs(dataset_root: Path) -> List[FamilyJob]:
    """
    列出数据集中所有 family。
    """

    jobs: List[FamilyJob] = []
    for suite_dir in sorted(dataset_root.iterdir()):
        if not suite_dir.is_dir():
            continue
        for category_dir in sorted(suite_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            for family_dir in sorted(category_dir.iterdir()):
                if not family_dir.is_dir():
                    continue
                jobs.append(
                    FamilyJob(
                        suite=suite_dir.name,
                        category=category_dir.name,
                        name=family_dir.name,
                        path=family_dir,
                    )
                )
    return jobs


def _filter_jobs(
    jobs: Sequence[FamilyJob],
    suites: Sequence[str] | None,
    categories: Sequence[str] | None,
    families: Sequence[str] | None,
) -> List[FamilyJob]:
    """
    根据命令行过滤条件筛选 family。
    """

    def _match(value: str, patterns: Sequence[str] | None) -> bool:
        return not patterns or value in patterns

    filtered: List[FamilyJob] = []
    for job in jobs:
        if not _match(job.suite, suites):
            continue
        if not _match(job.category, categories):
            continue
        if not _match(job.name, families):
            continue
        filtered.append(job)
    return filtered


def _available_networks(graph_keys: Iterable[str]) -> List[str]:
    """
    将 network 名称排序后返回。
    """

    return sorted({key.upper() for key in graph_keys})


def _pairwise_jobs(network_names: Sequence[str]) -> List[Tuple[str, str]]:
    """
    针对给定的 network 列表生成唯一的两两组合。
    """

    if len(network_names) < 2:
        return []
    return list(combinations(network_names, 2))


def _pair_label(source: str, target: str) -> str:
    """
    构造 pair 标签。
    """

    return f"{source}_vs_{target}"


def _extract_total_loss(loss_terms: Any, fallback: Any = float("nan")) -> float:
    """
    从 loss_terms 中抽取最终 total 损失值（shape=()）。

    Parameters
    ----------
    loss_terms : Any
        `model.loss` 归一化字典，或其他任意类型。
    fallback : Any
        兜底标量（shape=()），无法解析字典时使用。

    Returns
    -------
    float
        total 列表的末项（shape=()），若缺失则返回 fallback 或 NaN。
    """

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


def _write_family_results(
    family_dir: Path,
    job: FamilyJob,
    pair_records: Sequence[Dict[str, Any]],
) -> Path:
    """
    将 family 内所有 pair 的摘要写入 results.json。
    """

    payload: Dict[str, Any] = {
        "suite": job.suite,
        "category": job.category,
        "family": job.name,
        "pair_count": len(pair_records),
        "pair_results": pair_records,
    }
    results_path = family_dir / "results.json"
    family_dir.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return results_path


def _build_pair_record(
    job: FamilyJob,
    source: str,
    target: str,
    output_path: str,
    loss_value: float,
    loss_terms: Any,
    runtime_seconds: float,
    source_nodes: Sequence[str],
    target_nodes: Sequence[str],
) -> Dict[str, Any]:
    """
    整理单次对齐运行的摘要。

    Parameters
    ----------
    loss_value : float
        FUGW 最终损失（shape=()）。
    runtime_seconds : float
        运行耗时（shape=()）。
    """

    output_dir = Path(output_path).resolve()
    return {
        "suite": job.suite,
        "category": job.category,
        "family": job.name,
        "pair": _pair_label(source, target),
        "source": source,
        "target": target,
        "timestamp": output_dir.name,
        "output_path": str(output_dir),
        "loss": loss_value,
        "loss_terms": loss_terms,
        "runtime_seconds": runtime_seconds,
        "source_nodes": len(source_nodes),
        "target_nodes": len(target_nodes),
    }


def _align_family(
    job: FamilyJob,
    output_root: Path,
    feature_config: FeatureConfig,
    align_config: AlignmentConfig,
) -> None:
    """
    对单个 family 执行所有所需 pair 的对齐。
    """

    LOGGER.info(
        "Processing %s/%s/%s",
        job.suite,
        job.category,
        job.name,
    )
    family_data = load_family_data(job.path)
    network_names = _available_networks(family_data.graphs.keys())
    pair_jobs = _pairwise_jobs(network_names)
    if not pair_jobs:
        LOGGER.warning("Family %s has fewer than 2 networks, skipping.", job.name)
        return
    LOGGER.info(
        "Found %d networks -> %d unique pairs for %s",
        len(network_names),
        len(pair_jobs),
        job.name,
    )

    family_output_dir = output_root / job.suite / job.category / job.name
    pair_records: List[Dict[str, Any]] = []

    for source, target in pair_jobs:
        LOGGER.info("Aligning %s vs %s", source, target)
        pair_dir = family_output_dir / _pair_label(source, target)
        try:
            align_result = fugw_align(
                family_data.graphs[source],
                family_data.graphs[target],
                feature_config=feature_config,
                align_config=align_config,
                output_dir=str(pair_dir),
                sim_mat=None,
            )
        except Exception as exc:  # pragma: no cover - resilience
            LOGGER.exception(
                "Alignment failed for %s/%s/%s %s vs %s: %s",
                job.suite,
                job.category,
                job.name,
                source,
                target,
                exc,
            )
            continue

        loss_terms = align_result.get("loss_terms")
        loss_scalar = _extract_total_loss(loss_terms, align_result.get("loss"))

        pair_records.append(
            _build_pair_record(
                job=job,
                source=source,
                target=target,
                output_path=align_result["output_path"],
                loss_value=loss_scalar,
                loss_terms=loss_terms,
                runtime_seconds=float(align_result.get("runtime_seconds", float("nan"))),
                source_nodes=align_result.get("source_nodes", []),
                target_nodes=align_result.get("target_nodes", []),
            )
        )

    if not pair_records:
        LOGGER.warning(
            "Family %s produced no successful alignments, skipping summary.",
            job.name,
        )
        return
    summary_path = _write_family_results(family_output_dir, job, pair_records)
    LOGGER.info("Wrote summary to %s", summary_path)


def _parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    """

    parser = argparse.ArgumentParser(
        description="批量对齐 NAPAbenchVer2 benchmarkDataset 中的所有 family。",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="benchmarkDataset 根目录（默认：官方提供路径）。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="输出根目录（默认：/home/.../outputs）。",
    )
    parser.add_argument(
        "--suite",
        action="append",
        dest="suites",
        help="仅处理指定 suite，可重复使用该参数。",
    )
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        help="仅处理指定 category，可重复使用该参数。",
    )
    parser.add_argument(
        "--family",
        action="append",
        dest="families",
        help="仅处理指定 family，可重复使用该参数。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="可选：限制处理的 family 数量（shape=()）。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="FUGW 计算所用 device（传递给 AlignmentConfig.device）。",
    )
    return parser.parse_args()


def main() -> None:
    """
    CLI 入口。
    """

    args = _parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    jobs = _enumerate_family_jobs(dataset_root)
    jobs = _filter_jobs(jobs, args.suites, args.categories, args.families)
    if args.limit is not None:
        jobs = jobs[: max(args.limit, 0)]
    if not jobs:
        LOGGER.warning("No family matched the provided filters.")
        return

    feature_config = FeatureConfig()
    align_config = AlignmentConfig(device=args.device)

    for job in jobs:
        _align_family(
            job=job,
            output_root=output_root,
            feature_config=feature_config,
            align_config=align_config,
        )


if __name__ == "__main__":
    main()
