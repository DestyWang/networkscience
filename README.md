# Network Science Alignment Toolkit

This repository contains a lightweight pipeline for aligning protein interaction
networks from the NAPAbench v2 benchmark with **FUGW** (Fused Gromov–Wasserstein).
It focuses on three steps:

1. **Pre-processing** (`src/process.py`)
   - Load `.net`, `.fo`, `.sim` assets into the `FamilyData` container.
   - Compute graph statistics such as distance matrices, clustering/centrality
     profiles, and small-world / scale-free diagnostics.
2. **Feature extraction & alignment**
   - `src/features.get_node_features` builds a 5-D structural descriptor per node.
   - `src.align.fugw_align` feeds normalized features + geodesic distances to
     the `fugw` library and returns the optimal transport coupling.
3. **Evaluation & visualization**
   - `src.evaluate` implements similarity-based alignment scores, FO-consistency,
     top-k FO recall, and S³ (hard/soft) metrics.
   - `src.vis` exposes degree-distribution and generic heatmap helpers for
     inspecting networks, distance matrices, similarity tables, or couplings.

## Getting started

```bash
pip install -r requirements.txt  # fugw, networkx, pandas, torch, matplotlib, etc.
python -m pip install fugw
```

Explore the provided notebooks (`process.ipynb`, `test.ipynb`) or call the
functions directly. Example alignment snippet:

```python
from pathlib import Path
from src.process import load_family_data
from src.align import fugw_align

family = load_family_data(Path("data/NAPAbench/.../family1"))
result = fugw_align(family.graphs["A"], family.graphs["B"])
pi = result["pi"].numpy()
```

## Repository layout

- `src/process.py` – dataset I/O + graph statistics utilities.
- `src/features.py` – deterministic structural feature builder.
- `src/align.py` – thin FUGW wrapper returning the coupling matrix.
- `src/evaluate.py` – alignment quality metrics。`uniform_coupling_from_clusters`
  采用 1-to-n / n-to-1 归一化策略并在全局层面再次标准化，能更真实地表达
  IsoRankN cluster 的置信度；后续可以在此基础上接入更细粒度的置信度估计。
- `src/align_all.py` – 批量遍历 NAPAbenchVer2 benchmarkDataset，针对 2way/5way/8way
  family 执行两两配准，并将每个 pair 的输出结构化写入
  `outputs/<suite>/<category>/<family>/<pair>/<timestamp>/...`，同时在 family
  级别生成整合的 `results.json` 方便后续检索。
- `src/vis.py` – matplotlib helpers (degree histogram, matrix heatmap).
- `src/utils.py` – shared helpers (tensor-to-numpy conversions).

Feel free to adapt the configs or plug new metrics as needed; each module is
kept intentionally concise to encourage experimentation inside notebooks.

## 批量对齐（`src/align_all.py`）

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u -m src.align_all \
  --suite 5way \
  --category CG \
  --device cuda
```
```bash
CUDA_VISIBLE_DEVICES=2 nohup python -u -m src.align_all > all_families.log 2>&1 &
 2864276
```

- 默认数据目录：`data/NAPAbench/NAPAbenchVer2/benchmarkDataset`；
  默认输出目录：`outputs/`。
- 5way/8way family 下会为每个组合网络建立 `A_vs_B` 子目录；`fugw_align` 会在其内
  生成时间戳子文件夹并保存 coupling/特征/元数据。2way 也沿用相同结构。
- 脚本会在 family 根目录生成 `results.json`，列出所有成功 pair 的
  `loss`（shape=()）、`runtime_seconds`（shape=()）、节点规模和输出路径，方便横向比对。
- 可使用 `--suite/--category/--family` 多次传入过滤目标，也可通过 `--limit`
  控制处理的 family 数量；`--device` 直接透传给 `AlignmentConfig.device`。