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
- `IsoRankN/eval_all.py` – 对 `IsoRankN/napa_outputs` 目录下 alpha=0.6/1.0、
  score/cluster 四种耦合矩阵分别批量计算 10 项对齐指标，并读取运行日志中的
  runtime（若日志缺失则置为 NaN），结果写入
  `IsoRankN/eval_all_{1..4}.csv`。
- `src/utils.py` – shared helpers (tensor-to-numpy conversions)；`src/align.py`
  默认在 `results.json` 写入完整的 FUGW `loss_terms` 字典，含
  `wasserstein`/`gromov_wasserstein`/`marginal_constraint_dim1`/
  `marginal_constraint_dim2`/`regularization`/`total` 六个序列，便于追踪收敛。
- `src/vis.py` – matplotlib helpers (degree histogram, matrix heatmap).
- `src/utils.py` – shared helpers (tensor-to-numpy conversions).

Feel free to adapt the configs or plug new metrics as needed; each module is
kept intentionally concise to encourage experimentation inside notebooks.

## 结构指标：Rich-clubness 与 LCP-corr（`utils.py`）

根目录下的 `utils.py` 提供两个常用的网络结构指标（与 `src/` 解耦，便于在 notebook 中直接调用）。

### Rich-clubness（Cannistraci & Muscoloni 口径，含 p-value）

- **函数**：`get_rich_clubness(G, n_random=200, nswap_factor=10.0, seed=0) -> (rc_value, p_value)`
- **输入**：`networkx.Graph`（默认按**无向简单图**处理；有向图会自动转为无向；自环会移除）
- **输出**：
  - **`rc_value`**：标量 rich-clubness 分数。通常 **`rc_value > 1`** 表示比“保持度序列的随机对照网络”更强的 rich-club 结构
  - **`p_value`**：单侧置换检验 p-value。经验判读：**`p_value < 0.05`** 表示 rich-clubness 显著高于随机对照

使用示例：

```python
import networkx as nx
from utils import get_rich_clubness

G = nx.karate_club_graph()
rc, p = get_rich_clubness(G, n_random=200, nswap_factor=10.0, seed=0)
print("rich-clubness:", rc, "p-value:", p)
```

### Local Community Paradigm correlation（LCP-corr）

- **函数**：`get_LCP_corr(G) -> r`
- **定义**：对每条真实边 \((u, v)\)，计算
  - **CN**：共同邻居数 \(|N(u) \cap N(v)|\)
  - **LCL**：共同邻居诱导子图中的边数
  然后在所有 **CN > 0** 的边样本上计算 CN 与 LCL 的 Pearson 相关系数
- **输出**：**`r`**（范围约为 \([-1, 1]\)）。`r` 越大且为正，表示网络越符合 “Local Community Paradigm”（共同邻居越多时，其内部也越倾向形成更紧密的连接）
- **异常情况**：若有效样本数不足（例如几乎没有三角/局部团结构），返回 `NaN`

使用示例：

```python
import networkx as nx
from utils import get_LCP_corr

G = nx.karate_club_graph()
r = get_LCP_corr(G)
print("LCP-corr:", r)
```

## 批量对齐（`src/align_all.py`）

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u -m src.align_all \
  --suite 5way \
  --category CG \
  --device cuda
```
```bash
CUDA_VISIBLE_DEVICES=2 nohup python -u -m src.align_all > all_families.log 2>&1 &
 3873633
```

- 默认数据目录：`data/NAPAbench/NAPAbenchVer2/benchmarkDataset`；
  默认输出目录：`outputs/`。
- 5way/8way family 下会为每个组合网络建立 `A_vs_B` 子目录；`fugw_align` 会在其内
  生成时间戳子文件夹并保存 coupling/特征/元数据。2way 也沿用相同结构。
- 脚本会在 family 根目录生成 `results.json`，列出所有成功 pair 的
  `loss`（shape=()，即 `loss_terms['total']` 的最终元素）、`loss_terms` 字典、
  `runtime_seconds`（shape=()）、节点规模和输出路径，方便横向比对。
- 可使用 `--suite/--category/--family` 多次传入过滤目标，也可通过 `--limit`
  控制处理的 family 数量；`--device` 直接透传给 `AlignmentConfig.device`。

## IsoRankN 批量评估（`IsoRankN/eval_all.py`）

```bash
python IsoRankN/eval_all.py
# 或者:
python -m IsoRankN.eval_all
```

- 默认扫描 `IsoRankN/napa_outputs`，对每个 family/pair/alpha 的
  score/cluster 结果生成 12 项指标（10 个对齐指标 + runtime + total loss）。
- runtime 通过 `IsoRankN/isorankn_all.log` 中的 `Finished in ... s` 自动解析；
  IsoRankN 本身未输出 loss，因此 `total_loss` 会置为 `NaN`。
- 四份汇总表分别写入：
  - `IsoRankN/eval_all_1.csv`：alpha=0.6 的 score coupling
  - `IsoRankN/eval_all_2.csv`：alpha=0.6 的 cluster coupling
  - `IsoRankN/eval_all_3.csv`：alpha=1.0 的 score coupling
  - `IsoRankN/eval_all_4.csv`：alpha=1.0 的 cluster coupling