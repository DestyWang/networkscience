# Networkscience 工具集说明

## 项目概览

本仓库用于探索冷冻电镜相关的网络对齐问题，当前重点是基于 **NAPAbench 2** 数据集准备好 Fused-Unbalanced-Gromov-Wasserstein（FUGW）距离所需的输入：网络结构、功能注释和跨网络相似度。`process.py` 提供了读取、结构梳理与基础可视化的工具函数，方便快速检查数据质量与统计特征。

## NAPAbench 2 数据结构

- 路径：`/home/bcl/wanghongyu/other/networkscience/data/NAPAbench/NAPAbenchVer2/benchmarkDataset`
- 层级：
  1. **任务维度**（`2way`、`5way`、`8way`）：分别对应 2/5/8 个网络需要被同时对齐的情形，节点规模与生成树结构见原始 NAPAbench 说明。
  2. **生成模型类别**（`DMR`、`DMC`、`CG`、`STICKY`）：描述使用的网络生长模型。
  3. **familyK**：每个类别下有 10 个独立样本（`family1` ~ `family10`），每个 family 含一组网络及其配套文件。
  4. **数据文件**：`A.net`, `A.fo`, `A-B.sim`, `log_file.txt` 等。

此结构让我们可以：先选任务规模，再挑模型，然后在 family 级别读取具体网络。对齐实验时通常固定某一 suite/category，遍历其 family 以获取统计显著性。

## 文件类型速查

| 文件后缀 | 含义 | 使用方式 |
| --- | --- | --- |
| `.net` | 每行两个节点 ID，表示无向边。节点命名以所属网络的首字母开头（如 `a2040`）。 | 读成 `networkx.Graph` 以便执行拓扑分析与对齐。 |
| `.fo` | 每行“节点 ID + FO 标签”，一个节点可出现多行表示多标签。 | 读成 `Dict[str, List[str]]` 用于 FUGW 中的 feature/cost term。 |
| `.sim` | 每行“源节点 目标节点 相似度”，通常为浮点得分。文件名 `X-Y.sim` 表示网络 X 和 Y 的跨网络相似度。 | 读成 `pandas.DataFrame`（列：`source_node`, `target_node`, `score`）供 FUGW 的边/点耦合初始化或先验。 |
| `log_file.txt` | 生成该 family 时的日志（参数与随机种子）。 | 仅在需要复现实验或理解生成细节时参考。 |

## `process.py` 功能

> 所有函数都具备类型注解，并在 docstring 中说明形状信息；示例默认以 `2way/DMR/family1` 为演示对象，可按需调整。

- `discover_dataset_structure(dataset_root)`：遍历目录层级，返回套嵌字典结构。
- `summarize_structure(structure)`：把目录结构格式化为可读文本，方便快速了解数据覆盖面。
- `read_network_file/.fo/.sim`：分别将 `.net` -> `nx.Graph`，`.fo` -> `Dict[str, List[str]]`，`.sim` -> `pd.DataFrame`。
- `load_family_data(family_dir)`：一次性加载单个 family 下所有网络、注释和相似度文件，返回 `FamilyData` 数据类。
- 可视化工具：
  - `plot_degree_distribution`：节点度分布直方图。
  - `plot_annotation_histogram`：FO 术语频次条形图。
  - `plot_similarity_heatmap`：截取得分最高的若干跨网络配对，绘制热力图。
  - `visualize_family`：打包上述图形，返回 `Dict[str, matplotlib.figure.Figure]`，调用者可自行 `fig.show()` 或 `fig.savefig(...)`（如需保存，请放在已有目录下，避免新建目录违背工作区约束）。
- `main()`：打印整个数据集的结构摘要，并对 `2way/DMR/family1` 演示加载与可视化。

### 运行方式

```bash
cd /home/bcl/wanghongyu/other/networkscience
python process.py
```

若只希望导入函数，可在 notebook 或脚本中 `import process`，然后调用上述接口并替换为目标 family 的路径。

## FUGW 对齐流水线（src/align.py 等）

- 核心模块
  - `src/align.py`：端到端对齐（特征计算→几何→FUGW→评估）；`align_family()` 针对单个 family 直接运行；`demo_family1_dmr()` 演示 `2way/DMR/family1`。
  - `src/features.py`：节点特征构造，包含度/邻居度/聚类系数/紧密度/近似 betweenness/平均最短路 + FO 多热编码（前 `fo_top_k` 高频标签）。
  - `src/geometry.py`：最短路距离矩阵，优先用 SciPy 的 `csgraph.shortest_path`，缺失时回退 networkx。
  - `src/evaluate.py`：NC、Precision@k、Recall@k、EC、S3 及鲁棒性曲线工具。
- 特征权重
  - `standardize_pair` 在源/靶联合上做 z-score；可传入同长度 `feature_weights` 以缩放每一维的 L2 cost（用于对齐数值尺度）。
- 几何项
  - 使用最短路距离矩阵作为 GW 部分的 cost；不连通的点对被填充为一个大数（默认 1e6）。
- 真值与评估
  - 若未提供真值，默认按“去掉字母前缀后的数字后缀相同”推断（NAPAbench 的常见命名约定）；可传入 `ground_truth: Dict[str,str]` 覆盖。
  - 指标：Node Correctness、Recall@k、Precision@k、EC、S3；鲁棒性曲线需提供自定义 `align_fn`（可基于 `align_family` 包装）与度量函数。

### 运行示例

```bash
cd /home/bcl/wanghongyu/other/networkscience
python - <<'PY'
from pathlib import Path
from src.align import align_family

res = align_family(
    family_dir=Path("data/NAPAbench/NAPAbenchVer2/benchmarkDataset/2way/DMR/family1"),
    eval_top_k=5,
)
print("metrics:", res["metrics"])
PY
```

在 notebook 中可直接 `from src.align import align_family, FeatureConfig, AlignmentConfig`，按需调整 `fo_top_k`、最短路采样大小或 FUGW 的 `alpha/rho/eps`、`solver_params` 等。

## 后续思考与改进计划

1. **FUGW 批量化**：遍历多组 family 自动汇总指标，并缓存最短路矩阵避免重复计算。
2. **真值来源**：当前默认用节点后缀推断；若找到官方映射文件，可直接接入 `ground_truth` 并下放到评估工具。
3. **日志解析**：必要时解析 `log_file.txt`，追踪生成参数，对实验记录更友好。

以上内容会随着开发进展持续更新，便于快速上手和协作。

