"""
本文件用于放置与 notebook/实验相关的轻量级通用函数（与 src/ 目录解耦）。

注意：
- 为了和网络科学文献中的定义保持一致，这里默认处理 **无向简单图**。
- 若输入是有向图，会自动转为无向图视图进行计算。
"""

from __future__ import annotations

from typing import Dict, Hashable, Optional, Tuple

import networkx as nx
import numpy as np


def _as_simple_undirected_graph_view(G: nx.Graph) -> nx.Graph:
    """
    将输入图转换为“无向简单图”的视图/拷贝：
    - 有向图：转为无向（忽略方向）
    - 多重图：转为简单图（并行边折叠为 1）
    - 自环：移除
    """
    if G.is_multigraph():
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        H.add_edges_from((u, v) for u, v in G.edges())
    elif G.is_directed():
        # as_view=True 不复制边属性，足够用于结构指标
        H = G.to_undirected(as_view=True)  # type: ignore[assignment]
    else:
        H = G

    # 去自环（视图上无法原地删除时，转为拷贝）
    if any(u == v for u, v in H.edges()):
        H2 = nx.Graph(H)
        H2.remove_edges_from(nx.selfloop_edges(H2))
        return H2
    return H


def get_LCP_corr(G: nx.Graph) -> float:
    """
    计算 Local Community Paradigm correlation (LCP-corr)。

    定义（Cannistraci 等的 LCP 体系常用口径）：
    对每一条真实边 (u, v) ∈ E，计算
    - CN(u, v): common neighbors 数量
    - LCL(u, v): local community links，即共同邻居诱导子图中的边数
    然后对所有 CN(u, v) > 0 的边样本，计算 CN 与 LCL 的 Pearson 相关系数。

    参数
    - G: networkx.Graph

    返回
    - r: float
        Pearson 相关系数，范围约为 [-1, 1]。
        若有效样本数 < 2（例如几乎没有三角结构），返回 np.nan。
    """
    H = _as_simple_undirected_graph_view(G)
    if H.number_of_edges() == 0:
        return float("nan")

    # 用邻接集合加速 CN 与 LCL 计算
    adj: Dict[Hashable, set[Hashable]] = {n: set(H.neighbors(n)) for n in H.nodes()}

    cn_list: list[int] = []
    lcl_list: list[int] = []

    for u, v in H.edges():
        cn = adj[u] & adj[v]
        cn_size = len(cn)
        if cn_size == 0:
            continue

        # LCL: 共同邻居诱导子图中的边数
        # 计数技巧：对每个 w∈CN 统计其在 CN 内的邻居数，累加后除以 2
        lcl_twice = 0
        for w in cn:
            lcl_twice += len(adj[w] & cn)
        lcl = lcl_twice // 2

        cn_list.append(cn_size)
        lcl_list.append(lcl)

    if len(cn_list) < 2:
        return float("nan")

    r = float(np.corrcoef(np.asarray(cn_list, dtype=float), np.asarray(lcl_list, dtype=float))[0, 1])
    return r


def get_rich_clubness(
    G: nx.Graph,
    *,
    n_random: int = 200,
    nswap_factor: float = 10.0,
    max_tries_factor: float = 50.0,
    seed: Optional[int] = 0,
) -> Tuple[float, float]:
    """
    计算 rich-clubness（富节点俱乐部倾向）及其置换检验 p-value（Cannistraci & Muscoloni 的常用做法：度保持随机化对照）。

    核心思路：
    - 先计算观测网络的 rich-club coefficient φ(k) 曲线（k 为度阈值）
    - 通过“保持度序列”的 double-edge-swap 生成随机对照网络集合
    - 用随机集合的 φ_rand_mean(k) 对 φ_obs(k) 做归一化：
        ρ(k) = φ_obs(k) / φ_rand_mean(k)
    - 将 ρ(k) 在可用的 k 上取平均，得到一个标量 rich-clubness 分数：
        RC = mean_k ρ(k)
    - p-value：将每个随机图的 RC_i = mean_k φ_i(k)/φ_rand_mean(k) 与 RC_obs 比较，做单侧检验

    参数
    - G: networkx.Graph
    - n_random: int
        随机对照网络数量（置换次数），越大 p-value 越稳定。
    - nswap_factor: float
        double_edge_swap 的 nswap = int(nswap_factor * |E|)，控制随机化强度。
    - max_tries_factor: float
        max_tries = int(max_tries_factor * nswap)，避免 swap 失败时无限尝试。
    - seed: Optional[int]
        随机种子，便于复现。设为 None 则不固定。

    返回
    - rc_value: float
        rich-clubness 标量分数（通常 rc_value > 1 表示比随机对照更强的 rich-club 结构）。
    - p_value: float
        单侧置换检验 p-value。经验判读：p < 0.05 表示 rich-clubness 显著高于随机对照。
        若网络规模/结构导致无法形成可用 k（例如边太少），返回 (np.nan, np.nan)。
    """
    H = _as_simple_undirected_graph_view(G)
    m = H.number_of_edges()
    if m < 3:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)

    # 观测 φ(k)
    phi_obs: Dict[int, float] = nx.rich_club_coefficient(H, normalized=False, Q=100)  # type: ignore[assignment]

    # 生成随机对照图并计算 φ_i(k)
    nswap = max(1, int(nswap_factor * m))
    max_tries = max(10, int(max_tries_factor * nswap))

    phi_rands: list[Dict[int, float]] = []
    for i in range(n_random):
        Hr = nx.Graph(H)
        try:
            # networkx 的 seed 支持 int / RandomState；这里用 int 以保证可复现
            nx.double_edge_swap(Hr, nswap=nswap, max_tries=max_tries, seed=int(rng.integers(0, 2**31 - 1)))
        except Exception:
            # 对极稀疏/特殊结构图，swap 可能失败；失败则跳过该样本
            continue
        phi_i: Dict[int, float] = nx.rich_club_coefficient(Hr, normalized=False, Q=100)  # type: ignore[assignment]
        phi_rands.append(phi_i)

    if len(phi_rands) < max(20, n_random // 5):
        # 随机化失败过多时，结果不可靠
        return float("nan"), float("nan")

    # 对齐 k 并计算 φ_rand_mean(k)
    ks = sorted(set(phi_obs.keys()).intersection(*(set(d.keys()) for d in phi_rands)))
    if not ks:
        return float("nan"), float("nan")

    phi_mean: Dict[int, float] = {}
    for k in ks:
        vals = [d[k] for d in phi_rands]
        phi_mean[k] = float(np.mean(vals))

    # 选择可用 k（分母>0）
    ks_use = [k for k in ks if phi_mean.get(k, 0.0) > 0.0 and phi_obs.get(k, 0.0) >= 0.0]
    if not ks_use:
        return float("nan"), float("nan")

    rho_obs = [phi_obs[k] / phi_mean[k] for k in ks_use]
    rc_obs = float(np.mean(rho_obs))

    rc_null = []
    for d in phi_rands:
        rc_i = float(np.mean([d[k] / phi_mean[k] for k in ks_use]))
        rc_null.append(rc_i)

    # 单侧：观测值是否“显著更大”
    rc_null_arr = np.asarray(rc_null, dtype=float)
    p_value = float((np.sum(rc_null_arr >= rc_obs) + 1.0) / (len(rc_null_arr) + 1.0))
    return rc_obs, p_value
