from __future__ import annotations

from typing import List

import networkx as nx
import numpy as np

try:
    from scipy.sparse import csr_array
    from scipy.sparse.csgraph import shortest_path
except Exception:  # pragma: no cover - SciPy 可能未安装
    shortest_path = None
    csr_array = None


def shortest_path_distance_matrix(graph: nx.Graph, node_list: List[str] | None = None, large_value: float = 1e6,
) -> np.ndarray:
    """
    生成节点对的最短路距离矩阵，用于 GW 几何项。

    Parameters
    ----------
    graph : nx.Graph
        输入无向图。
    node_list : List[str] | None
        节点顺序；None 时使用排序后的节点列表。
    large_value : float
        当两节点不连通时替代 inf 的值。

    Returns
    -------
    np.ndarray
        距离矩阵，形状 (n, n)，未连接节点距离为 large_value。
    """

    nodes = node_list or sorted(graph.nodes())
    n = len(nodes)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)

    if shortest_path is not None and csr_array is not None:
        adjacency = nx.to_scipy_sparse_array(graph, nodelist=nodes, dtype=np.float32)
        dist = shortest_path(
            csr_array(adjacency),
            directed=False,
            unweighted=True,
        )
    else:
        dist_dict = dict(nx.all_pairs_shortest_path_length(graph))
        dist = np.full((n, n), np.inf, dtype=np.float32)
        idx = {node: i for i, node in enumerate(nodes)}
        for u, nbrs in dist_dict.items():
            i = idx[u]
            for v, d in nbrs.items():
                j = idx[v]
                dist[i, j] = float(d)

    finite_max = np.nanmax(dist[np.isfinite(dist)])
    fill_value = max(finite_max * 1.05, large_value) if np.isfinite(finite_max) else large_value
    dist[np.isinf(dist)] = fill_value
    return dist.astype(np.float32)

