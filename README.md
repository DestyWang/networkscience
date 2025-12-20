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
- `src/evaluate.py` – alignment quality metrics.
- `src/vis.py` – matplotlib helpers (degree histogram, matrix heatmap).
- `src/utils.py` – shared helpers (tensor-to-numpy conversions).

Feel free to adapt the configs or plug new metrics as needed; each module is
kept intentionally concise to encourage experimentation inside notebooks.