"""Feature extraction for the GNN frontier-assigner (Plan A: imitation of GPT).

This module turns the runtime structures produced by ``Global_Map_Proc`` and
the per-robot ``pose_pred`` list into fixed-shape tensors suitable for a
small cross-attention model.

Design notes
------------
* No ``torch_geometric`` dependency: the bipartite robot<->frontier graph is
  small (typically R<=4, F<=10) so we just emit dense tensors.
* All features are normalised to map units in [-1, 1] (or [0, 1] for sizes)
  so that a model trained on one map size still works on another.
* The function is pure and side-effect free; it can be used both at
  inference time (``agents/gnn_assigner.py``) and at trace-collection time
  (``utils/gpt_trace.py``).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

ROBOT_FEAT_DIM = 5      # [x_norm, y_norm, cos_yaw, sin_yaw, num_frontiers_norm]
FRONTIER_FEAT_DIM = 5   # [x_norm, y_norm, score_norm, area_norm, idx_norm]
EDGE_FEAT_DIM = 4       # [dx_norm, dy_norm, dist_norm, cos_alignment]


def _normalise_xy(xy: Sequence[float], map_size: int) -> np.ndarray:
    """Map grid coordinates in [0, map_size) to [-1, 1]."""
    arr = np.asarray(xy, dtype=np.float32)
    return arr / float(map_size) * 2.0 - 1.0


def build_features(
    target_point_list: Sequence[Sequence[int]],
    target_score: Optional[Sequence[float]],
    target_edge_map: Optional[np.ndarray],
    pose_pred: Sequence[Sequence[float]],
    map_size: int,
) -> Optional[Dict[str, torch.Tensor]]:
    """Build dense feature tensors for the assigner.

    Parameters
    ----------
    target_point_list : list of [grid_x, grid_y]
        Frontier centroids returned by ``Frontier_Det``.
    target_score : list of float, optional
        Per-frontier exploration score (FMM distance). May be ``None``.
    target_edge_map : ndarray (H, W), optional
        Labelled frontier mask. Used to derive a normalised area feature.
    pose_pred : list of [y_grid, x_grid, yaw_rad]
        One entry per robot. Note the (y, x) ordering matches main.py.
    map_size : int
        Side length of the global map in grid cells.

    Returns
    -------
    dict with keys:
        ``robot_feat``    : (R, ROBOT_FEAT_DIM)
        ``frontier_feat`` : (F, FRONTIER_FEAT_DIM)
        ``edge_feat``     : (R, F, EDGE_FEAT_DIM)
    or ``None`` if no frontiers are available.
    """
    num_frontiers = len(target_point_list)
    num_robots = len(pose_pred)
    if num_frontiers == 0 or num_robots == 0:
        return None

    # ------------------------------------------------------------------
    # Frontier features: position, score, area, index
    # ------------------------------------------------------------------
    frontier_xy = np.array(target_point_list, dtype=np.float32)  # (F, 2) order: [x, y]
    frontier_xy_norm = frontier_xy / float(map_size) * 2.0 - 1.0

    if target_score is not None and len(target_score) == num_frontiers:
        score_arr = np.asarray(target_score, dtype=np.float32)
        score_max = float(score_arr.max()) if score_arr.size else 1.0
        score_norm = score_arr / score_max if score_max > 0 else score_arr
    else:
        score_norm = np.zeros(num_frontiers, dtype=np.float32)

    if target_edge_map is not None:
        total_cells = float(target_edge_map.size)
        area_norm = np.array(
            [float((target_edge_map == i + 1).sum()) / total_cells for i in range(num_frontiers)],
            dtype=np.float32,
        )
    else:
        area_norm = np.zeros(num_frontiers, dtype=np.float32)

    idx_norm = np.arange(num_frontiers, dtype=np.float32) / max(num_frontiers - 1, 1)

    frontier_feat = np.concatenate(
        [
            frontier_xy_norm,
            score_norm[:, None],
            area_norm[:, None],
            idx_norm[:, None],
        ],
        axis=1,
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # Robot features: pose (note the y, x swap to match main.py convention)
    # ------------------------------------------------------------------
    pose_arr = np.asarray(pose_pred, dtype=np.float32)  # (R, 3) [y, x, yaw]
    robot_xy = np.stack([pose_arr[:, 1], pose_arr[:, 0]], axis=1)  # -> [x, y]
    robot_xy_norm = robot_xy / float(map_size) * 2.0 - 1.0
    cos_yaw = np.cos(pose_arr[:, 2])
    sin_yaw = np.sin(pose_arr[:, 2])
    nf_feat = np.full((num_robots, 1), num_frontiers / 10.0, dtype=np.float32)
    robot_feat = np.concatenate(
        [robot_xy_norm, cos_yaw[:, None], sin_yaw[:, None], nf_feat],
        axis=1,
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # Edge features: relative geometry between each robot/frontier pair
    # ------------------------------------------------------------------
    # delta = frontier_xy - robot_xy   shape (R, F, 2)
    delta = frontier_xy[None, :, :] - robot_xy[:, None, :]
    delta_norm = delta / float(map_size)
    dist = np.linalg.norm(delta, axis=-1, keepdims=True) / float(map_size)
    # alignment between robot heading and direction to frontier
    heading = np.stack([cos_yaw, sin_yaw], axis=1)  # (R, 2)
    dir_to = delta / (np.linalg.norm(delta, axis=-1, keepdims=True) + 1e-6)
    cos_align = (heading[:, None, :] * dir_to).sum(axis=-1, keepdims=True)
    edge_feat = np.concatenate([delta_norm, dist, cos_align], axis=-1).astype(np.float32)

    return {
        "robot_feat": torch.from_numpy(robot_feat),
        "frontier_feat": torch.from_numpy(frontier_feat),
        "edge_feat": torch.from_numpy(edge_feat),
    }


def features_to_jsonable(features: Dict[str, torch.Tensor]) -> Dict[str, List]:
    """Convert tensor dict to plain lists for JSONL serialisation."""
    return {k: v.detach().cpu().numpy().tolist() for k, v in features.items()}


def features_from_jsonable(payload: Dict[str, List]) -> Dict[str, torch.Tensor]:
    """Inverse of :func:`features_to_jsonable`."""
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in payload.items()}
