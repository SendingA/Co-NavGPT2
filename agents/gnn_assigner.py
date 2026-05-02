"""GNN-based frontier assigner (Plan A: Behavior Cloning of GPT-4o).

The model is intentionally tiny (a few cross-attention layers operating on
the dense robot/frontier feature tensors produced by
:mod:`utils.graph_builder`). It learns to imitate GPT's frontier choice
from offline traces collected via :mod:`utils.gpt_trace`.

Inference output mimics ``utils.chat_utils.chat_with_gpt4v`` exactly so the
caller in ``main.py`` can use a uniform interface.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.graph_builder import (
    EDGE_FEAT_DIM,
    FRONTIER_FEAT_DIM,
    ROBOT_FEAT_DIM,
    build_features,
)


class CrossAttentionAssigner(nn.Module):
    """Lightweight robot<->frontier cross-attention scorer.

    For each robot ``r`` and frontier ``f`` we produce a logit
    ``s[r, f]``. At inference we argmax over ``f`` for each robot
    (independently) to mirror what the GPT prompt asks the model to do.
    """

    def __init__(self, hidden: int = 64, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        self.hidden = hidden
        self.robot_in = nn.Linear(ROBOT_FEAT_DIM, hidden)
        self.frontier_in = nn.Linear(FRONTIER_FEAT_DIM, hidden)
        self.edge_in = nn.Linear(EDGE_FEAT_DIM, hidden)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "r2f": nn.MultiheadAttention(hidden, num_heads, batch_first=True),
                        "f2r": nn.MultiheadAttention(hidden, num_heads, batch_first=True),
                        "ln_r": nn.LayerNorm(hidden),
                        "ln_f": nn.LayerNorm(hidden),
                        "mlp_r": nn.Sequential(
                            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)
                        ),
                        "mlp_f": nn.Sequential(
                            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)
                        ),
                    }
                )
            )

        self.score_head = nn.Sequential(
            nn.Linear(hidden * 2 + hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        robot_feat: torch.Tensor,      # (R, Dr) or (B, R, Dr)
        frontier_feat: torch.Tensor,   # (F, Df) or (B, F, Df)
        edge_feat: torch.Tensor,       # (R, F, De) or (B, R, F, De)
    ) -> torch.Tensor:
        """Return logits of shape ``(B, R, F)``."""
        if robot_feat.dim() == 2:
            robot_feat = robot_feat.unsqueeze(0)
            frontier_feat = frontier_feat.unsqueeze(0)
            edge_feat = edge_feat.unsqueeze(0)

        r = self.robot_in(robot_feat)            # (B, R, H)
        f = self.frontier_in(frontier_feat)      # (B, F, H)
        e = self.edge_in(edge_feat)              # (B, R, F, H)

        # Inject edge bias by adding mean-pooled edge embedding to each side.
        r = r + e.mean(dim=2)
        f = f + e.mean(dim=1)

        for layer in self.layers:
            r_attn, _ = layer["r2f"](r, f, f)
            r = layer["ln_r"](r + r_attn)
            r = r + layer["mlp_r"](r)

            f_attn, _ = layer["f2r"](f, r, r)
            f = layer["ln_f"](f + f_attn)
            f = f + layer["mlp_f"](f)

        # Pairwise score: concat robot, frontier, edge -> MLP
        B, R, H = r.shape
        Fn = f.shape[1]
        r_exp = r.unsqueeze(2).expand(B, R, Fn, H)
        f_exp = f.unsqueeze(1).expand(B, R, Fn, H)
        pair = torch.cat([r_exp, f_exp, e], dim=-1)
        logits = self.score_head(pair).squeeze(-1)  # (B, R, F)
        return logits


class GNNAssigner:
    """High-level wrapper used at runtime by ``main.py``.

    If no checkpoint is supplied the assigner falls back to a deterministic
    nearest-frontier heuristic so the ``--nav_mode gnn`` code path is
    runnable end-to-end before any training data exists.
    """

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        device: Optional[str] = None,
        hidden: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: Optional[CrossAttentionAssigner] = None
        if ckpt_path and os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device)
            cfg = ckpt.get("config", {"hidden": hidden, "num_layers": num_layers, "num_heads": num_heads})
            self.model = CrossAttentionAssigner(**cfg).to(self.device)
            self.model.load_state_dict(ckpt["model"])
            self.model.eval()
            print(f"[GNNAssigner] loaded checkpoint: {ckpt_path}")
        else:
            print(
                "[GNNAssigner] no checkpoint found at "
                f"{ckpt_path!r}; falling back to nearest-frontier heuristic."
            )

    @torch.no_grad()
    def assign(
        self,
        target_point_list: Sequence[Sequence[int]],
        target_score: Optional[Sequence[float]],
        target_edge_map: Optional[np.ndarray],
        pose_pred: Sequence[Sequence[float]],
        map_size: int,
        num_agents: int,
    ) -> Dict[str, str]:
        """Return ``{"robot_i": "frontier_j", ..., "reason": "gnn"}``.

        If features cannot be built (no frontiers) returns the same default
        as ``chat_with_gpt4v`` does on failure.
        """
        feats = build_features(target_point_list, target_score, target_edge_map, pose_pred, map_size)
        if feats is None:
            return {f"robot_{i}": "frontier_0" for i in range(num_agents)} | {"reason": "no_frontier"}

        if self.model is None:
            return self._nearest_fallback(target_point_list, pose_pred, num_agents)

        robot_feat = feats["robot_feat"].to(self.device)
        frontier_feat = feats["frontier_feat"].to(self.device)
        edge_feat = feats["edge_feat"].to(self.device)
        logits = self.model(robot_feat, frontier_feat, edge_feat)[0]  # (R, F)
        choice = logits.argmax(dim=-1).cpu().tolist()
        result: Dict[str, str] = {f"robot_{i}": f"frontier_{choice[i]}" for i in range(min(num_agents, len(choice)))}
        # pad if model produced fewer rows than agents (shouldn't happen)
        for i in range(len(choice), num_agents):
            result[f"robot_{i}"] = "frontier_0"
        result["reason"] = "gnn"
        return result

    @staticmethod
    def _nearest_fallback(
        target_point_list: Sequence[Sequence[int]],
        pose_pred: Sequence[Sequence[float]],
        num_agents: int,
    ) -> Dict[str, str]:
        result: Dict[str, str] = {}
        frontier_xy = np.asarray(target_point_list, dtype=np.float32)
        for i in range(num_agents):
            # pose_pred[i] = [y, x, yaw]; frontier list uses [x, y]
            robot_xy = np.array([pose_pred[i][1], pose_pred[i][0]], dtype=np.float32)
            dists = np.linalg.norm(frontier_xy - robot_xy, axis=1)
            result[f"robot_{i}"] = f"frontier_{int(np.argmin(dists))}"
        result["reason"] = "nearest_fallback"
        return result
