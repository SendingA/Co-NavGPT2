from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FireCluster:
    center: np.ndarray
    radius: float
    intensity: float


class FireSimulatorWrapper:
    """Compatibility fire simulator used when physical simulator is unavailable."""

    def __init__(
        self,
        sim: Any = None,
        num_fire_sources: int = 2,
        observations: Optional[List[Dict[str, np.ndarray]]] = None,
        use_physical: bool = False,
    ) -> None:
        _ = observations
        self.sim = sim
        self.use_physical = use_physical
        self.fire_clusters: List[FireCluster] = []

        rng = np.random.default_rng(1234)
        for _ in range(max(1, num_fire_sources)):
            center = np.array(
                [rng.uniform(-3.0, 3.0), rng.uniform(0.1, 0.8), rng.uniform(-3.0, 3.0)],
                dtype=np.float32,
            )
            self.fire_clusters.append(FireCluster(center=center, radius=1.5, intensity=0.8))

    def update(self, dt: float = 0.1) -> None:
        for c in self.fire_clusters:
            c.radius = min(4.0, c.radius + 0.03 * dt)
            c.intensity = max(0.1, c.intensity - 0.001 * dt)

    def get_fire_intensity_at_position(self, world_pos: np.ndarray) -> float:
        pos = np.asarray(world_pos, dtype=np.float32)
        max_intensity = 0.0
        for c in self.fire_clusters:
            d = float(np.linalg.norm(pos - c.center))
            if d <= c.radius:
                val = c.intensity * (1.0 - d / max(c.radius, 1e-6))
                max_intensity = max(max_intensity, float(val))
        return max_intensity

    def is_position_in_severe_fire(self, world_pos: np.ndarray, intensity_threshold: float = 0.7) -> bool:
        return self.get_fire_intensity_at_position(world_pos) >= intensity_threshold

    def get_smoke_density_at_position(self, world_pos: np.ndarray) -> float:
        return min(1.0, self.get_fire_intensity_at_position(world_pos) * 0.5)


def create_physical_fire_simulator(
    sim: Any,
    num_fires: int = 2,
    enable_smoke: bool = True,
    enable_spreading: bool = True,
    map_resolution: float = 0.05,
    map_size: int = 500,
    random_seed: Optional[int] = None,
) -> FireSimulatorWrapper:
    _ = (enable_smoke, enable_spreading, map_resolution, map_size, random_seed)
    return FireSimulatorWrapper(sim=sim, num_fire_sources=num_fires, use_physical=False)


def apply_fire_effects_to_observations(
    observations: List[Dict[str, np.ndarray]],
    fire_simulator: Any,
    agent_states: List[Any],
    apply_smoke: bool = True,
    apply_visual_fire: bool = True,
    fire_depth_threshold: float = 6.0,
) -> Tuple[List[Dict[str, np.ndarray]], List[bool]]:
    _ = (fire_simulator, agent_states, apply_smoke, apply_visual_fire, fire_depth_threshold)
    return observations, [False for _ in observations]


def get_hazard_metrics(
    fire_simulator: Any,
    agent_states: List[Any],
    intensity_threshold: float = 0.1,
) -> Dict[str, float]:
    if fire_simulator is None or len(agent_states) == 0:
        return {"avg_hazard": 0.0, "max_hazard": 0.0, "contact_ratio": 0.0}

    vals = []
    for st in agent_states:
        pos = np.asarray(st.position, dtype=np.float32)
        vals.append(float(fire_simulator.get_fire_intensity_at_position(pos)))

    arr = np.asarray(vals, dtype=np.float32)
    return {
        "avg_hazard": float(np.mean(arr)),
        "max_hazard": float(np.max(arr)),
        "contact_ratio": float(np.mean(arr > intensity_threshold)),
    }


def save_fire_simulation_video(
    fire_simulator: Any,
    output_dir: str,
    episode_id: int,
    fps: int = 10,
) -> Optional[str]:
    _ = (fire_simulator, output_dir, episode_id, fps)
    return None
