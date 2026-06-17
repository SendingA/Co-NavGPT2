"""Voxel discretisation of a Habitat scene's bounding volume.

Stage-3 helper that turns a plan's world AABB + the inventory's per-object
AABBs into the initial fields the propagation engine integrates over time.

We deliberately keep the world tiny: HM3D Y is up, so the array layout is
``(Nx, Ny, Nz)`` and ``y`` indexes the vertical axis. Voxel size is
typically 0.15 m (the project default agreed in design discussion).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class VoxelWorld:
    """A 3D voxel grid plus its world->grid affine.

    Attributes:
      origin: ``(3,)`` world coordinates of voxel ``(0, 0, 0)`` corner.
      voxel:  scalar voxel edge length in metres.
      shape:  ``(Nx, Ny, Nz)``.
      fuel:   ``(Nx, Ny, Nz)`` float32, [0, 1].
      temp:   ``(Nx, Ny, Nz)`` float32, deg C.
      flame:  ``(Nx, Ny, Nz)`` float32, [0, 1].
      smoke:  ``(Nx, Ny, Nz)`` float32, [0, 1].
      ambient_c: scalar background temperature (deg C).
      walls:    ``(Nx, Ny, Nz)`` bool, structural barriers (impermeable
                to heat/smoke). When None, treated as all-False.
      floors:   ``(Nx, Ny, Nz)`` bool, walkable floor voxels.
      ceilings: ``(Nx, Ny, Nz)`` bool, ceiling voxels (cap on plumes).
    """

    origin: np.ndarray
    voxel: float
    shape: Tuple[int, int, int]
    fuel: np.ndarray
    temp: np.ndarray
    flame: np.ndarray
    smoke: np.ndarray
    ambient_c: float = 25.0
    object_id_field: Optional[np.ndarray] = None  # (Nx,Ny,Nz) int32, -1 = none
    walls: Optional[np.ndarray] = None
    floors: Optional[np.ndarray] = None
    ceilings: Optional[np.ndarray] = None

    @classmethod
    def from_aabb(
        cls,
        world_aabb: List[float],
        voxel: float,
        ambient_c: float = 25.0,
    ) -> "VoxelWorld":
        amin = np.asarray(world_aabb[:3], dtype=np.float64)
        amax = np.asarray(world_aabb[3:], dtype=np.float64)
        extent = np.maximum(amax - amin, 1e-3)
        shape = tuple(int(np.ceil(e / voxel)) for e in extent)
        Nx, Ny, Nz = shape
        return cls(
            origin=amin,
            voxel=float(voxel),
            shape=shape,
            fuel=np.zeros((Nx, Ny, Nz), dtype=np.float32),
            temp=np.full((Nx, Ny, Nz), float(ambient_c), dtype=np.float32),
            flame=np.zeros((Nx, Ny, Nz), dtype=np.float32),
            smoke=np.zeros((Nx, Ny, Nz), dtype=np.float32),
            ambient_c=float(ambient_c),
            object_id_field=None,
        )

    # ------------------------------------------------------------------
    # Structural masks
    # ------------------------------------------------------------------
    def attach_structural_masks(
        self,
        wall_path: Optional[str] = None,
        floor_path: Optional[str] = None,
        ceiling_path: Optional[str] = None,
    ) -> None:
        """Load wall/floor/ceiling boolean masks and resize to current
        voxel grid shape via nearest-neighbour. Mismatched grid spacing
        between scan-time and run-time is tolerated.
        """
        def _load_and_fit(path: Optional[str]) -> Optional[np.ndarray]:
            if not path:
                return None
            try:
                arr = np.load(path)
            except Exception as e:
                print(f"[voxel_world] failed to load {path}: {e}")
                return None
            if arr.shape == self.shape:
                return arr.astype(bool)
            # Nearest-neighbour resample to self.shape (cheap; rarely run).
            ix = np.linspace(0, arr.shape[0] - 1, self.shape[0]).astype(int)
            iy = np.linspace(0, arr.shape[1] - 1, self.shape[1]).astype(int)
            iz = np.linspace(0, arr.shape[2] - 1, self.shape[2]).astype(int)
            return arr[ix[:, None, None], iy[None, :, None], iz[None, None, :]].astype(bool)

        self.walls = _load_and_fit(wall_path)
        self.floors = _load_and_fit(floor_path)
        self.ceilings = _load_and_fit(ceiling_path)

    # ------------------------------------------------------------------
    # World <-> grid
    # ------------------------------------------------------------------
    def world_to_grid(self, p: np.ndarray) -> np.ndarray:
        return ((p - self.origin) / self.voxel).astype(np.int32)

    def grid_to_world(self, ijk: np.ndarray) -> np.ndarray:
        return self.origin + (np.asarray(ijk, dtype=np.float32) + 0.5) * self.voxel

    # ------------------------------------------------------------------
    # Field initialisers
    # ------------------------------------------------------------------
    def stamp_object_aabbs(self, objects: List[Dict]) -> None:
        """Bake fuel + object id from a list of inventory entries.

        Accepts either:
          - schema v1 ``objects`` items (with ``object_id``), OR
          - schema v2 ``instances`` items (with ``instance_id`` and a
            ``structural`` flag we honour to skip walls/floors/etc.).

        Walls and structural items contribute nothing to the fuel field;
        their geometry is consumed via ``attach_structural_masks``.
        """
        if self.object_id_field is None:
            self.object_id_field = np.full(self.shape, -1, dtype=np.int32)
        for obj in objects:
            if bool(obj.get("structural", False)):
                continue
            f = float(obj.get("flammability", 0.0))
            if f <= 0.0:
                continue
            i0 = self.world_to_grid(np.asarray(obj["aabb_min"]))
            i1 = self.world_to_grid(np.asarray(obj["aabb_max"]))
            i0 = np.maximum(i0, 0)
            i1 = np.minimum(i1, np.asarray(self.shape) - 1)
            if np.any(i1 < i0):
                continue
            sl = (
                slice(int(i0[0]), int(i1[0]) + 1),
                slice(int(i0[1]), int(i1[1]) + 1),
                slice(int(i0[2]), int(i1[2]) + 1),
            )
            self.fuel[sl] = np.maximum(self.fuel[sl], f)
            obj_id = int(obj.get("object_id", obj.get("instance_id", -1)))
            self.object_id_field[sl] = obj_id

    def kindle_ignition(self, position: np.ndarray, radius_m: float,
                        temp_c: float, smoke_yield: float = 0.5) -> Tuple[Tuple[slice, slice, slice], np.ndarray]:
        """Inject a hot, fuel-loaded sphere at world position ``p``.

        We *also* deposit a fuel floor inside the sphere so the fire can
        sustain itself even if the goal-object AABB approximation does
        not perfectly align with the ignition centre.

        Returns the slice and falloff field so callers can keep the source
        warm for the duration of the ignition.
        """
        ijk = self.world_to_grid(np.asarray(position, dtype=np.float64))
        r_cells = max(1, int(np.ceil(radius_m / self.voxel)))

        Nx, Ny, Nz = self.shape
        i0 = np.maximum(ijk - r_cells, 0)
        i1 = np.minimum(ijk + r_cells + 1, [Nx, Ny, Nz])

        # Build local mesh and falloff.
        xs = np.arange(i0[0], i1[0])
        ys = np.arange(i0[1], i1[1])
        zs = np.arange(i0[2], i1[2])
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        d2 = (X - ijk[0]) ** 2 + (Y - ijk[1]) ** 2 + (Z - ijk[2]) ** 2
        falloff = np.clip(1.0 - np.sqrt(d2) / max(r_cells, 1), 0.0, 1.0).astype(np.float32)
        sl = (
            slice(int(i0[0]), int(i1[0])),
            slice(int(i0[1]), int(i1[1])),
            slice(int(i0[2]), int(i1[2])),
        )
        self.temp[sl] = np.maximum(
            self.temp[sl],
            self.ambient_c + (temp_c - self.ambient_c) * falloff,
        )
        self.flame[sl] = np.maximum(self.flame[sl], 0.85 * falloff)
        # Local smoke seed so the plume is visible from t=0.
        self.smoke[sl] = np.maximum(self.smoke[sl], 0.4 * falloff * smoke_yield)
        # Fuel floor inside the sphere (only where currently lower). Set
        # well above the propagation flammable_threshold so the source
        # voxels reliably ignite and stay alight.
        self.fuel[sl] = np.maximum(self.fuel[sl], 0.85 * falloff)
        return sl, falloff
