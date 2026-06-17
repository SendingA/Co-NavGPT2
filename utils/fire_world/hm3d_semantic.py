"""Parser for HM3D ``*.semantic.glb`` + ``*.semantic.txt`` pairs.

Two facts that took us a while to nail down:

1. The vertex COLOR_0 attribute in HM3D semantic.glb is **linear uint16
   intensity** that must be converted to **sRGB-encoded 8-bit** before it
   matches the 6-hex IDs in semantic.txt::

       linear = u16 / 65535
       srgb   = sRGB_OETF(linear)              # the IEC 61966-2-1 curve
       hex    = round(srgb * 255)

   In particular, ``round(u16 / 65535 * 255)`` (which would be the
   "naive" linear remap) does **not** match - we tested it and found
   0/209 hits on a sample scene. The sRGB OETF is the right decoder,
   confirmed at ~187/209 (the rest come from primitives that contain
   *multiple* instance colours, which is point 2).

2. HM3D splits meshes into geometry chunks for view-frustum culling, NOT
   per instance. A single primitive routinely carries 5-50 distinct
   instance colours. So "one mesh = one instance" is wrong; we group
   triangles by their per-vertex colour to recover instances.

This module only depends on numpy and the stdlib so it can be invoked
without rendering anything.
"""
from __future__ import annotations

import dataclasses
import io
import json
import re
import struct
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


GLB_MAGIC = 0x46546C67
GLB_CHUNK_JSON = 0x4E4F534A
GLB_CHUNK_BIN = 0x004E4942


# ---------------------------------------------------------------------------
# Color decoder
# ---------------------------------------------------------------------------
def hm3d_u16_to_srgb_hex(rgb_u16: np.ndarray) -> np.ndarray:
    """``(N, 3) uint16`` linear vertex colour -> ``(N,)`` 6-char hex.

    The sRGB OETF (IEC 61966-2-1) is::

        srgb(L) = 12.92 * L                            for L <= 0.0031308
        srgb(L) = 1.055 * L**(1/2.4) - 0.055           otherwise

    where ``L`` is the linear intensity in [0, 1].
    """
    rgb_u16 = np.asarray(rgb_u16)
    if rgb_u16.ndim == 1:
        rgb_u16 = rgb_u16[None, :]
    lin = rgb_u16.astype(np.float64) / 65535.0
    lin = np.clip(lin, 0.0, 1.0)
    srgb = np.where(
        lin <= 0.0031308,
        12.92 * lin,
        1.055 * np.power(lin, 1.0 / 2.4) - 0.055,
    )
    u8 = np.clip(np.round(srgb * 255.0), 0, 255).astype(np.uint8)
    out = np.empty(u8.shape[0], dtype="<U6")
    for i in range(u8.shape[0]):
        out[i] = "{:02X}{:02X}{:02X}".format(int(u8[i, 0]), int(u8[i, 1]), int(u8[i, 2]))
    return out


def hm3d_u16_to_srgb_uint8(rgb_u16: np.ndarray) -> np.ndarray:
    """Same as :func:`hm3d_u16_to_srgb_hex` but returns an ``(N, 3) uint8``
    array (faster when callers want an integer key rather than a hex str).
    """
    rgb_u16 = np.asarray(rgb_u16)
    lin = rgb_u16.astype(np.float64) / 65535.0
    lin = np.clip(lin, 0.0, 1.0)
    srgb = np.where(
        lin <= 0.0031308,
        12.92 * lin,
        1.055 * np.power(lin, 1.0 / 2.4) - 0.055,
    )
    return np.clip(np.round(srgb * 255.0), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# semantic.txt
# ---------------------------------------------------------------------------
_TXT_RE = re.compile(r'^(\d+),([0-9A-Fa-f]{6}),"([^"]+)",(\d+)$')


@dataclasses.dataclass
class SemanticLabel:
    instance_id: int
    color_hex: str          # 6-char upper-case
    category: str
    region_id: int


def read_semantic_txt(path: Path) -> Tuple[Dict[str, SemanticLabel], List[SemanticLabel]]:
    """Returns ``(by_hex, ordered_list)``."""
    rows: List[SemanticLabel] = []
    by_hex: Dict[str, SemanticLabel] = {}
    if not path.exists():
        return by_hex, rows
    for line in path.read_text().splitlines():
        m = _TXT_RE.match(line.strip())
        if not m:
            continue
        lbl = SemanticLabel(
            instance_id=int(m.group(1)),
            color_hex=m.group(2).upper(),
            category=m.group(3),
            region_id=int(m.group(4)),
        )
        rows.append(lbl)
        by_hex[lbl.color_hex] = lbl
    return by_hex, rows


# ---------------------------------------------------------------------------
# GLB parsing (small, dependency-free)
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class GlbView:
    raw: bytes
    json: dict
    bin_off: int

    def buffer_view(self, idx: int) -> memoryview:
        bv = self.json['bufferViews'][idx]
        s = self.bin_off + (bv.get('byteOffset') or 0)
        return memoryview(self.raw)[s: s + bv['byteLength']]


def parse_glb(path: Path) -> GlbView:
    buf = path.read_bytes()
    magic, _, total = struct.unpack_from('<III', buf, 0)
    if magic != GLB_MAGIC:
        raise ValueError(f"not a GLB: {path}")
    off = 12
    chunks = []
    while off < total:
        clen, ctype = struct.unpack_from('<II', buf, off)
        chunks.append((ctype, off + 8, clen))
        off = off + 8 + clen
    cj = next(c for c in chunks if c[0] == GLB_CHUNK_JSON)
    cb = next(c for c in chunks if c[0] == GLB_CHUNK_BIN)
    gj = json.loads(buf[cj[1]: cj[1] + cj[2]].decode('utf-8'))
    return GlbView(raw=buf, json=gj, bin_off=cb[1])


_DT_MAP = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_TYPE_DIMS = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}


def read_accessor(g: GlbView, idx: int) -> np.ndarray:
    a = g.json['accessors'][idx]
    bv = g.json['bufferViews'][a['bufferView']]
    s = g.bin_off + (bv.get('byteOffset') or 0) + (a.get('byteOffset') or 0)
    cnt = a['count']
    dim = _TYPE_DIMS[a['type']]
    arr = np.frombuffer(
        g.raw, dtype=_DT_MAP[a['componentType']], count=cnt * dim, offset=s
    )
    return arr.reshape(cnt, dim) if dim > 1 else arr


# ---------------------------------------------------------------------------
# Per-instance aggregation
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class InstanceGeom:
    instance_id: int
    color_hex: str
    category: str
    region_id: int
    aabb_min: np.ndarray  # (3,)
    aabb_max: np.ndarray  # (3,)
    centroid: np.ndarray  # (3,)
    n_vertices: int
    n_faces: int

    def as_dict(self) -> Dict:
        return {
            "instance_id": int(self.instance_id),
            "category": self.category,
            "color_hex": self.color_hex,
            "region_id": int(self.region_id),
            "aabb_min": self.aabb_min.tolist(),
            "aabb_max": self.aabb_max.tolist(),
            "centroid": self.centroid.tolist(),
            "n_vertices": int(self.n_vertices),
            "n_faces": int(self.n_faces),
        }


def aggregate_instances(
    glb_path: Path,
    txt_path: Path,
    progress: bool = False,
    apply_habitat_axis_fix: bool = True,
) -> Tuple[List[InstanceGeom], Dict]:
    """Walk every primitive, group its faces by HM3D instance colour, and
    accumulate per-instance vertex sets to AABBs and centroids.

    Coordinates: HM3D ``*.semantic.glb`` stores vertices in the GLB
    convention (Y in the GLB file is the original Z up direction), but
    Habitat reports world coordinates with Y up after applying
    ``R = [[1,0,0],[0,0,-1],[0,1,0]]`` (a -90 deg rotation about X).
    With ``apply_habitat_axis_fix=True`` we apply the same rotation here
    so the resulting AABBs match the positions Habitat / ObjectGoal use
    (y is up, +z is the agent's forward at scene zero).

    Returns ``(instance_list, summary_dict)``.
    """
    by_hex, rows = read_semantic_txt(txt_path)
    g = parse_glb(glb_path)

    # Buffers per instance: list of vertex chunks + face count.
    verts_by_id: Dict[int, List[np.ndarray]] = defaultdict(list)
    faces_by_id: Dict[int, int] = defaultdict(int)
    unmapped_face_count = 0
    total_faces = 0

    n_primitives = len(g.json.get('meshes', []))
    for mi, mesh in enumerate(g.json.get('meshes', [])):
        if progress and (mi % 50 == 0 or mi == n_primitives - 1):
            print(f"  primitives: {mi + 1}/{n_primitives}", flush=True)
        prim = mesh['primitives'][0]
        attrs = prim.get('attributes', {})
        if 'POSITION' not in attrs or 'COLOR_0' not in attrs or 'indices' not in prim:
            continue
        pos = read_accessor(g, attrs['POSITION'])               # (V, 3) f32
        if apply_habitat_axis_fix:
            # GLB is +Z-up; Habitat is +Y-up. Rotate -90 deg about X:
            # (x, y, z)_glb -> (x, z, -y)_habitat
            pos = np.column_stack([pos[:, 0], pos[:, 2], -pos[:, 1]]).astype(np.float32)
        col_u16 = read_accessor(g, attrs['COLOR_0'])[:, :3]     # (V, 3) u16
        idx = read_accessor(g, prim['indices'])                 # (F*3,)
        idx = idx.reshape(-1)
        faces = idx.reshape(-1, 3)
        total_faces += faces.shape[0]

        # Decode each *vertex* color once; HM3D stores all three vertices
        # of a face with the same instance colour, so we group by the
        # first vertex of each face.
        face_v0 = faces[:, 0]
        u8 = hm3d_u16_to_srgb_uint8(col_u16[face_v0])     # (F, 3)
        # Map per-face hex string -> instance id via by_hex
        # Use uint32 packing for fast bucketing.
        keys = (u8[:, 0].astype(np.uint32) << 16) | (u8[:, 1].astype(np.uint32) << 8) | u8[:, 2].astype(np.uint32)
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        for ki, k in enumerate(unique_keys):
            r = (k >> 16) & 0xFF
            gx = (k >> 8) & 0xFF
            b = k & 0xFF
            hex_str = "{:02X}{:02X}{:02X}".format(int(r), int(gx), int(b))
            lbl = by_hex.get(hex_str)
            if lbl is None:
                # Unknown colour (HM3D occasionally stores boundary
                # smoothing artefacts; small triangle counts).
                cnt = int((inverse == ki).sum())
                unmapped_face_count += cnt
                continue
            face_mask = inverse == ki
            face_indices = faces[face_mask].ravel()
            verts = pos[face_indices]
            verts_by_id[lbl.instance_id].append(verts)
            faces_by_id[lbl.instance_id] += int(face_mask.sum())

    out: List[InstanceGeom] = []
    for inst_id, vert_chunks in verts_by_id.items():
        lbl = by_hex.get(_id_to_hex(rows, inst_id))
        if lbl is None:
            continue
        verts = np.concatenate(vert_chunks, axis=0)
        out.append(InstanceGeom(
            instance_id=inst_id,
            color_hex=lbl.color_hex,
            category=lbl.category,
            region_id=lbl.region_id,
            aabb_min=verts.min(axis=0).astype(np.float64),
            aabb_max=verts.max(axis=0).astype(np.float64),
            centroid=verts.mean(axis=0).astype(np.float64),
            n_vertices=int(verts.shape[0]),
            n_faces=faces_by_id[inst_id],
        ))
    out.sort(key=lambda x: x.instance_id)

    summary = {
        "total_faces": int(total_faces),
        "unmapped_face_count": int(unmapped_face_count),
        "n_instances_in_txt": len(rows),
        "n_instances_recovered": len(out),
    }
    return out, summary


def _id_to_hex(rows: List[SemanticLabel], inst_id: int) -> Optional[str]:
    for r in rows:
        if r.instance_id == inst_id:
            return r.color_hex
    return None


# ---------------------------------------------------------------------------
# Tiny self-test (CLI)
# ---------------------------------------------------------------------------
def _cli() -> int:
    import argparse
    p = argparse.ArgumentParser(description="Parse HM3D semantic.glb to a per-instance AABB list")
    p.add_argument("--glb", required=True)
    p.add_argument("--txt", required=True)
    p.add_argument("--top", type=int, default=10)
    args = p.parse_args()

    instances, summary = aggregate_instances(Path(args.glb), Path(args.txt), progress=True)
    print(json.dumps(summary, indent=2))
    print(f"recovered {len(instances)} / {summary['n_instances_in_txt']} instances")
    print()
    print(f"top {args.top} largest instances by face count:")
    instances.sort(key=lambda x: -x.n_faces)
    for inst in instances[: args.top]:
        ext = (inst.aabb_max - inst.aabb_min).round(2).tolist()
        print(f"  id={inst.instance_id:>3}  cat={inst.category:>22}  "
              f"faces={inst.n_faces:>5}  extent_xyz={ext}  "
              f"centroid={inst.centroid.round(2).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
