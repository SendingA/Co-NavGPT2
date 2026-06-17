"""Smoke tests for utils.fire_world.propagation.

Goals:
  - propagation runs end-to-end on the canonical fixture and writes a
    timeline.npz that round-trips.
  - the simulation is deterministic: two identical runs produce
    bit-identical fields.
  - basic physics invariants:
        * flame stays in [0, 1] and is non-zero at t=0 and during the
          first ignition's lifetime,
        * smoke grows over the first minute then begins decaying after
          all sources are gone,
        * temperature stays bounded and exceeds the ignition threshold at
          the source.

Run with::

    python scripts/test_fire_propagation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.fire_world.propagation import run_propagation  # noqa: E402


SCENE = "TEEsavR23oF"
PLAN_ID = "d4f8b9c253ab"


def _load_fixtures():
    inv = json.loads((ROOT / "scenes" / SCENE / "inventory.json").read_text())
    plan = json.loads((ROOT / "scenes" / SCENE / "plans" / f"{PLAN_ID}.json").read_text())
    return inv, plan


def test_run_and_roundtrip(tmp_dir: Path) -> None:
    inv, plan = _load_fixtures()
    out_dir = tmp_dir / SCENE / PLAN_ID
    res = run_propagation(
        inv, plan,
        voxel_m=0.20,            # coarser to keep test fast
        dt=0.5,
        save_dt=10.0,
        out_dir=out_dir,
        verbose=False,
    )
    assert (out_dir / "timeline.npz").exists()
    d = np.load(out_dir / "timeline.npz", allow_pickle=True)
    assert set(d.keys()) >= {"flame", "smoke", "temp", "times", "meta"}
    flame = d["flame"].astype(np.float32)
    smoke = d["smoke"].astype(np.float32)
    temp = d["temp"].astype(np.float32)

    assert flame.min() >= 0.0 and flame.max() <= 1.0 + 1e-3
    assert smoke.min() >= 0.0 and smoke.max() <= 1.0 + 1e-3
    assert temp.max() < 1500.0
    assert flame[0].max() > 0.5, "no flame at t=0"
    print(f"run_and_roundtrip: OK ({flame.shape[0]} frames, "
          f"final flame max={float(flame[-1].max()):.3f})")


def test_determinism(tmp_dir: Path) -> None:
    inv, plan = _load_fixtures()
    res1 = run_propagation(
        inv, plan, voxel_m=0.25, dt=0.5, save_dt=20.0,
        out_dir=tmp_dir / "a", verbose=False,
    )
    res2 = run_propagation(
        inv, plan, voxel_m=0.25, dt=0.5, save_dt=20.0,
        out_dir=tmp_dir / "b", verbose=False,
    )
    a = np.load(tmp_dir / "a" / "timeline.npz", allow_pickle=True)
    b = np.load(tmp_dir / "b" / "timeline.npz", allow_pickle=True)
    for k in ["flame", "smoke", "temp", "times"]:
        assert np.array_equal(a[k], b[k]), f"{k} not deterministic"
    print("determinism: OK")


def test_smoke_growth_then_decay(tmp_dir: Path) -> None:
    inv, plan = _load_fixtures()
    res = run_propagation(
        inv, plan, voxel_m=0.20, dt=0.5, save_dt=10.0,
        out_dir=tmp_dir / "c", verbose=False,
    )
    smoke = np.load(tmp_dir / "c" / "timeline.npz", allow_pickle=True)["smoke"].astype(np.float32)
    s_sum = smoke.reshape(smoke.shape[0], -1).sum(axis=1)
    # Should grow during the first 60s.
    assert s_sum[6] > s_sum[0] * 1.5, \
        f"smoke did not grow in the first minute (s[0]={s_sum[0]:.1f} s[60s]={s_sum[6]:.1f})"
    # After all sources are out (t > 0.5*duration), smoke should be
    # non-decreasing in coverage during the burn and decay afterwards.
    peak = s_sum.max()
    final = s_sum[-1]
    assert final < peak, "smoke never decayed below its peak"
    print(f"smoke growth+decay: peak sum={peak:.1f}, final={final:.1f} -> OK")


def test_temp_threshold_at_source(tmp_dir: Path) -> None:
    inv, plan = _load_fixtures()
    res = run_propagation(
        inv, plan, voxel_m=0.20, dt=0.5, save_dt=10.0,
        out_dir=tmp_dir / "d", verbose=False,
    )
    temp = np.load(tmp_dir / "d" / "timeline.npz", allow_pickle=True)["temp"].astype(np.float32)
    # The first ignition should reach > 600 C somewhere in the field.
    assert temp[0].max() > 600.0
    print(f"temp at source: {float(temp[0].max()):.1f} C -> OK")


def main() -> int:
    tmp = ROOT / "outputs" / "test_fire_propagation_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    for p in tmp.rglob("*"):
        if p.is_file():
            p.unlink()

    test_run_and_roundtrip(tmp)
    test_determinism(tmp)
    test_smoke_growth_then_decay(tmp)
    test_temp_threshold_at_source(tmp)
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
