#!/usr/bin/env python3
"""
Standalone CLI: Simulate mmWave radar point clouds from saved RGBD point clouds.

- Scans an input folder for per-episode per-agent RGBD PLY files
- Generates sparse mmWave radar point clouds via MMWaveRadarSimulator
- Produces fused point clouds (RGBD + Radar) and comparison images

Usage:
  python scripts/simulate_mmwave_from_pointclouds.py \
    --input ./tmp/pointclouds \
    --output ./tmp/pointclouds_radar \
    --beams-h 128 --beams-v 8 --range-max 50 --range-std 0.03

Notes:
- Requires Open3D and Matplotlib.
- Works purely offline on previously exported PLYs.
"""

import os
import sys
import argparse
import glob
import logging
import numpy as np

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import open3d as o3d
except Exception as e:
    print("Error: open3d not available. Install with: pip install open3d")
    raise

# Ensure project root is on path for utils import
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.mmwave_radar_simulator import MMWaveRadarSimulator


def find_rgbd_plys(input_dir: str):
    """Find candidate RGBD PLY files. Avoid already-generated radar/fused files."""
    patterns = [
        os.path.join(input_dir, "**", "*_rgbd.ply"),
        os.path.join(input_dir, "**", "episode_*_agent_*.ply"),
        os.path.join(input_dir, "**", "ep*_ag*_rgbd.ply"),
    ]
    files = []
    for pat in patterns:
        files += glob.glob(pat, recursive=True)
    # Deduplicate
    files = sorted(set(files))
    # Filter out radar/fused
    files = [f for f in files if ("_radar" not in f and "_fused" not in f)]
    return files


def _axis_spread(pts: np.ndarray):
    if pts.size == 0:
        return (0.0, 0.0, 0.0)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    spread = maxs - mins
    return tuple(spread.tolist())


def _choose_top_plane(pts: np.ndarray, top_plane: str):
    # Determine which axes form the top view plane
    if top_plane == "auto":
        sx, sy, sz = _axis_spread(pts)
        # Treat smallest spread as the up-axis
        spreads = {"x": sx, "y": sy, "z": sz}
        up_axis = min(spreads, key=spreads.get)
        if up_axis == "x":
            return ("y", "z")
        if up_axis == "y":
            return ("x", "z")
        return ("x", "y")
    if top_plane == "xy":
        return ("x", "y")
    if top_plane == "xz":
        return ("x", "z")
    if top_plane == "yz":
        return ("y", "z")
    # default fallback
    return ("x", "y")


def _to_indices(ax_pair):
    m = {"x": 0, "y": 1, "z": 2}
    return (m[ax_pair[0]], m[ax_pair[1]])


def _euler_to_matrix(roll: float, pitch: float, yaw: float):
    # Angles in degrees -> radians
    r = np.deg2rad(roll)
    p = np.deg2rad(pitch)
    y = np.deg2rad(yaw)
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    # ZYX order (yaw, pitch, roll)
    return Rz @ Ry @ Rx


def _transform_pcd(pcd: o3d.geometry.PointCloud, R: np.ndarray, t: np.ndarray):
    q = o3d.geometry.PointCloud(pcd)
    pts = np.asarray(q.points)
    if pts.size > 0:
        pts = (R @ pts.T).T + t
        q.points = o3d.utility.Vector3dVector(pts)
    return q


def _centroid_xyz(pts: np.ndarray):
    if pts.size == 0:
        return np.zeros(3)
    return pts.mean(axis=0)


def derive_radar_from_rgbd(rgbd_pcd: o3d.geometry.PointCloud,
                           n_beams_h: int,
                           n_beams_v: int,
                           range_max: float,
                           range_std: float,
                           origin_cam: np.ndarray,
                           R_sensor_cam: np.ndarray) -> o3d.geometry.PointCloud:
    """Directly derive sparse radar-like returns by binning RGBD points into beam indices.
    - origin_cam: sensor origin in camera/world frame (3,)
    - R_sensor_cam: rotation matrix mapping sensor axes into camera/world frame (3x3)
    Returns: PointCloud in camera/world frame with orange color.
    """
    pts = np.asarray(rgbd_pcd.points)
    if pts.size == 0:
        return o3d.geometry.PointCloud()

    # Transform points into sensor local frame: p_s = R^T * (p_c - origin)
    rel = pts - origin_cam[None, :]
    rel_s = (R_sensor_cam.T @ rel.T).T

    x, y, z = rel_s[:, 0], rel_s[:, 1], rel_s[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    mask = r > 0
    if range_max is not None:
        mask &= (r <= range_max)
    if not np.any(mask):
        return o3d.geometry.PointCloud()
    x, y, z, r = x[mask], y[mask], z[mask], r[mask]
    idx_all = np.nonzero(mask)[0]

    # Angles
    az = np.arctan2(y, x)  # [-pi, pi]
    el = np.arctan2(z, np.sqrt(x * x + y * y))  # [-pi/2, pi/2]

    # Discretize into beams
    az_min, az_max = -np.pi, np.pi
    el_min, el_max = -np.pi / 2.0, np.pi / 2.0
    az_bin = (az - az_min) / (az_max - az_min) * n_beams_h
    el_bin = (el - el_min) / (el_max - el_min) * n_beams_v
    h_idx = np.clip(np.floor(az_bin).astype(int), 0, n_beams_h - 1)
    v_idx = np.clip(np.floor(el_bin).astype(int), 0, n_beams_v - 1)

    # Select nearest point per beam cell
    key = h_idx * n_beams_v + v_idx
    # For efficiency, sort by range and take first occurrence per key
    order = np.argsort(r)
    key_sorted = key[order]
    unique_keys, first_pos = np.unique(key_sorted, return_index=True)
    chosen = order[first_pos]

    # Reconstruct chosen points in camera/world frame
    rel_s_chosen = np.stack([x[chosen], y[chosen], z[chosen]], axis=1)
    rel_c = (R_sensor_cam @ rel_s_chosen.T).T
    pts_c = rel_c + origin_cam[None, :]

    # Add radial noise
    if range_std and range_std > 0:
        # Perturb along local sensor ray direction
        dirs_s = rel_s_chosen / (np.linalg.norm(rel_s_chosen, axis=1, keepdims=True) + 1e-8)
        noise = np.random.normal(0.0, range_std, size=(dirs_s.shape[0], 1)) * dirs_s
        noise_c = (R_sensor_cam @ noise.T).T
        pts_c = pts_c + noise_c

    radar_pcd = o3d.geometry.PointCloud()
    radar_pcd.points = o3d.utility.Vector3dVector(pts_c)
    radar_pcd.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1.0, 0.2, 0.0]]), (pts_c.shape[0], 1)))
    return radar_pcd


def save_comparison_image(rgbd_pcd: o3d.geometry.PointCloud,
                          radar_pcd: o3d.geometry.PointCloud,
                          fused_pcd: o3d.geometry.PointCloud,
                          out_path: str,
                          title: str,
                          top_plane: str = "auto"):
    pts = np.asarray(rgbd_pcd.points)
    cols = np.asarray(rgbd_pcd.colors) if len(rgbd_pcd.colors) > 0 else None

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=150)
    fig.suptitle(title, fontsize=16)

    # Determine top-plane axes
    ax_pair = _choose_top_plane(pts, top_plane)
    ai, aj = _to_indices(ax_pair)

    # RGBD top
    if len(pts) > 0:
        axes[0, 0].scatter(pts[:, ai], pts[:, aj], c=cols if cols is not None else "tab:cyan", s=0.5, alpha=0.7)
        axes[0, 0].set_title(f'RGBD Top View ({ax_pair[0]}-{ax_pair[1]}) ({len(pts)} points)')
    else:
        axes[0, 0].text(0.5, 0.5, 'Empty RGBD', ha='center', va='center')
    axes[0, 0].set_xlabel(f'{ax_pair[0].upper()} (m)')
    axes[0, 0].set_ylabel(f'{ax_pair[1].upper()} (m)')
    axes[0, 0].axis('equal')
    axes[0, 0].grid(alpha=0.2)

    # RGBD side
    if len(pts) > 0:
        axes[0, 1].scatter(pts[:, 0], pts[:, 2], c=cols if cols is not None else "tab:cyan", s=0.5, alpha=0.7)
        axes[0, 1].set_title('RGBD Side View')
    else:
        axes[0, 1].text(0.5, 0.5, 'Empty RGBD', ha='center', va='center')
    axes[0, 1].set_xlabel('X (m)')
    axes[0, 1].set_ylabel('Z (m)')
    axes[0, 1].axis('equal')
    axes[0, 1].grid(alpha=0.2)

    # Radar top
    if len(radar_pcd.points) > 0:
        rpts = np.asarray(radar_pcd.points)
        rcols = np.asarray(radar_pcd.colors) if len(radar_pcd.colors) > 0 else None
        axes[1, 0].scatter(rpts[:, ai], rpts[:, aj], c=rcols if rcols is not None else "tab:red", s=2, alpha=0.8)
        axes[1, 0].set_title(f'mmWave Radar Top View ({ax_pair[0]}-{ax_pair[1]}) ({len(radar_pcd.points)} points)')
    else:
        axes[1, 0].text(0.5, 0.5, 'No Radar Points', ha='center', va='center')
        axes[1, 0].set_title('mmWave Radar Top View (0 points)')
    axes[1, 0].set_xlabel(f'{ax_pair[0].upper()} (m)')
    axes[1, 0].set_ylabel(f'{ax_pair[1].upper()} (m)')
    axes[1, 0].axis('equal')
    axes[1, 0].grid(alpha=0.2)

    # Fused top
    if len(fused_pcd.points) > 0:
        fpts = np.asarray(fused_pcd.points)
        fcols = np.asarray(fused_pcd.colors) if len(fused_pcd.colors) > 0 else None
        axes[1, 1].scatter(fpts[:, ai], fpts[:, aj], c=fcols if fcols is not None else "tab:purple", s=1.5, alpha=0.8)
        axes[1, 1].set_title(f'Fused (RGBD+Radar) Top View ({ax_pair[0]}-{ax_pair[1]}) ({len(fused_pcd.points)} points)')
    axes[1, 1].set_xlabel(f'{ax_pair[0].upper()} (m)')
    axes[1, 1].set_ylabel(f'{ax_pair[1].upper()} (m)')
    axes[1, 1].axis('equal')
    axes[1, 1].grid(alpha=0.2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Simulate mmWave radar from saved RGBD pointclouds")
    parser.add_argument("--input", type=str, default=os.path.join(ROOT, "tmp", "pointclouds"), help="Input folder with RGBD PLYs")
    parser.add_argument("--output", type=str, default=os.path.join(ROOT, "tmp", "pointclouds_radar"), help="Output folder for radar/fused results")
    parser.add_argument("--beams-h", type=int, default=128, help="Horizontal beams")
    parser.add_argument("--beams-v", type=int, default=8, help="Vertical beams")
    parser.add_argument("--range-max", type=float, default=50.0, help="Max range (m)")
    parser.add_argument("--range-std", type=float, default=0.03, help="Range noise std (m)")
    parser.add_argument("--top-plane", type=str, choices=["auto", "xy", "xz", "yz"], default="auto", help="Axes for top view plots")
    parser.add_argument("--mode", type=str, choices=["scan", "derive"], default="scan", help="Radar generation mode: physics scan or RGBD-derived")
    parser.add_argument("--sensor-origin", type=str, choices=["manual", "auto"], default="manual", help="Sensor origin: use extrinsics or cloud centroid")
    parser.add_argument("--radar-tx", type=float, default=0.0, help="Radar extrinsic translation X (m) in RGBD frame")
    parser.add_argument("--radar-ty", type=float, default=0.0, help="Radar extrinsic translation Y (m) in RGBD frame")
    parser.add_argument("--radar-tz", type=float, default=0.0, help="Radar extrinsic translation Z (m) in RGBD frame")
    parser.add_argument("--radar-roll", type=float, default=0.0, help="Radar extrinsic roll (deg) relative to RGBD frame")
    parser.add_argument("--radar-pitch", type=float, default=0.0, help="Radar extrinsic pitch (deg) relative to RGBD frame")
    parser.add_argument("--radar-yaw", type=float, default=0.0, help="Radar extrinsic yaw (deg) relative to RGBD frame")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Input:  {args.input}")
    logging.info(f"Output: {args.output}")

    files = find_rgbd_plys(args.input)
    if not files:
        logging.warning("No RGBD PLYs found. Ensure you exported per-episode per-agent PLYs.")
        return
    logging.info(f"Found {len(files)} RGBD files")

    radar_sim = MMWaveRadarSimulator(
        n_beams_h=args.beams_h,
        n_beams_v=args.beams_v,
        range_max=args.range_max,
        range_std=args.range_std,
    )

    # Build extrinsic transforms
    R_r2c = _euler_to_matrix(args.radar_roll, args.radar_pitch, args.radar_yaw)  # radar->camera rotation
    t_r2c = np.array([args.radar_tx, args.radar_ty, args.radar_tz])               # radar->camera translation
    # Inverse: camera->radar
    R_c2r = R_r2c.T
    t_c2r = -(R_c2r @ t_r2c)

    for filepath in files:
        rel = os.path.relpath(filepath, args.input)
        base = os.path.splitext(os.path.basename(filepath))[0]
        out_dir = os.path.join(args.output, os.path.dirname(rel))
        os.makedirs(out_dir, exist_ok=True)

        # Load RGBD point cloud
        rgbd_pcd = o3d.io.read_point_cloud(filepath)
        if len(rgbd_pcd.points) == 0:
            logging.warning(f"Empty PLY skipped: {filepath}")
            continue

        # Decide sensor origin in camera/world frame
        pts_np = np.asarray(rgbd_pcd.points)
        if args.sensor_origin == "auto" and pts_np.size > 0:
            origin_cam = _centroid_xyz(pts_np)
        else:
            origin_cam = t_r2c

        if args.mode == "scan":
            # Transform RGBD to radar frame for simulation (camera->radar)
            rgbd_in_radar = _transform_pcd(rgbd_pcd, R_c2r, t_c2r)

            # Simulate radar in radar frame
            radar_in_radar, radar_info = radar_sim.simulate_radar_pointcloud(rgbd_in_radar)

            # Transform radar back to camera/RGBD frame (radar->camera)
            radar_pcd = _transform_pcd(radar_in_radar, R_r2c, t_r2c)
        else:
            # Directly derive radar points by binning onto beams from origin_cam with orientation R_r2c
            radar_pcd = derive_radar_from_rgbd(
                rgbd_pcd,
                n_beams_h=args.beams_h,
                n_beams_v=args.beams_v,
                range_max=args.range_max,
                range_std=args.range_std,
                origin_cam=origin_cam,
                R_sensor_cam=R_r2c,
            )
            radar_info = {
                "n_points": len(radar_pcd.points),
                "point_density": (len(radar_pcd.points) / max(len(rgbd_pcd.points), 1)),
                "avg_reflectance": 0.0,
            }

        fused_pcd = radar_sim.fuse_pointclouds(rgbd_pcd, radar_pcd)

        logging.info(f"Processed {rel}: RGBD={len(rgbd_pcd.points)} Radar={radar_info['n_points']} Fused={len(fused_pcd.points)}")

        # Save PLYs
        rgbd_out = os.path.join(out_dir, f"{base}_rgbd.ply") if not base.endswith("_rgbd") else os.path.join(out_dir, f"{base}.ply")
        radar_out = os.path.join(out_dir, f"{base}_radar.ply")
        fused_out = os.path.join(out_dir, f"{base}_fused.ply")
        try:
            # Copy/Rewrite RGBD to output for consistency
            o3d.io.write_point_cloud(rgbd_out, rgbd_pcd)
            o3d.io.write_point_cloud(radar_out, radar_pcd)
            o3d.io.write_point_cloud(fused_out, fused_pcd)
        except Exception as e:
            logging.error(f"PLY export failed for {rel}: {e}")

        # Save comparison image
        img_out = os.path.join(out_dir, f"{base}_comparison.png")
        title = f"{base}"
        save_comparison_image(rgbd_pcd, radar_pcd, fused_pcd, img_out, title, top_plane=args.top_plane)

    logging.info("Done.")


if __name__ == "__main__":
    main()
