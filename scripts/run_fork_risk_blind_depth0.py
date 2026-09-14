"""Repeat saved person/bed fork cases with smoky depth or no fire; export MP4s.

Run with the co-nav3 Python environment. Inputs are reused without rebuilding
datasets or fire timelines. --dry-run validates arguments and records commands.
--no-fire uses clean normal observations with FireWorld and risk disabled.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "person": "person_fork_detour_nfv_three_source",
    "bed": "fork_detour_nfv_three_source",
}


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def sha256(path):
    with path.open("rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(case, no_fire=False):
    base = ROOT / "outputs/fire_cost_experiments" / CASES[case]
    output = base / ("normal_no_fire" if no_fire else "risk_blind_depth0")
    output.mkdir(exist_ok=True)
    log = base / "risk_none/navigation/logs/co_ut/output.log"
    line = next(line for line in log.read_text().splitlines() if "Namespace(" in line)
    tree = ast.parse(line[line.index("Namespace("):], mode="eval").body
    baseline = {kw.arg: ast.literal_eval(kw.value) for kw in tree.keywords}
    params = dict(baseline)
    params.update(
        depth_use_clean=0, risk_enabled=1, risk_source="none",
        risk_run_id=f"{case}-fork-three-source-risk-blind-depth0",
        dump_location=str(output.relative_to(ROOT) / "navigation"),
        fire_dump_dir=str(output.relative_to(ROOT) / "fire"),
        risk_dump_dir=str(output.relative_to(ROOT) / "risk"),
    )
    if no_fire:
        params.update(
            fire_world=0, fire_world_plan_id=None, fire_apply_to_obs=0,
            smoke_density=0.0, depth_use_clean=1, use_thermal_perception=0,
            risk_enabled=0, risk_source="none",
            risk_run_id=f"{case}-fork-normal-no-fire",
        )
    command = [sys.executable, "main.py"]
    for key, value in params.items():
        if key not in {"cuda", "turn_angle"} and value is not None:
            command.extend([f"--{key}", str(value)])
    # Check actual CLI/config parsing before any expensive simulation.
    sys.path.insert(0, str(ROOT))
    from arguments import get_args, load_config
    saved_argv = sys.argv
    try:
        sys.argv = command[1:]
        parsed = get_args()
        config = load_config(parsed)
    finally:
        sys.argv = saved_argv
    dataset = ROOT / params["dataset_path"]
    shard = dataset.parent / "content/Nfvxx8J5NCo.json.gz"
    with gzip.open(shard, "rt") as handle:
        episodes = json.load(handle)["episodes"]
    assert len(episodes) == 1
    inputs = [dataset, shard, ROOT / "configs" / params["task_config"], ROOT / "configs/multi_objectnav_hm3d.yaml"]
    if not no_fire:
        plan = ROOT / "scenes/Nfvxx8J5NCo/plans" / (params["fire_world_plan_id"] + ".json")
        timeline = ROOT / "outputs/fire_world/Nfvxx8J5NCo" / params["fire_world_plan_id"] / "timeline.npz"
        inputs.extend([plan, timeline])
    fingerprints = {str(path.relative_to(ROOT)): sha256(path) for path in inputs}
    manifest = dict(
        created_at=datetime.now(timezone.utc).isoformat(), case=case, no_fire=no_fire,
        status="prepared", baseline_log=str(log.relative_to(ROOT)),
        baseline_parameters=baseline, requested_parameters=params,
        parameter_changes={k: {"before": baseline.get(k), "after": v} for k, v in params.items() if baseline.get(k) != v},
        episode={k: v for k, v in episodes[0].items() if k not in {"goals", "shortest_paths"}},
        input_sha256=fingerprints, command=command, shell_command=shlex.join(command),
        resolved_config=__import__("omegaconf").OmegaConf.to_container(config, resolve=True),
        notes=("Historical navigation CLI parameters and original dataset retained; current repository runtime. FireWorld, fire observation processing, smoke and risk runtime disabled; clean native RGB/depth observations and native navigation metrics. No fire plan or timeline loaded."
               if no_fire else "Historical scalar CLI parameters retained; current repository runtime. Existing fire timeline reused without rebaking. depth_use_clean=0 applies smoky depth to mapping; risk_source=none disables risk in planning but retains evaluation."),
    )
    return output, manifest


def export(output, fps):
    import cv2
    import imageio_ffmpeg
    from PIL import Image
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    folders = list((output / "navigation/dump/co_ut").glob("episodes_multi/*/eps_*"))
    assert len(folders) == 1, folders
    video_dir = output / "videos"
    video_dir.mkdir(exist_ok=True)
    records = []
    for view, prefix in [("agent_0", "agent-0-Vis-"), ("agent_1", "agent-1-Vis-"), ("merged", "Merged_Vis-")]:
        frames = sorted(folders[0].glob(prefix + "*.png"), key=lambda p: int(p.stem.rsplit("-", 1)[1]))
        indices = [int(p.stem.rsplit("-", 1)[1]) for p in frames]
        assert indices and indices == list(range(indices[0], indices[-1] + 1))
        sizes = set()
        for frame in frames:
            with Image.open(frame) as im:
                sizes.add(im.size)
                im.verify()
        assert len(sizes) == 1, sizes
        width, height = sizes.pop()
        dest = video_dir / (view + ".mp4")
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-framerate", str(fps), "-start_number", str(indices[0]), "-i", str(folders[0] / (prefix + "%d.png")), "-frames:v", str(len(frames)), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:color=white", "-c:v", "libx264", "-threads", "2", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dest)]
        subprocess.run(command, check=True)
        cap = cv2.VideoCapture(str(dest))
        assert cap.isOpened()
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            assert frame.shape[:2] == (height + height % 2, width + width % 2)
            count += 1
        cap.release()
        assert count == len(frames) and abs(actual_fps - fps) < 1e-6
        subprocess.run([ffmpeg, "-v", "error", "-xerror", "-i", str(dest), "-f", "null", "-"], check=True)
        records.append(dict(view=view, path=str(dest.relative_to(ROOT)), frames=count, fps=fps, duration_seconds=count / fps, first_frame=frames[0].name, last_frame=frames[-1].name, width=width + width % 2, height=height + height % 2, bytes=dest.stat().st_size, full_decode="passed", command=command))
        print(f"Validated {dest}: {count} frames, {count / fps:.1f} seconds", flush=True)
    assert len({r["frames"] for r in records}) == 1
    save_json(video_dir / "manifest.json", dict(videos=records))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", *CASES], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--no-fire", action="store_true", help="run normal navigation with clean observations and no fire/risk runtime")
    parser.add_argument("--fps", type=int, default=5)
    args = parser.parse_args()
    if args.fps <= 0 or (args.dry_run and args.export_only):
        parser.error("fps must be positive; dry-run and export-only are exclusive")
    os.chdir(ROOT)
    for case in CASES if args.case == "all" else [args.case]:
        output, manifest = prepare(case, no_fire=args.no_fire)
        path = output / "run_manifest.json"
        if args.export_only:
            manifest = json.loads(path.read_text())
            assert manifest["status"] in {"simulation_completed", "completed"}
        else:
            if (output / "navigation").exists():
                raise FileExistsError(f"Existing navigation output: {output}; use --export-only to export a finished run")
            save_json(path, manifest)
            print(manifest["shell_command"], flush=True)
            if args.dry_run:
                continue
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; run with GPU access to preserve original rendering settings")
            env = dict(os.environ, MAGNUM_LOG="quiet", HABITAT_SIM_LOG="quiet", MPLCONFIGDIR="/tmp/matplotlib-cache", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
            manifest["status"] = "running"
            save_json(path, manifest)
            with (output / "console.log").open("w") as log:
                result = subprocess.run(manifest["command"], env=env, stdout=log, stderr=subprocess.STDOUT)
            manifest["returncode"] = result.returncode
            manifest["status"] = "simulation_completed" if result.returncode == 0 else "failed"
            save_json(path, manifest)
            result.check_returncode()
        for relative, expected in manifest["input_sha256"].items():
            assert sha256(ROOT / relative) == expected, f"Input changed during run: {relative}"
        metrics = json.loads((output / "navigation/metrics/resume_state.json").read_text())
        assert metrics["episodes_completed"] == 1, metrics
        manifest["metrics"] = metrics
        manifest["videos"] = export(output, args.fps)
        manifest["status"] = "completed"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        save_json(path, manifest)


if __name__ == "__main__":
    main()
