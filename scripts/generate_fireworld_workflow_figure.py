#!/usr/bin/env python3
"""Generate the publication workflow figure for Co-NavGPT2 FireWorld.

The figure deliberately distinguishes offline fire propagation from online,
pose-conditioned observation rendering.  It follows the visual theme of the
Robot-Sim/Fire-Sim reference diagram without claiming a ROS or online fire-
physics loop that the active implementation does not use.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/conavgpt2-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


Color = Tuple[float, float, float]

COLORS = {
    "ink": "#263342",
    "muted": "#647180",
    "line": "#4E5965",
    "panel": "#EEF1F4",
    "panel_edge": "#D8DDE3",
    "robot": "#EF685B",
    "robot_light": "#FDE9E6",
    "sensor": "#4D82D0",
    "sensor_light": "#E7F0FC",
    "fire": "#F19A32",
    "fire_light": "#FDF0DE",
    "risk": "#45A46C",
    "risk_light": "#E4F4EB",
    "white": "#FFFFFF",
}


def rounded_box(
    ax,
    xy: Tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str = "none",
    radius: float = 0.018,
    linewidth: float = 1.2,
    zorder: int = 1,
):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        transform=ax.transAxes,
        clip_on=False,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def node(
    ax,
    center: Tuple[float, float],
    size: Tuple[float, float],
    text: str,
    *,
    color: str,
    fontsize: float = 9.3,
    textcolor: str = COLORS["ink"],
    linewidth: float = 0.0,
    edgecolor: str = "none",
):
    x, y = center
    w, h = size
    rounded_box(
        ax,
        (x - w / 2, y - h / 2),
        w,
        h,
        facecolor=color,
        edgecolor=edgecolor,
        linewidth=linewidth,
        radius=0.012,
        zorder=4,
    )
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=fontsize,
        color=textcolor,
        fontweight="medium",
        linespacing=1.18,
        zorder=5,
    )


def arrow(
    ax,
    start: Tuple[float, float],
    end: Tuple[float, float],
    *,
    color: str = COLORS["line"],
    dashed: bool = False,
    curved: float = 0.0,
    linewidth: float = 1.45,
    mutation_scale: float = 10.0,
    zorder: int = 2,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        connectionstyle=f"arc3,rad={curved}",
        linewidth=linewidth,
        color=color,
        linestyle=(0, (4, 3)) if dashed else "solid",
        mutation_scale=mutation_scale,
        transform=ax.transAxes,
        clip_on=False,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def panel(
    ax,
    bounds: Tuple[float, float, float, float],
    label: str,
    *,
    label_color: str,
):
    x, y, w, h = bounds
    rounded_box(
        ax,
        (x, y),
        w,
        h,
        facecolor=COLORS["panel"],
        edgecolor=COLORS["panel_edge"],
        linewidth=0.8,
        radius=0.018,
        zorder=0,
    )
    ax.text(
        x + 0.012,
        y + h - 0.025,
        label.upper(),
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=7.6,
        color=label_color,
        fontweight="bold",
        zorder=6,
    )


def label_on_arrow(ax, xy: Tuple[float, float], text: str, color: str):
    ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.2,
        color=color,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": COLORS["white"],
            "edgecolor": "none",
            "alpha": 0.96,
        },
        zorder=7,
    )


def build_figure():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.972,
        "Co-NavGPT2 FireWorld Workflow",
        ha="center",
        va="center",
        fontsize=20,
        color=COLORS["ink"],
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.943,
        "Offline fire propagation  |  Online pose-conditioned rendering  |  Risk-aware multi-agent navigation",
        ha="center",
        va="center",
        fontsize=9.5,
        color=COLORS["muted"],
        transform=ax.transAxes,
    )

    # Column titles retain the thematic framework of the reference diagram.
    headers = (
        (0.17, "Habitat Robot Simulation", COLORS["robot"]),
        (0.50, "Fire Observation & Risk", COLORS["sensor"]),
        (0.83, "FireWorld", COLORS["fire"]),
    )
    for x, title, color in headers:
        ax.text(
            x,
            0.902,
            title,
            ha="center",
            va="center",
            fontsize=14.5,
            color=COLORS["ink"],
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.plot(
            [x - 0.105, x + 0.105],
            [0.883, 0.883],
            color=color,
            linewidth=3.0,
            solid_capstyle="round",
            transform=ax.transAxes,
            zorder=3,
        )

    # ------------------------- Habitat column -------------------------
    panel(ax, (0.035, 0.765, 0.27, 0.10), "Initialization", label_color=COLORS["robot"])
    node(
        ax,
        (0.17, 0.806),
        (0.205, 0.048),
        "HM3D 3-D Scene + ObjectNav Episode",
        color=COLORS["robot"],
        textcolor=COLORS["white"],
        fontsize=9.0,
    )

    panel(ax, (0.035, 0.105, 0.27, 0.625), "Online Habitat Runtime Loop", label_color=COLORS["robot"])
    node(ax, (0.17, 0.645), (0.195, 0.065), "Action  →  env.step()", color=COLORS["robot_light"], edgecolor=COLORS["robot"], linewidth=1.2)
    node(ax, (0.17, 0.535), (0.195, 0.075), "Current Agent Pose\n+ Clean RGB-D", color=COLORS["robot"], textcolor=COLORS["white"])
    node(ax, (0.17, 0.335), (0.195, 0.075), "Per-Agent Mapping\n& Object Detection", color=COLORS["robot_light"], edgecolor=COLORS["robot"], linewidth=1.2)
    node(ax, (0.17, 0.205), (0.195, 0.075), "Global Frontier Assignment\n+ Local Planner", color=COLORS["robot"], textcolor=COLORS["white"], fontsize=8.9)
    arrow(ax, (0.17, 0.612), (0.17, 0.575), color=COLORS["robot"])
    arrow(ax, (0.17, 0.297), (0.17, 0.243), color=COLORS["robot"])
    arrow(ax, (0.07, 0.205), (0.07, 0.645), color=COLORS["robot"], curved=-0.16)
    label_on_arrow(ax, (0.066, 0.435), "navigation loop", COLORS["robot"])

    # ------------------------ FireWorld column ------------------------
    panel(ax, (0.695, 0.595, 0.27, 0.27), "Offline Asset Preparation", label_color=COLORS["fire"])
    node(ax, (0.83, 0.807), (0.205, 0.048), "Scene Scan  →  Semantic Inventory", color=COLORS["fire_light"], edgecolor=COLORS["fire"], linewidth=1.1, fontsize=8.7)
    node(ax, (0.83, 0.737), (0.205, 0.048), "Fire Plan  →  Sources & Intensity", color=COLORS["fire_light"], edgecolor=COLORS["fire"], linewidth=1.1, fontsize=8.7)
    node(ax, (0.83, 0.667), (0.205, 0.048), "FirePropagation  →  Bake Timeline", color=COLORS["fire"], textcolor=COLORS["white"], fontsize=8.7)
    arrow(ax, (0.83, 0.782), (0.83, 0.762), color=COLORS["fire"], dashed=True)
    arrow(ax, (0.83, 0.712), (0.83, 0.692), color=COLORS["fire"], dashed=True)

    panel(ax, (0.695, 0.105, 0.27, 0.455), "Online World Query (Read-Only)", label_color=COLORS["fire"])
    node(ax, (0.765, 0.455), (0.112, 0.065), "Baked\ntimeline.npz", color=COLORS["fire"], textcolor=COLORS["white"], fontsize=8.4)
    node(ax, (0.895, 0.455), (0.105, 0.065), "FireClock\nwallclock | step", color=COLORS["fire_light"], edgecolor=COLORS["fire"], linewidth=1.1, fontsize=8.1)
    node(ax, (0.83, 0.325), (0.205, 0.060), "Nearest Timeline Frame Query", color=COLORS["fire_light"], edgecolor=COLORS["fire"], linewidth=1.1)
    node(ax, (0.83, 0.190), (0.205, 0.070), "Flame  |  Smoke  |  Temperature\n3-D Voxel Fields", color=COLORS["fire"], textcolor=COLORS["white"], fontsize=8.8)
    arrow(ax, (0.83, 0.637), (0.765, 0.489), color=COLORS["fire"], dashed=True, curved=0.08)
    arrow(ax, (0.765, 0.422), (0.800, 0.357), color=COLORS["fire"])
    arrow(ax, (0.895, 0.422), (0.860, 0.357), color=COLORS["fire"])
    arrow(ax, (0.83, 0.295), (0.83, 0.225), color=COLORS["fire"])

    # ----------------- Observation and risk middle column -----------------
    panel(ax, (0.345, 0.485, 0.31, 0.245), "Synchronous In-Process Inputs", label_color=COLORS["sensor"])
    node(ax, (0.405, 0.630), (0.080, 0.050), "Pose", color=COLORS["sensor"], textcolor=COLORS["white"])
    node(ax, (0.500, 0.630), (0.080, 0.050), "Clean RGB", color=COLORS["sensor"], textcolor=COLORS["white"], fontsize=8.5)
    node(ax, (0.595, 0.630), (0.080, 0.050), "Clean Depth", color=COLORS["sensor"], textcolor=COLORS["white"], fontsize=8.5)
    node(ax, (0.500, 0.535), (0.240, 0.065), "Pose-Conditioned Voxel Ray Marching\nTorch/CUDA or NumPy", color=COLORS["sensor_light"], edgecolor=COLORS["sensor"], linewidth=1.2, fontsize=8.8)
    arrow(ax, (0.405, 0.604), (0.460, 0.568), color=COLORS["sensor"])
    arrow(ax, (0.500, 0.604), (0.500, 0.568), color=COLORS["sensor"])
    arrow(ax, (0.595, 0.604), (0.540, 0.568), color=COLORS["sensor"])

    panel(ax, (0.345, 0.105, 0.31, 0.345), "Multimodal Observation & Risk", label_color=COLORS["sensor"])
    node(
        ax,
        (0.500, 0.395),
        (0.240, 0.066),
        "FireSensor Composite\nSmoky RGB • Thermal • Transmittance",
        color=COLORS["sensor"],
        textcolor=COLORS["white"],
        fontsize=8.6,
    )
    node(
        ax,
        (0.500, 0.300),
        (0.240, 0.060),
        "Depth / LiDAR / Radar Smoke Models",
        color=COLORS["sensor_light"],
        edgecolor=COLORS["sensor"],
        linewidth=1.1,
        fontsize=8.5,
    )
    node(ax, (0.420, 0.190), (0.115, 0.070), "Patch Habitat\nObservation", color=COLORS["sensor"], textcolor=COLORS["white"], fontsize=8.4)
    node(ax, (0.580, 0.190), (0.115, 0.070), "Dynamic Risk Map\noracle | sensed | none", color=COLORS["risk"], textcolor=COLORS["white"], fontsize=8.0)
    arrow(ax, (0.500, 0.502), (0.500, 0.430), color=COLORS["sensor"])
    arrow(ax, (0.500, 0.362), (0.500, 0.331), color=COLORS["sensor"])
    arrow(ax, (0.475, 0.270), (0.430, 0.226), color=COLORS["sensor"])
    arrow(ax, (0.525, 0.270), (0.570, 0.226), color=COLORS["risk"])
    label_on_arrow(ax, (0.557, 0.253), "sensed", COLORS["risk"])

    # -------------------------- Cross-column flows -------------------------
    arrow(ax, (0.273, 0.806), (0.728, 0.807), color=COLORS["muted"], dashed=True)
    label_on_arrow(ax, (0.500, 0.822), "3-D scene geometry + semantic metadata", COLORS["muted"])

    arrow(ax, (0.268, 0.535), (0.365, 0.630), color=COLORS["sensor"])
    label_on_arrow(ax, (0.321, 0.594), "direct call", COLORS["sensor"])

    arrow(ax, (0.728, 0.190), (0.620, 0.520), color=COLORS["fire"], curved=-0.11)
    label_on_arrow(ax, (0.674, 0.350), "voxel fields", COLORS["fire"])

    arrow(ax, (0.728, 0.190), (0.638, 0.190), color=COLORS["risk"])
    label_on_arrow(ax, (0.683, 0.207), "oracle GT", COLORS["risk"])

    arrow(ax, (0.362, 0.190), (0.268, 0.335), color=COLORS["sensor"], curved=0.08)
    label_on_arrow(ax, (0.313, 0.267), "RGB-D + thermal", COLORS["sensor"])

    arrow(ax, (0.550, 0.157), (0.268, 0.190), color=COLORS["risk"], curved=0.22)
    label_on_arrow(ax, (0.410, 0.130), "risk cost / hard-unsafe mask", COLORS["risk"])

    # Legend and architectural disclosure.
    ax.plot([0.055, 0.105], [0.056, 0.056], color=COLORS["line"], linewidth=1.7, transform=ax.transAxes)
    ax.text(0.112, 0.056, "online synchronous flow", va="center", fontsize=7.7, color=COLORS["muted"], transform=ax.transAxes)
    ax.plot([0.255, 0.305], [0.056, 0.056], color=COLORS["line"], linewidth=1.7, linestyle=(0, (4, 3)), transform=ax.transAxes)
    ax.text(0.312, 0.056, "offline asset generation", va="center", fontsize=7.7, color=COLORS["muted"], transform=ax.transAxes)
    ax.text(
        0.955,
        0.056,
        "Active runtime: single-process Python calls; no ROS/IPC",
        ha="right",
        va="center",
        fontsize=8.0,
        color=COLORS["ink"],
        fontweight="bold",
        transform=ax.transAxes,
    )
    return fig


def parse_args(argv: Optional[Iterable[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper_figures/fireworld_workflow"),
    )
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    stem = args.output_dir / "conavgpt2_fireworld_workflow"
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=args.dpi, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)

    manifest = {
        "title": "Co-NavGPT2 FireWorld Workflow",
        "design_intent": (
            "Reference-inspired three-column theme with an explicit split "
            "between offline fire propagation and online observation."
        ),
        "runtime_semantics": "single-process synchronous Python calls",
        "offline_semantics": "scene scan -> plan -> propagation -> timeline.npz",
        "files": [
            stem.with_suffix(suffix).name for suffix in (".svg", ".png", ".pdf")
        ],
        "png_dpi": int(args.dpi),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(stem.with_suffix(".svg"))
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
