"""Сохранить галереи исходных поз для проверки схемы и границ повторений."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rehab26_data import load_repetitions, LABEL_NAMES
from rehab26_features import extract_signals


EDGES = ((11, 12), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
         (24, 26), (26, 28), (27, 29), (29, 31), (31, 27), (28, 30), (30, 32), (32, 28))
JOINTS = sorted({j for pair in EDGES for j in pair})


def render(record, ax):
    p = np.asarray(record["poses"])
    xyz = p[:, :, :3]
    shoulders, hips = (xyz[:, 11] + xyz[:, 12]) / 2, (xyz[:, 23] + xyz[:, 24]) / 2
    scale = np.median(np.linalg.norm(shoulders - hips, axis=-1))
    centered = (xyz - hips[:, None]) / scale
    signals = extract_signals(p)
    bottom = int(np.argmin(signals[:, :2].mean(axis=1)))
    for idx, offset, color in zip((0, bottom, len(p) - 1), (-2.2, 0., 2.2), ("#8d99ae", "#0077b6", "#e09f3e")):
        xy = centered[idx, :, :2].copy()
        xy[:, 0] += offset
        for first, last in EDGES:
            ax.plot(xy[[first, last], 0], xy[[first, last], 1], color=color, linewidth=1.2)
        ax.scatter(xy[JOINTS, 0], xy[JOINTS, 1], c=color, s=8)
        low = [j for j in JOINTS if p[idx, j, 3] < .5]
        if low:
            ax.scatter(xy[low, 0], xy[low, 1], facecolors="none", edgecolors="#c1121f", s=35, linewidths=.8)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(f"{record['camera']} / view {record['view_angle']:g}°\nframes {record['first_frame']}–{record['last_frame']}", fontsize=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/rehab26"))
    parser.add_argument("--output", type=Path, default=Path("reports/rehab26_baseline/pose_galleries"))
    args = parser.parse_args()
    records, _ = load_repetitions(args.data_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    for person in sorted({r["person"] for r in records}):
        fig, axes = plt.subplots(3, 4, figsize=(15, 10), constrained_layout=True)
        for label in (0, 1, 2):
            candidates = [r for r in records if r["person"] == person and r["label"] == label]
            physical = sorted({r["physical_id"] for r in candidates}, key=lambda pid: min(r["first_frame"] for r in candidates if r["physical_id"] == pid))[:2]
            for repetition_index, physical_id in enumerate(physical):
                for camera_index, camera in enumerate(("cam1", "cam3")):
                    found = [r for r in candidates if r["physical_id"] == physical_id and r["camera"] == camera]
                    ax = axes[label, repetition_index * 2 + camera_index]
                    if found:
                        render(found[0], ax)
                    else:
                        ax.text(.5, .5, "Missing observation", ha="center")
                        ax.axis("off")
                    if repetition_index == camera_index == 0:
                        ax.text(-.05, .5, LABEL_NAMES[label], transform=ax.transAxes, rotation=90, va="center", fontsize=10)
        fig.suptitle(f"{person}: two repetitions per category, two synchronized camera views\nWithin each panel: start / estimated bottom / end. Red rings: visibility < 0.5. World XY projection, not RGB.", fontsize=12)
        fig.savefig(args.output / f"{person}.png", dpi=130)
        plt.close(fig)
    print(f"Галереи сохранены: {args.output}")


if __name__ == "__main__":
    main()
