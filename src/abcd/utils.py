from pathlib import Path
from re import search

import numpy as np
import polars as pl


def cleanup_checkpoints(checkpoint_dir, mode="min"):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    metrics = [
        (float(match.group(1)), checkpoint)
        for checkpoint in checkpoints
        if (match := search(r"(\d+\.\d+)(?=[^\d]|$)", checkpoint.name))
    ]
    metrics.sort(key=lambda x: x[0], reverse=(mode == "max"))
    for _, checkpoint in metrics[1:]:
        checkpoint.unlink()


def get_best_checkpoint(ckpt_folder: Path, mode: str) -> Path:
    checkpoint_paths = list(ckpt_folder.glob("*"))
    metrics = [
        float(match.group(1))
        for filepath in checkpoint_paths
        if (match := search(r"(\d+\.\d+)(?=[^\d]|$)", filepath.stem))
    ]
    if mode == "min":
        index = np.argmin(metrics)
    else:
        index = np.argmax(metrics)
    return checkpoint_paths[index]


EVENTS = [
    "baseline_year_1_arm_1",
    "1_year_follow_up_y_arm_1",
    "2_year_follow_up_y_arm_1",
    "3_year_follow_up_y_arm_1",
    "4_year_follow_up_y_arm_1",
]
EVENT_INDEX = list(range(len(EVENTS)))
EVENT_MAPPING = dict(zip(EVENTS, EVENT_INDEX))
GROUP_ORDER = pl.Enum(
    [
        "Conversion",
        "Persistence",
        "Agnostic",
        "1",
        "2",
        "3",
        "4",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "2016",
        "2017",
        "2018",
        "2019",
        "2020",
        "2021",
        "Baseline",
        "1-year",
        "2-year",
        "3-year",
        "Asian",
        "Black",
        "Hispanic",
        "White",
        "Other",
        "Female",
        "Male",
    ]
)
RISK_GROUPS = {
    "1": "No risk",
    "2": "Low risk",
    "3": "Moderate risk",
    "4": "High risk",
}
