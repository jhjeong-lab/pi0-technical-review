from __future__ import annotations

import json
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image

from _common import add_openpi_to_path, brief, env, missing_dependency, tree_brief


add_openpi_to_path()

try:
    from huggingface_hub import snapshot_download
    from openpi.training import config
    from openpi import transforms
except ModuleNotFoundError as error:
    raise missing_dependency(error) from error


def decode_image(value) -> np.ndarray:
    if isinstance(value, dict) and "bytes" in value:
        return np.asarray(Image.open(BytesIO(value["bytes"])).convert("RGB"))
    return np.asarray(value)


def load_tasks(local_dir: Path) -> dict[int, str]:
    candidates = [local_dir / "tasks.jsonl", local_dir / "meta" / "tasks.jsonl"]
    tasks_path = next((path for path in candidates if path.exists()), None)
    if tasks_path is None:
        matches = list(local_dir.rglob("tasks.jsonl"))
        tasks_path = matches[0] if matches else None
    if tasks_path is None:
        return {}

    tasks = {}
    with tasks_path.open() as file:
        for line in file:
            item = json.loads(line)
            task_id = item.get("task_index", item.get("index", item.get("id")))
            task = item.get("task", item.get("name", item.get("text")))
            if task_id is not None and task is not None:
                tasks[int(task_id)] = task
    return tasks


repo_id = env("LEROBOT_REPO_ID", "physical-intelligence/libero")
episode_file = env("LEROBOT_EPISODE_FILE", "data/chunk-000/episode_000000.parquet")
sample_index = int(env("SAMPLE_INDEX", "0"))

local_dir = Path(
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["episodes.jsonl", "tasks.jsonl", "stats.json", "info.json", "meta/**", episode_file],
    )
)

frame = pd.read_parquet(local_dir / episode_file)
row = frame.iloc[sample_index].to_dict()
tasks = load_tasks(local_dir)
prompt = tasks.get(int(row.get("task_index", 0)), "do something")

raw = {
    "image": decode_image(row["image"]),
    "wrist_image": decode_image(row["wrist_image"]),
    "state": np.asarray(row["state"], dtype=np.float32),
    "actions": np.repeat(np.asarray(row["actions"], dtype=np.float32)[None], 10, axis=0),
    "task_index": row.get("task_index"),
    "prompt": prompt,
}

cfg = config.get_config("pi0_libero")
data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
norm_stats = data_cfg.norm_stats

steps = [
    ("raw parquet row normalized for OpenPI", lambda x: x),
    ("repack_transforms", transforms.compose(data_cfg.repack_transforms.inputs)),
    ("data_transforms / LiberoInputs", transforms.compose(data_cfg.data_transforms.inputs)),
    ("Normalize", transforms.Normalize(norm_stats, use_quantiles=data_cfg.use_quantile_norm)),
    ("model_transforms / tokenize + pad", transforms.compose(data_cfg.model_transforms.inputs)),
]

data = raw
print("repo_id:", repo_id)
print("episode_file:", episode_file)
print("sample_index:", sample_index)
print("prompt:", prompt)
print("norm_stats_loaded:", norm_stats is not None)

for name, step in steps:
    data = step(data)
    tree_brief(name, data)

print("\nFinal model input keys:", list(data.keys()))
print("Final actions:", brief(data.get("actions")))
