from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _common import brief, env, missing_dependency

try:
    from huggingface_hub import snapshot_download
except ModuleNotFoundError as error:
    raise missing_dependency(error) from error


repo_id = env("LEROBOT_REPO_ID", "physical-intelligence/libero")
episode_files = [
    item.strip()
    for item in env(
        "LEROBOT_EPISODE_FILES",
        "data/chunk-000/episode_000000.parquet,data/chunk-000/episode_000001.parquet",
    ).split(",")
    if item.strip()
]

local_dir = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=[
        "episodes.jsonl",
        "tasks.jsonl",
        "stats.json",
        "info.json",
        "meta/**",
        *episode_files,
    ],
)

states = []
actions = []

for episode_file in episode_files:
    frame = pd.read_parquet(Path(local_dir) / episode_file)
    print("episode_file:", episode_file)
    print("rows:", len(frame))
    print("columns:", list(frame.columns))
    if not frame.empty:
        print("sample:", brief(frame.iloc[0].to_dict()))
    states.extend(frame["observation.state"].to_list() if "observation.state" in frame else frame["state"].to_list())
    actions.extend(frame["action"].to_list() if "action" in frame else frame["actions"].to_list())

states_np = np.asarray(states, dtype=np.float32)
actions_np = np.asarray(actions, dtype=np.float32)

print("state:", brief(states_np), "mean=", states_np.mean(axis=0), "std=", states_np.std(axis=0))
print("actions:", brief(actions_np), "mean=", actions_np.mean(axis=0), "std=", actions_np.std(axis=0))
