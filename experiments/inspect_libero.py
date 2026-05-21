from __future__ import annotations

from pathlib import Path

import pandas as pd

from _common import brief, env, missing_dependency

try:
    from huggingface_hub import snapshot_download
except ModuleNotFoundError as error:
    raise missing_dependency(error) from error


repo_id = env("LEROBOT_REPO_ID", "physical-intelligence/libero")
episode_file = env("LEROBOT_EPISODE_FILE", "data/chunk-000/episode_000000.parquet")
cache_dir = env("HF_CACHE_DIR", "")

allow_patterns = [
    "README.md",
    ".gitattributes",
    "episodes.jsonl",
    "tasks.jsonl",
    "stats.json",
    "info.json",
    "meta/**",
    episode_file,
]

local_dir = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=allow_patterns,
    cache_dir=cache_dir or None,
)

episode_path = Path(local_dir) / episode_file
frame = pd.read_parquet(episode_path)
sample = frame.iloc[0].to_dict()

print("repo_id:", repo_id)
print("local_dir:", local_dir)
print("episode_file:", episode_file)
print("episode_rows:", len(frame))
print("columns:", list(frame.columns))
print("sample:", brief(sample))
