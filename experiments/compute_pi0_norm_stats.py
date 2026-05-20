from __future__ import annotations

from _common import env, run_openpi


config_name = env("CONFIG_NAME", "pi0_libero_low_mem_finetune")
max_frames = env("MAX_FRAMES", "2000")

run_openpi(
    [
        "uv",
        "run",
        "scripts/compute_norm_stats.py",
        "--config-name",
        config_name,
        "--max-frames",
        max_frames,
    ]
)
