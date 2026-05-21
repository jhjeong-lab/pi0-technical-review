from __future__ import annotations

from _common import env, run_openpi


config_name = env("CONFIG_NAME", "pi0_libero_low_mem_finetune")
max_frames = env("MAX_FRAMES", "2000")
allow_full_download = env("ALLOW_FULL_LIBERO_DOWNLOAD", "0") == "1"

if not allow_full_download:
    raise SystemExit(
        "Refusing to run OpenPI compute_norm_stats by default.\n"
        "OpenPI's LeRobotDataset construction may download the full LIBERO dataset before MAX_FRAMES is applied.\n"
        "Use experiments/compute_libero_mini_stats.py for a small raw-data check, or rerun with\n"
        "ALLOW_FULL_LIBERO_DOWNLOAD=1 only when the pod has enough disk for the full dataset."
    )

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
