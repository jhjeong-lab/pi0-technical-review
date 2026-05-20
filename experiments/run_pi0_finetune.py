from __future__ import annotations

from _common import env, run_openpi


config_name = env("CONFIG_NAME", "pi0_libero_low_mem_finetune")
exp_name = env("EXP_NAME", "pi0_run")
steps = env("NUM_TRAIN_STEPS", "100")
save_interval = env("SAVE_INTERVAL", "100")

run_openpi(
    [
        "uv",
        "run",
        "scripts/train.py",
        config_name,
        f"--exp-name={exp_name}",
        f"--num-train-steps={steps}",
        f"--save-interval={save_interval}",
        "--overwrite",
        "--wandb-enabled=false",
    ]
)
