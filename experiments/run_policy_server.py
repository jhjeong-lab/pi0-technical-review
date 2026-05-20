from __future__ import annotations

from _common import env, run_openpi


config_name = env("PI0_POLICY_CONFIG", "pi0_libero_low_mem_finetune")
checkpoint = env("PI0_POLICY_CHECKPOINT", "checkpoints/pi0_libero_low_mem_finetune/pi0_run/29999")
port = env("PORT", "8000")

run_openpi(
    [
        "uv",
        "run",
        "scripts/serve_policy.py",
        f"--port={port}",
        "policy:checkpoint",
        f"--policy.config={config_name}",
        f"--policy.dir={checkpoint}",
    ]
)
