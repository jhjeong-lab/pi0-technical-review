from __future__ import annotations

from _common import env, run_openpi


host = env("POLICY_HOST", "0.0.0.0")
port = env("PORT", "8000")
env_name = env("CLIENT_ENV", "LIBERO")
steps = env("NUM_STEPS", "5")

run_openpi(
    [
        "uv",
        "run",
        "examples/simple_client/main.py",
        f"--host={host}",
        f"--port={port}",
        f"--env={env_name}",
        f"--num-steps={steps}",
    ]
)
