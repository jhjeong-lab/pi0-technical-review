from __future__ import annotations

import numpy as np

from _common import add_openpi_to_path, brief, env, missing_dependency


add_openpi_to_path()

try:
    from openpi.policies import policy_config  # noqa: E402
    from openpi.training import config  # noqa: E402
except ModuleNotFoundError as error:
    raise missing_dependency(error) from error


def droid_obs() -> dict:
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "pick up the fork",
    }


def libero_obs() -> dict:
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the object",
    }


config_name = env("PI0_CONFIG_NAME", "pi0_droid")
checkpoint = env("PI0_CHECKPOINT", "gs://openpi-assets/checkpoints/pi0_droid")
obs_mode = env("OBS_MODE", "droid")

obs = {"droid": droid_obs, "libero": libero_obs}[obs_mode]()
policy = policy_config.create_trained_policy(config.get_config(config_name), checkpoint)
result = policy.infer(obs)

print("config:", config_name)
print("checkpoint:", checkpoint)
print("obs_mode:", obs_mode)
print("result:", brief(result))
