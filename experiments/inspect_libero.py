from __future__ import annotations

from _common import add_openpi_to_path, brief, env, missing_dependency


add_openpi_to_path()

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
except ModuleNotFoundError as error:
    raise missing_dependency(error) from error


repo_id = env("LEROBOT_REPO_ID", "physical-intelligence/libero")
index = int(env("SAMPLE_INDEX", "0"))

dataset = LeRobotDataset(repo_id)
sample = dataset[index]

print("repo_id:", repo_id)
print("length:", len(dataset))
print("sample_index:", index)
print("sample:", brief(sample))
