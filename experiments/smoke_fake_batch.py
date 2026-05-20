from __future__ import annotations

from _common import add_openpi_to_path, brief, missing_dependency


add_openpi_to_path()

try:
    from openpi.training import config, data_loader  # noqa: E402
except ModuleNotFoundError as error:
    raise missing_dependency(error) from error


cfg = config.get_config("debug")
loader = data_loader.create_data_loader(
    cfg,
    num_batches=1,
    skip_norm_stats=True,
    framework="pytorch",
)

observation, actions = next(iter(loader))

print("config:", cfg.name)
print("observation:", brief(observation.to_dict()))
print("actions:", brief(actions))
