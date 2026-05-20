from __future__ import annotations

from _common import add_openpi_to_path, missing_dependency


add_openpi_to_path()

try:
    import openpi  # noqa: E402
    from openpi.training import config  # noqa: E402
except ModuleNotFoundError as error:
    raise missing_dependency(error) from error


cfg = config.get_config("pi0_libero")

print("openpi:", openpi.__file__)
print("config:", cfg.name)
print("model:", type(cfg.model).__name__)
print("data:", type(cfg.data).__name__)
print("batch_size:", cfg.batch_size)
print("num_train_steps:", cfg.num_train_steps)
