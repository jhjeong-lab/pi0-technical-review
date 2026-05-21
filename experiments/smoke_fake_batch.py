from __future__ import annotations

import dataclasses

from _common import add_openpi_to_path, brief, missing_dependency


add_openpi_to_path()

try:
    from openpi.training import config, data_loader  # noqa: E402
except ModuleNotFoundError as error:
    raise missing_dependency(error) from error



def main() -> None:
    cfg = dataclasses.replace(config.get_config("debug"), num_workers=0)
    loader = data_loader.create_data_loader(
        cfg,
        num_batches=1,
        skip_norm_stats=True,
        framework="pytorch",
    )

    observation, actions = next(iter(loader))

    print("config:", cfg.name)
    print("num_workers:", cfg.num_workers)
    print("observation:", brief(observation.to_dict()))
    print("actions:", brief(actions))


if __name__ == "__main__":
    main()
