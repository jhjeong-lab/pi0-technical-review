# pi0 Technical Review

Workspace for inspecting and experimenting with pi0/OpenPI implementation details.

## Structure

```text
experiments/
  Small reproducible probes for imports, model execution, data inspection,
  and shape observation.

upstream/
  .gitkeep
  openpi/        # ignored local checkout of Physical-Intelligence/openpi
```

## Upstream

From the repository root, clone OpenPI under `upstream/openpi`:

```bash
cd pi0-technical-review
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git upstream/openpi
```

Primary implementation targets:

```text
upstream/openpi/src/openpi/models_pytorch/
upstream/openpi/src/openpi/models_pytorch/pi0_pytorch.py
```

## Dataset

Use LIBERO through the LeRobot dataset loader:

```text
physical-intelligence/libero
```

The OpenPI `pi0_libero` training config already points to this dataset.

## Experiments

Install OpenPI dependencies:

```bash
cd upstream/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cd ../..
```

Run compact checks from the repository root:

```bash
python experiments/smoke_env.py
upstream/openpi/.venv/bin/python experiments/smoke_openpi.py
upstream/openpi/.venv/bin/python experiments/smoke_fake_batch.py
upstream/openpi/.venv/bin/python experiments/inspect_libero.py
upstream/openpi/.venv/bin/python experiments/smoke_pi0_policy.py
```

## Implementation Guide

- Keep runnable probes under `experiments/`.
- Keep each experiment small and reproducible.
- Use `pi0` configs and checkpoints for experiments.
- Do not commit local OpenPI checkout contents, model weights, datasets, logs, or generated artifacts.
- Put large runtime outputs under ignored paths such as `outputs/`, `runs/`, or `artifacts/`.
- Keep personal review guides under ignored `docs/` files.
