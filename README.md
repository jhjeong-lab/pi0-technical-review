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

Clone OpenPI locally when needed:

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git upstream/openpi
```

Primary implementation targets:

```text
upstream/openpi/src/openpi/models_pytorch/
upstream/openpi/src/openpi/models_pytorch/pi0_pytorch.py
```

## Implementation Guide

- Keep runnable probes under `experiments/`.
- Keep each experiment small and reproducible.
- Do not commit local OpenPI checkout contents, model weights, datasets, logs, or generated artifacts.
- Put large runtime outputs under ignored paths such as `outputs/`, `runs/`, or `artifacts/`.
- Keep personal review guides under ignored `docs/` files.
