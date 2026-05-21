from __future__ import annotations

from io import BytesIO
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

from _common import add_openpi_to_path, brief, env, missing_dependency, torch_tree_to, tree_brief


add_openpi_to_path()

try:
    from huggingface_hub import snapshot_download
    from openpi import transforms
    from openpi.models.model import Observation
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
    from openpi.training import config
except ModuleNotFoundError as error:
    raise missing_dependency(error) from error
except ValueError as error:
    raise SystemExit(
        f"{error}\n"
        "If this mentions transformers_replace, run:\n"
        "  cd upstream/openpi\n"
        "  cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/"
    ) from error


def decode_image(value) -> np.ndarray:
    if isinstance(value, dict) and "bytes" in value:
        return np.asarray(Image.open(BytesIO(value["bytes"])).convert("RGB"))
    return np.asarray(value)


def load_tasks(local_dir: Path) -> dict[int, str]:
    candidates = [local_dir / "tasks.jsonl", local_dir / "meta" / "tasks.jsonl"]
    tasks_path = next((path for path in candidates if path.exists()), None)
    if tasks_path is None:
        matches = list(local_dir.rglob("tasks.jsonl"))
        tasks_path = matches[0] if matches else None
    if tasks_path is None:
        return {}

    tasks = {}
    with tasks_path.open() as file:
        for line in file:
            item = json.loads(line)
            task_id = item.get("task_index", item.get("index", item.get("id")))
            task = item.get("task", item.get("name", item.get("text")))
            if task_id is not None and task is not None:
                tasks[int(task_id)] = task
    return tasks


def load_libero_sample(repo_id: str, episode_file: str, sample_index: int) -> dict:
    local_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["episodes.jsonl", "tasks.jsonl", "stats.json", "info.json", "meta/**", episode_file],
        )
    )
    frame = pd.read_parquet(local_dir / episode_file)
    row = frame.iloc[sample_index].to_dict()
    tasks = load_tasks(local_dir)
    prompt = tasks.get(int(row.get("task_index", 0)), "do something")

    return {
        "image": decode_image(row["image"]),
        "wrist_image": decode_image(row["wrist_image"]),
        "state": np.asarray(row["state"], dtype=np.float32),
        "actions": np.repeat(np.asarray(row["actions"], dtype=np.float32)[None], 10, axis=0),
        "task_index": row.get("task_index"),
        "prompt": prompt,
    }


def apply_openpi_transforms(raw: dict, transform_config_name: str) -> dict:
    train_cfg = config.get_config(transform_config_name)
    data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    norm_stats = data_cfg.norm_stats

    steps = [
        ("raw LIBERO sample", lambda x: x),
        ("repack_transforms", transforms.compose(data_cfg.repack_transforms.inputs)),
        ("data_transforms", transforms.compose(data_cfg.data_transforms.inputs)),
        ("Normalize", transforms.Normalize(norm_stats, use_quantiles=data_cfg.use_quantile_norm)),
        ("model_transforms", transforms.compose(data_cfg.model_transforms.inputs)),
    ]

    data = raw
    print("transform_config:", transform_config_name)
    print("norm_stats_loaded:", norm_stats is not None)
    for name, step in steps:
        data = step(data)
        tree_brief(name, data)
    return data


def batch_to_torch(data: dict, device: torch.device) -> tuple[Observation, torch.Tensor]:
    obs_dict = {}
    obs_dict["image"] = {
        key: torch.as_tensor(value, dtype=torch.uint8).unsqueeze(0) for key, value in data["image"].items()
    }
    obs_dict["image_mask"] = {
        key: torch.as_tensor([value], dtype=torch.bool) for key, value in data["image_mask"].items()
    }
    obs_dict["state"] = torch.as_tensor(data["state"], dtype=torch.float32).unsqueeze(0)
    obs_dict["tokenized_prompt"] = torch.as_tensor(data["tokenized_prompt"], dtype=torch.int32).unsqueeze(0)
    obs_dict["tokenized_prompt_mask"] = torch.as_tensor(data["tokenized_prompt_mask"], dtype=torch.bool).unsqueeze(0)

    actions = torch.as_tensor(data["actions"], dtype=torch.float32).unsqueeze(0)
    obs_dict = torch_tree_to(obs_dict, device)
    actions = actions.to(device)
    return Observation.from_dict(obs_dict), actions


def default_device() -> torch.device:
    selected = "cuda" if torch.cuda.is_available() else "cpu"
    if selected == "cuda":
        major, _minor = torch.cuda.get_device_capability()
        if major >= 12:
            selected = "cpu"
    return torch.device(env("DEVICE", selected))


def make_model_config(model_config_name: str, actions: torch.Tensor, observation: Observation, device: torch.device):
    train_cfg = config.get_config(model_config_name)
    model_cfg = dataclasses.replace(
        train_cfg.model,
        action_horizon=actions.shape[1],
        action_dim=actions.shape[2],
        max_token_len=observation.tokenized_prompt.shape[1],
        pytorch_compile_mode=None,
    )
    if device.type == "cpu":
        model_cfg = dataclasses.replace(model_cfg, dtype="float32")
    return model_cfg


def channels_first_for_embed(images):
    converted = []
    result = []
    for index, image in enumerate(images):
        if image.ndim == 4 and image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2).contiguous()
            converted.append(index)
        result.append(image)
    return result, converted


def trace_embed_prefix_parts(model, images, img_masks, lang_tokens, lang_masks):
    embs = []
    pad_masks = []
    att_masks = []

    for index, (image, img_mask) in enumerate(zip(images, img_masks, strict=True)):
        img_emb = model.paligemma_with_expert.embed_image(image)
        tree_brief(f"embed_prefix.image[{index}].emb", img_emb)

        bsize, num_img_embs = img_emb.shape[:2]
        embs.append(img_emb)
        pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
        att_masks += [0] * num_img_embs

    lang_emb = model.paligemma_with_expert.embed_language_tokens(lang_tokens)
    lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
    tree_brief("embed_prefix.lang.emb", lang_emb)

    embs.append(lang_emb)
    pad_masks.append(lang_masks)
    att_masks += [0] * lang_emb.shape[1]

    embed_dims = [emb.shape[-1] for emb in embs]
    print("\nembed_prefix.concat_dims:", embed_dims)
    if len(set(embed_dims)) != 1:
        print("embed_prefix.skipped:", "cannot concatenate image/language embeddings with different hidden sizes")
        return None

    pad_masks = torch.cat(pad_masks, dim=1)
    att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
    att_masks = att_masks[None, :].expand(pad_masks.shape[0], len(att_masks))
    return torch.cat(embs, dim=1), pad_masks, att_masks


def main() -> None:
    repo_id = env("LEROBOT_REPO_ID", "physical-intelligence/libero")
    episode_file = env("LEROBOT_EPISODE_FILE", "data/chunk-000/episode_000000.parquet")
    sample_index = int(env("SAMPLE_INDEX", "0"))
    transform_config = env("TRANSFORM_CONFIG", "pi0_libero")
    model_config = env("MODEL_CONFIG", "debug")
    num_steps = int(env("NUM_DENOISE_STEPS", "2"))
    device = default_device()

    print("repo_id:", repo_id)
    print("episode_file:", episode_file)
    print("sample_index:", sample_index)
    print("device:", device)
    print("model_config:", model_config)

    raw = load_libero_sample(repo_id, episode_file, sample_index)
    data = apply_openpi_transforms(raw, transform_config)
    observation, actions = batch_to_torch(data, device)

    tree_brief("batched Observation", observation.to_dict())
    tree_brief("batched actions", actions)

    model_cfg = make_model_config(model_config, actions, observation, device)
    model = PI0Pytorch(model_cfg).to(device).eval()
    print("pi0_pytorch_config:", model_cfg)

    with torch.no_grad():
        images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(observation, train=False)
        tree_brief("_preprocess_observation.images", images)
        tree_brief("_preprocess_observation.img_masks", img_masks)
        tree_brief("_preprocess_observation.lang_tokens", lang_tokens)
        tree_brief("_preprocess_observation.lang_masks", lang_masks)
        tree_brief("_preprocess_observation.state", state)

        noise = model.sample_noise(actions.shape, device)
        time = model.sample_time(actions.shape[0], device)
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        tree_brief("sample_noise", noise)
        tree_brief("sample_time", time)
        tree_brief("x_t", x_t)

        images, converted_images = channels_first_for_embed(images)
        if converted_images:
            print("\nimage_format:", f"converted NHWC -> NCHW for image indexes {converted_images}")
            tree_brief("embed_prefix.images", images)

        prefix = trace_embed_prefix_parts(model, images, img_masks, lang_tokens, lang_masks)
        if prefix is not None:
            prefix_embs, prefix_pad_masks, prefix_att_masks = prefix
            tree_brief("embed_prefix.embs", prefix_embs)
            tree_brief("embed_prefix.pad_masks", prefix_pad_masks)
            tree_brief("embed_prefix.att_masks", prefix_att_masks)
        else:
            prefix_embs = prefix_pad_masks = prefix_att_masks = None

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(state, x_t, time)
        tree_brief("embed_suffix.embs", suffix_embs)
        tree_brief("embed_suffix.pad_masks", suffix_pad_masks)
        tree_brief("embed_suffix.att_masks", suffix_att_masks)
        tree_brief("embed_suffix.adarms_cond", adarms_cond)

        if prefix is None:
            print("\nforward.skipped: full forward requires compatible prefix image/language hidden sizes")
            return

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        att_4d_masks = model._prepare_attention_masks_4d(att_2d_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        tree_brief("combined pad_masks", pad_masks)
        tree_brief("combined att_masks", att_masks)
        tree_brief("make_att_2d_masks", att_2d_masks)
        tree_brief("_prepare_attention_masks_4d", att_4d_masks)
        tree_brief("position_ids", position_ids)

        try:
            loss = model.forward(observation, actions, noise=noise, time=time)
            tree_brief("forward loss per action dim", loss)
            sampled = model.sample_actions(device, observation, num_steps=num_steps)
            tree_brief(f"sample_actions(num_steps={num_steps})", sampled)
        except RuntimeError as error:
            print("\nforward.failed:", error)


if __name__ == "__main__":
    main()
