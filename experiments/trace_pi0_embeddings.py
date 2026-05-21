from __future__ import annotations

import dataclasses

import torch

from _common import add_openpi_to_path, brief, env, missing_dependency, torch_tree_to, tree_brief


add_openpi_to_path()

try:
    from openpi.models.model import Observation
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
    from openpi.training import config, data_loader
except ModuleNotFoundError as error:
    raise missing_dependency(error) from error
except ValueError as error:
    raise SystemExit(
        f"{error}\n"
        "If this mentions transformers_replace, run:\n"
        "  cd upstream/openpi\n"
        "  cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/"
    ) from error


default_device = "cuda" if torch.cuda.is_available() else "cpu"
if default_device == "cuda":
    major, minor = torch.cuda.get_device_capability()
    if major >= 12:
        default_device = "cpu"
device = torch.device(env("DEVICE", default_device))
num_steps = int(env("NUM_DENOISE_STEPS", "2"))

cfg = dataclasses.replace(config.get_config("debug"), num_workers=0)
model_cfg = dataclasses.replace(cfg.model, pytorch_compile_mode=None)
cfg = dataclasses.replace(cfg, model=model_cfg)

loader = data_loader.create_data_loader(
    cfg,
    num_batches=1,
    skip_norm_stats=True,
    framework="pytorch",
)
observation, actions = next(iter(loader))

obs_dict = torch_tree_to(observation.to_dict(), device)
observation = Observation.from_dict(obs_dict)
actions = actions.to(device)

model = PI0Pytorch(model_cfg).to(device).eval()

print("device:", device)
print("config:", cfg.name)
print("model_config:", model_cfg)
tree_brief("loader observation", observation.to_dict())
tree_brief("loader actions", actions)

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
    tree_brief("x_t = time * noise + (1 - time) * actions", x_t)

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    tree_brief("embed_prefix.embs", prefix_embs)
    tree_brief("embed_prefix.pad_masks", prefix_pad_masks)
    tree_brief("embed_prefix.att_masks", prefix_att_masks)

    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(state, x_t, time)
    tree_brief("embed_suffix.embs", suffix_embs)
    tree_brief("embed_suffix.pad_masks", suffix_pad_masks)
    tree_brief("embed_suffix.att_masks", suffix_att_masks)
    tree_brief("embed_suffix.adarms_cond", adarms_cond)

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

    loss = model.forward(observation, actions, noise=noise, time=time)
    tree_brief("forward loss per action dim", loss)

    sampled = model.sample_actions(device, observation, num_steps=num_steps)
    tree_brief(f"sample_actions(num_steps={num_steps})", sampled)
