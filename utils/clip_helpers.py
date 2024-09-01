from functools import partial
from torchmetrics.functional.multimodal import clip_score
import torch

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")


def get_clip_score(images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score_val = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score_val), 4)
