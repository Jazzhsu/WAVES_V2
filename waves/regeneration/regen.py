from PIL import Image
import numpy as np
import torch
from skimage.util import random_noise
from torchvision.transforms import functional as F
from tqdm import tqdm
from compressai.zoo import (
    bmshj2018_factorized,
    bmshj2018_hyperprior,
    mbt2018_mean,
    mbt2018,
    cheng2020_anchor,
)
import numpy as np

from waves.regeneration.pipeline_re_sd import ReSDPipeline

"""
Credit to: https://github.com/XuandongZhao/WatermarkAttacker
"""

class WMAttacker:
    def attack(self, images: list[Image.Image]):
        raise NotImplementedError

QUALITY = [8, 7, 6, 5, 4, 3, 2, 1]

class VAEAttacker(WMAttacker):
    def __init__(self, strength: float, model_name='bmshj2018-factorized', device="cuda"):
        quality_idx = min(int(strength * len(QUALITY)), len(QUALITY) - 1)
        quality = QUALITY[quality_idx]

        if model_name == "bmshj2018-factorized":
            self._model = (
                bmshj2018_factorized(quality=quality, pretrained=True)
                .eval()
                .to(device)
            )
        elif model_name == "bmshj2018-hyperprior":
            self._model = (
                bmshj2018_hyperprior(quality=quality, pretrained=True)
                .eval()
                .to(device)
            )
        elif model_name == "mbt2018-mean":
            self._model = (
                mbt2018_mean(quality=quality, pretrained=True).eval().to(device)
            )
        elif model_name == "mbt2018":
            self._model = mbt2018(quality=quality, pretrained=True).eval().to(device)
        elif model_name == "cheng2020-anchor":
            self._model = (
                cheng2020_anchor(quality=quality, pretrained=True).eval().to(device)
            )
        else:
            raise ValueError("model name not supported")
        self._device = device

    def attack(self, images: list[Image.Image]):
        images = [ F.to_tensor(im.convert('RGB').resize((512, 512))) for im in images ]
        images = torch.stack(images).to(self._device)
        out = self._model(images)
        out["x_hat"].clamp_(0, 1)
        out = [ F.to_pil_image(im) for im in out["x_hat"].cpu()]
        return out

REGEN_DIFF_NOISE = [20, 40, 60, 80, 100]

class RegenDiffusionAttacker(WMAttacker):
    def __init__(self, strength: float, repeat: int, device=torch.device('cuda')):
        self._pipe = ReSDPipeline.from_pretrained(
            'CompVis/stable-diffusion-v1-4',
            torch_dtype=torch.float16,
            variant='fp16'
        ).to(device)

        self._device = device
        self._noise_step = REGEN_DIFF_NOISE[min(int(strength * 5), 4)]
        self._repeat = repeat

        print(f"Diffuse attack initialized with noise step {self._noise_step}")

    @torch.no_grad()
    def attack(self, images: list[Image.Image]):
        generator = torch.Generator(self._device).manual_seed(1024) # subject to remove

        timestep = torch.tensor([self._noise_step], dtype=torch.long, device=self._device)
        images = np.asarray([ np.asarray(image) for image in images ]) / 255.

        batch_size = images.shape[0]
        prompts = [""] * batch_size

        for _ in range(self._repeat):
            # TODO: optimize the following data transformation. move out of this loop.
            images = (images - 0.5) * 2.
            # (B, H, W, C) -> (B, C, H, W)
            images = torch.tensor(images, dtype=torch.float16, device=self._device).permute(0, 3, 1, 2)

            latents = self._pipe.vae.encode(images).latent_dist
            latents = latents.sample(generator) * self._pipe.vae.config.scaling_factor
            noise = torch.randn(
                (batch_size, 4, images.shape[-2] // 8, images.shape[-1] // 8), device=self._device
            )

            latents = self._pipe.scheduler.add_noise(latents, noise, timestep).type(torch.half)
            images = self._pipe(
                prompts,
                head_start_latents = latents,
                head_start_step = 50 - max(self._noise_step // 20, 1),
                guidance_scale=7.5,
                generator=generator,
                output_type=None,
            )
        return self._pipe.numpy_to_pil(images)
