from enum import IntFlag, auto
import os
from typing import Optional
import json

import torch
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils.image_loader import ImageLoader, ImageData, NPCFImageData
from .distortions import distortions
from .regeneration.regen import RegenDiffusionAttacker, VAEAttacker
from .adversarial.embedding import AdversarialEmbeddingAttacker
from .adversarial.surrogate import AdversarialSurrogateAttacker
from .watermarker import Watermarker

class AttackMethods(IntFlag):
    # Distortion attacks
    DIST_ROTATION = auto()
    DIST_RESIZE_CROP = auto()
    DIST_ERASE = auto()
    DIST_BRIGHT = auto()
    DIST_CONTRAST = auto()
    DIST_BLUR = auto()
    DIST_NOISE = auto()
    DIST_JPEG = auto()
    COMB_DIST_GEOMETRIC = DIST_ROTATION | DIST_RESIZE_CROP | DIST_ERASE
    COMB_DIST_PHOTO = DIST_BRIGHT | DIST_CONTRAST
    COMB_DIST_DEGRAD = DIST_BLUR | DIST_NOISE | DIST_JPEG
    COMB_DIST_ALL = COMB_DIST_GEOMETRIC | COMB_DIST_PHOTO | COMB_DIST_DEGRAD

    # Regeneration attacks
    REGEN_VAE = auto()
    REGEN_DIFF = auto()
    RINSE_2X_DIFF = auto()
    RINSE_4X_DIFF = auto()

    # Adversarial embedding attack
    ADV_EMB_RESNET_18 = auto()
    ADV_EMB_CLIP = auto()
    ADV_EMB_KLVAE_16 = auto()
    ADV_EMB_SDXL_VAE = auto()
    ADV_EMB_KLVAE_8 = auto()

    # Adversarial surrogate attack
    ADV_SURROGATE = auto()

# TODO: This is a temporary solution to match the AttackMethod flag to strings in distortion.py.
# Fix it later.
attack_map = {
    AttackMethods.DIST_ROTATION: 'rotation',
    AttackMethods.DIST_RESIZE_CROP: 'resizedcrop',
    AttackMethods.DIST_ERASE: 'erasing',
    AttackMethods.DIST_BRIGHT: 'brightness',
    AttackMethods.DIST_CONTRAST: 'contrast',
    AttackMethods.DIST_BLUR: 'blurring',
    AttackMethods.DIST_NOISE: 'noise',
    AttackMethods.DIST_JPEG: 'compression',
}

repeats = {
    AttackMethods.REGEN_DIFF: 1,
    AttackMethods.RINSE_2X_DIFF: 2,
    AttackMethods.RINSE_4X_DIFF: 4,
}

adv_emb_map = {
    AttackMethods.ADV_EMB_RESNET_18: 'resnet18',
    AttackMethods.ADV_EMB_CLIP: 'clip',
    AttackMethods.ADV_EMB_KLVAE_8: 'klvae8',
    AttackMethods.ADV_EMB_SDXL_VAE: 'sdxlvae',
    AttackMethods.ADV_EMB_KLVAE_16: 'klvae16',
}

# TODO: Find a better way to get batch size.
DIST_BATCH_SIZE = 32

def _get_attacks(attacks: AttackMethods) -> list[AttackMethods]:
    assert attacks in AttackMethods.COMB_DIST_ALL

    return [ attack for attack in AttackMethods if attack.name.startswith('DIST_') and attack in attacks ]

def _is_distortion_attack(attack: AttackMethods):
    return attack in AttackMethods.COMB_DIST_ALL

def _is_regen_diffution_attack(attack: AttackMethods):
    return attack in AttackMethods.REGEN_DIFF | AttackMethods.RINSE_2X_DIFF | AttackMethods.RINSE_4X_DIFF

def _is_adversarial_emb_attack(attack: AttackMethods):
    return attack.name.startswith('ADV_EMB_')

# def _save_images(imgs: list[Image.Image], dir: str, start_idx: int):
#     for i, img in enumerate(imgs):
#         img.save(os.path.join(dir, f'{start_idx + i:08d}.png'))

def _distortion_attack(wm_img_dir: str, out_dir: str, attack_methods: list[AttackMethods], attack_strength: float, decoder: Watermarker):
    if isinstance(attack_methods, AttackMethods):
        attack_methods = [attack_methods]

    bits_diff = 0
    total_bits = 0

    # stegastamp data
    img_data = ImageData(wm_img_dir, ext='pt', n_image=100)

    # npcf data
    # img_data = NPCFImageData(wm_img_dir)

    loader = DataLoader(img_data, batch_size=32, shuffle=False, num_workers=4)
    bar = tqdm(total=len(img_data))
    for images, names, secrets in loader:
        images = images.clamp(0, 1)
        for attack in attack_methods:
            images = distortions.apply_distortion(images, attack_map[attack], attack_strength, return_image=False)

        result = decoder.decode(images)
        bits_diff += (torch.abs(secrets - result)).sum().item()
        total_bits += secrets.shape[0] * secrets.shape[1]

        # TODO: save image?
        # _save_images(out_dir, _tensor_to_pil(images), names)

        bar.update(len(images))

    print(f'{attack_methods} ber = {bits_diff}/{total_bits}')

def _regeneration_attack(wm_img_dir: str, out_dir: str, attack_method: AttackMethods, attack_strength: float, decoder: Watermarker, surrogate_model_path: str | None = None):

    # stegastamp data
    img_data = ImageData(wm_img_dir, ext='pt', n_image=100)

    # npcf data
    # img_data = NPCFImageData(wm_img_dir)

    loader = DataLoader(img_data, batch_size=2, shuffle=True)

    attacker = None
    if _is_regen_diffution_attack(attack_method):
        attacker = RegenDiffusionAttacker(attack_strength, repeats[attack_method])
    elif _is_adversarial_emb_attack(attack_method):
        attacker = AdversarialEmbeddingAttacker(adv_emb_map[attack_method], attack_strength)
    elif attack_method == AttackMethods.ADV_SURROGATE:

        if surrogate_model_path is None:
            raise ValueError('Missing surrogate model path')

        attacker = AdversarialSurrogateAttacker(surrogate_model_path, attack_strength, 400)
        # attacker.warmup(img_data)
    else:
        attacker = VAEAttacker(attack_strength)

    bits_diff = 0
    total_bits = 0

    bar = tqdm(total=len(img_data))
    for imgs, names, secrets in loader:
        imgs = imgs.clamp(0, 1)
        attacked_imgs = attacker.attack(imgs.float())
        result = decoder.decode(attacked_imgs) 

        bits_diff += (torch.abs(result - secrets)).sum().item()
        total_bits += secrets.shape[0] * secrets.shape[1]

        # TODO: save?
        _save_images(out_dir, _tensor_to_pil(attacked_imgs), names)

        bar.update(len(imgs))

    print(f'{attack_method} ber = {bits_diff}/{total_bits}')

def attack(
    wm_img_dir: str, 
    out_dir: str, 
    attack_method: AttackMethods, 
    attack_strength: float, 
    decoder: Watermarker,
    surrogate_model_path: str | None = None
):
    if _is_distortion_attack(attack_method):
        _distortion_attack(wm_img_dir, out_dir, _get_attacks(attack_method), attack_strength, decoder)

    else:
        _regeneration_attack(wm_img_dir, out_dir, attack_method, attack_strength, decoder, surrogate_model_path)

def _tensor_to_pil(imgs: torch.Tensor) -> list[Image.Image]:
    np_imgs = np.round(imgs.permute(0, 2, 3, 1).numpy() * 255.).astype(np.uint8)
    return [ Image.fromarray(img) for img in np_imgs ]

def _save_images(path: str, images: list[Image.Image], images_names):
    for img, name in zip(images, images_names):
        img.save(os.path.join(path, f'{name}.png'))

