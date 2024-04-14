from enum import IntFlag, auto
import os

from PIL import Image

from waves.utils.image_loader import ImageLoader
from waves.distortions import distortions

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
    REGEN_DIFF = auto()
    REGEN_VAE = auto()
    RINSE_2X_DIFF = auto()
    RINSE_4X_DIFF = auto()

    # Adversarial embedding attack
    ADV_EMB_RESNET_18 = auto()
    ADV_EMB_CLIP = auto()
    ADV_EMB_KLVAE_16 = auto()
    ADV_EMB_SDXL_VAE = auto()
    ADV_EMB_KLVAE_8 = auto()

    # Adversarial surrogate attack
    ADV_CLS_UNWM_WM = auto()
    ADV_CLS_REAL_WM = auto()
    ADV_CLS_WM1_WM2 = auto()

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

# TODO: Find a better way to get batch size.
DIST_BATCH_SIZE = 32

def _get_attacks(attacks: AttackMethods) -> list[AttackMethods]:
    assert attacks in AttackMethods.COMB_DIST_ALL

    return [ attack for attack in AttackMethods if attack.name.startswith('DIST_') and attack in attacks ]

def _is_distortion_attack(attack: AttackMethods):
    return attack in AttackMethods.COMB_DIST_ALL

def _save_images(imgs: list[Image.Image], dir: str, start_idx: int):
    for i, img in enumerate(imgs):
        img.save(os.path.join(dir, f'{start_idx + i:08d}.png'))

def _distortion_attack(wm_img_dir: str, out_dir: str, attack_methods: list[AttackMethods], attack_strength: float):
    if isinstance(attack_methods, AttackMethods):
        attack_methods = [attack_methods]

    image_loader = ImageLoader(wm_img_dir)
    img_idx = 0
    while image_loader.has_next():
        images = image_loader.load_batch(DIST_BATCH_SIZE)

        for attack in attack_methods:
            images = distortions.apply_distortion(images, attack_map[attack], attack_strength)

        _save_images(images, out_dir, img_idx)
        img_idx += len(images)

def attack(clean_img_dir: str, wm_img_dir: str, out_dir: str, attack_method: AttackMethods, attack_strength: float):
    # Restrict strength to [0.0, 1.0]
    attack_strength = min(1.0, max(0.0, attack_strength))

    if _is_distortion_attack(attack_method):
        _distortion_attack(wm_img_dir, out_dir, _get_attacks(attack_method), attack_strength)


