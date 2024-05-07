import os
import json

import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm

from .watermarker import Watermarker, WatermarkerType
from .utils.image_loader import ImageLoader, ImageData
from . import attacks

WAVE_CACHE_DIR = 'waves/'

class WavesBenchmark:

    def __init__(self, watermarker: Watermarker, image_src: str, cache_folder: str = '.cache/'):
        self._watermarker = watermarker

        # Cache folders
        self._cache_folder = cache_folder
        self._image_src = image_src

        self._waves_cache_folder = os.path.join(cache_folder, WAVE_CACHE_DIR)
        self._clean_img_dir = os.path.join(self._waves_cache_folder, 'clean/')
        self._wm_img_dir = os.path.join(self._waves_cache_folder, 'wm/')
        self._attack_img_dir = os.path.join(self._waves_cache_folder, 'attack/')
        self._adv_attck_training_dir = os.path.join(self._waves_cache_folder, 'adv_train/')

        # Create necessary cache directories.
        self._init_cache_folders()

    def generate_wm_images(self, n_images: int | None = None):
        if self._watermarker.type == WatermarkerType.POST_PROCESS:
            self._generate_post_process_wm(n_images)

    def generate_attack_images(self, attack: attacks.AttackMethods, strength: float, surrogate_model_path: str = None):
        attacks.attack(self._wm_img_dir, self._attack_img_dir, attack, strength, self._watermarker, surrogate_model_path)

    def _generate_attacks_from_profile(self, attack_profile: dict, surrogate_mode_paths: list = None):
        # A list of tuples, where each item is (AttackMethods, strength, additional_info)
        attack_list = []
        for attack_name, strength_list in attack_profile:
            for strength in strength_list:
                if attack_name != attacks.AttackMethods.ADV_SURROGATE:
                    attack_list.append((attack_name, strength, None))
                    continue

                if surrogate_mode_paths == None or len(surrogate_mode_paths) == 0:
                    raise ValueError(f'Attack profiles contains surrogate attack, but surrogate model path is not provided')

                for model_path in surrogate_mode_paths:
                    attack_list.append((attack_name, strength, model_path))

        return attack_list 

    def _generate_post_process_wm(self, n_images: int | None = None):
        image_data = ImageData(self._image_src, ext='png', n_image=n_images)
        loader = DataLoader(image_data, batch_size=self._watermarker.batch_size, shuffle=False)
        message_dict = {}

        bar = tqdm(total=len(image_data))
        for imgs, img_names in loader:
            for i in range(8):
                names = [ f'{n}_{i}' for n in img_names ]
                messages = self._watermarker.generate_messages(len(img_names))
                _, wm_imgs = self._watermarker.encode(imgs, prompts=[], messages=messages)

                # self._save_images(self._wm_img_dir, self._tensor_to_pil(wm_imgs), names)
                self._save_tensor(self._wm_img_dir, wm_imgs, names)

                message_dict.update(dict(zip(names, messages.tolist())))
            bar.update(len(imgs))

        with open(os.path.join(self._waves_cache_folder, 'secret.json'), 'w', encoding='utf-8') as f:
            json.dump(message_dict, f)

    def _tensor_to_pil(self, imgs: torch.Tensor) -> list[Image.Image]:
        np_imgs = np.round(imgs.permute(0, 2, 3, 1).numpy() * 255.).astype(np.uint8)
        return [ Image.fromarray(img) for img in np_imgs ]

    def _save_images(self, path: str, images: list[Image.Image], images_names):
        for img, name in zip(images, images_names):
            img.save(os.path.join(path, f'{name}.png'))

    def _save_tensor(self, path: str, images: torch.Tensor, images_names):
        for img, img_name in zip(images, images_names):
            torch.save(img.clone(), os.path.join(path, f'{img_name}.pt'))

    def _init_cache_folders(self):
        if not os.path.exists(self._clean_img_dir):
            os.makedirs(self._clean_img_dir)

        if not os.path.exists(self._wm_img_dir):
            os.makedirs(self._wm_img_dir)

        if not os.path.exists(self._attack_img_dir):
            os.makedirs(self._attack_img_dir)


