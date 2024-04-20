import os

from PIL import Image

from .watermarker import Watermarker, WatermarkerType
from .utils.image_loader import ImageLoader
from . import attacks

WAVE_CACHE_DIR = 'waves/'

class WavesBenchmark:

    def __init__(self, watermarker: Watermarker, image_src: str, cache_folder: str = '.cache/'):
        self._watermarker = watermarker

        # Cache folders
        self._cache_folder = cache_folder
        self._image_src = image_src

        img_cache_folder = os.path.join(cache_folder, WAVE_CACHE_DIR)
        self._clean_img_dir = os.path.join(img_cache_folder, 'clean/')
        self._wm_img_dir = os.path.join(img_cache_folder, 'wm/')
        self._attack_img_dir = os.path.join(img_cache_folder, 'attack/')
        self._adv_attck_training_dir = os.path.join(img_cache_folder, 'adv_train/')

        # Create necessary cache directories.
        self._init_cache_folders()

    def generate_wm_images(self, n_images: int = None):
        if self._watermarker.type == WatermarkerType.POST_PROCESS:
            self._generate_post_process_wm(n_images)
        else:
            self._generate_in_process_wm(n_images)

    def generate_attack_images(self, attack: attacks.AttackMethods, strength: float, surrogate_model_path: str = None):
        attacks.attack(self._wm_img_dir, self._attack_img_dir, attack, strength, surrogate_model_path)

    def _generate_post_process_wm(self, n_images: int = None):
        img_loader = ImageLoader(self._image_src, n_images=n_images)
        image_idx = 0
        while img_loader.has_next():
            imgs = img_loader.load_batch(self._watermarker.batch_size)

            messages = self._watermarker.generate_messages(self._watermarker.batch_size)
            clean_imgs, wm_imgs = self._watermarker.encode(imgs, prompts=[], messages=messages)

            if clean_imgs is not None:
                self._save_images(self._clean_img_dir, clean_imgs, image_idx)

            self._save_images(self._wm_img_dir, wm_imgs, image_idx)

            image_idx += len(wm_imgs)

    def _generate_in_process_wm(self, n_images: int):
        if n_images is None:
            raise ValueError('Your watermarker is set to in-process but number of generated image is not specified')

        for idx in range(0, n_images, self._watermarker.batch_size):
            batch_size = min(self._watermarker.batch_size, n_images - idx)

            message = self._watermarker.generate_messages(self._watermarker.batch_size)
            clean_imgs, wm_imgs = self._watermarker.encode(None, prompts=[], messages=message)

            if clean_imgs is not None:
                self._save_images(self._clean_img_dir, clean_imgs[:batch_size], idx)

            self._save_images(self._wm_img_dir, wm_imgs[:batch_size], idx)

    def _save_images(self, path: str, images: list[Image.Image], start_idx: int):
        for i, img in enumerate(images):
            img.save(os.path.join(path, f'{start_idx + i:08d}.png'))

    def _init_cache_folders(self):
        if not os.path.exists(self._clean_img_dir):
            os.makedirs(self._clean_img_dir)

        if not os.path.exists(self._wm_img_dir):
            os.makedirs(self._wm_img_dir)

        if not os.path.exists(self._attack_img_dir):
            os.makedirs(self._attack_img_dir)


