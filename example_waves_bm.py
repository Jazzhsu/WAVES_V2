from PIL import Image
from waves.watermarker import Watermarker, WatermarkerType
from waves.waves_benchmark import WavesBenchmark
from waves.attacks import AttackMethods
import numpy as np

""" A naive watermarking algorithm that marks the paints 200x200 block of the images white.
"""
class MyWatermarker(Watermarker):
    def __init__(self):
        super().__init__(batch_size=2, type=WatermarkerType.POST_PROCESS)

    def encode(self, images: list[Image.Image] | None, prompts: list[str], messages: list[list[bool | float]]) -> tuple[list[Image.Image], list[Image.Image]]:
        wm_images = [ np.asarray(image.copy()) for image in images ]

        wm_images = np.asarray(wm_images)
        wm_images[:, 400:600, 400:600, :] = 255

        wm_images = [ Image.fromarray(im) for im in wm_images ]
        return images, wm_images

    def decode(self, images: list[Image.Image]) -> list[list[bool | float]]:
        return super().decode(images)

    def generate_messages(self, num_message: int) -> list[list[bool | float]]:
        return super().generate_messages(num_message)

if __name__ == '__main__':
    watermarker = MyWatermarker()

    wave_bm = WavesBenchmark(watermarker, image_src='images/', cache_folder='test_cache/')
    wave_bm.generate_wm_images()

    wave_bm.generate_attack_images(AttackMethods.COMB_DIST_DEGRAD, strength=0.5)

    # More to be implemented.
