from os import replace
from typing import Union, Tuple
import json

import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import onnxruntime as ort
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from waves.waves_benchmark import WavesBenchmark
from waves.watermarker import Watermarker, WatermarkerType
from waves.utils.image_loader import ImageLoader, ImageData
from waves.attacks import AttackMethods

SIZE = (400, 400)

class StegaStamp(Watermarker):
    def __init__(self, model_path: str, batch_size: int = 25):
        super().__init__(batch_size=batch_size, type=WatermarkerType.POST_PROCESS)
        self._model = ort.InferenceSession(model_path)
        self._secret_pool = np.random.randint(0, 2, size=(50, 100))

    def encode(
        self, 
        images: torch.Tensor, 
        prompts: list[str],
        messages: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        secret = messages.numpy().astype(np.float32)
        np_img = F.resize(images, list(SIZE)).permute(0, 2, 3, 1).numpy()

        wm_img, _ = self._model.run(
            output_names=['stegastamp', 'residual'],
            input_feed={'image': np_img, 'secret': secret}
        )

        return images, torch.tensor(wm_img).permute(0, 3, 1, 2)

    def decode(self, images: torch.Tensor) -> torch.Tensor:
        np_img = F.resize(images, list(SIZE)).permute(0, 2, 3, 1).numpy()

        secret = self._model.run(
            output_names=['decoded'],
            input_feed={'image': np_img, 'secret': np.zeros((len(images), 100), dtype=np.float32)}
        )

        return torch.tensor(np.array(secret).squeeze(0))

    def generate_messages(self, num_message: int) -> torch.Tensor:
        secret = self._secret_pool[np.random.choice(self._secret_pool[0], num_message, replace=True), :]
        return torch.tensor(secret)

from PIL import Image
if __name__ == '__main__':

    # with open('./result/waves/secret.json', 'r') as f:
    #     secret_dict = json.load(f)

    # with open('./result/waves/attack/adv_surrogate_1.0.json', 'r') as f:
    #     attacked_secret_dict = json.load(f)

    # diffs = 0
    # total_bits = 0
    # for key, s1 in attacked_secret_dict.items():
    #     s2 = secret_dict[key]

    #     s1 = np.array(s1)
    #     s2 = np.array(s2)
    #     diffs += int(np.sum(np.abs(s1 - s2)))
    #     total_bits += len(s1)

    # print(f'bit error rate = {diffs} / {total_bits} = {diffs / total_bits:.4f}')   

    stega_stamp = StegaStamp('./models/stega_stamp.onnx', batch_size=128)
    benchmark = WavesBenchmark(stega_stamp, './image_data', './result')

    for attack in AttackMethods:
        if 'DIST' in attack.name:
            benchmark.generate_attack_images(attack, 0.2)

    # benchmark.generate_wm_images()
    # with open('./result/waves/secret.json', 'r') as f:
    #     secret_dict = json.load(f)

    # diffs = 0
    # total_bits = 0
    # dataset = ImageData(path='./pix2pix', ext='png')
    # loader = DataLoader(dataset, batch_size=64, shuffle=False)
    # bar = tqdm(total=len(dataset))
    # for imgs, img_names in loader:
    #     result = stega_stamp.decode(imgs)

    #     expected_result = np.array([secret_dict[name[:-7]] for name in img_names])
    #     diffs += np.sum(np.abs(result.numpy() - expected_result))
    #     total_bits += (result.shape[0] * result.shape[1])
    #     bar.update(len(imgs))

    # print(f'bit error rate = {diffs} / {total_bits} = {diffs / total_bits:.4f}')














