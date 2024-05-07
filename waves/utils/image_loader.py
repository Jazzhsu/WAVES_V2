import os
from typing import Tuple
import json

import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F

class ImageData(Dataset):
    def __init__(self, path: str, ext: str = 'png', n_image: int | None = None):
        self._path = path
        self._data = sorted([
            file for file in os.listdir(path) if file.endswith(ext)
        ])[:n_image]
        self._ext = ext

        secret_path = os.path.join(path, '..', 'secret.json')
        with open(secret_path, mode='r', encoding='utf-8') as f:
            self._secret = json.load(f)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self._path, self._data[idx])
        if self._ext == 'png':
            img = Image.open(img_path)
            img = transforms.PILToTensor()(img) / 255.

        elif self._ext == 'pt':
            img = torch.load(img_path)

        else:
            img = None

        name = self._data[idx].split('.')[0]
        return img, name, torch.tensor(self._secret[name])

class NPCFImageData(Dataset):
    def __init__(self, path: str, data: str = 'watermarked_no_lpips_std_12.pt', key: str = 'key_no_lpips_std_12.pt'):
        self._image_data = torch.load(os.path.join(path, data), map_location='cpu')
        self._key = torch.load(os.path.join(path, key))

    def __len__(self):
        return len(self._key)

    def __getitem__(self, idx: int):
        img_name = list(self._image_data)[idx]

        return self._image_data[img_name].squeeze(0), torch.tensor(self._key[img_name])

class SurrogateImageData(Dataset):
    def __init__(
        self, 
        path_a: str, 
        ext_a: str,
        path_b: str, 
        ext_b: str, 
        size: Tuple[int, int], 
        n_image: int | None = None
    ):
        data_a = [ (os.path.join(path_a, file), 0) for file in os.listdir(path_a) if file.endswith(ext_a) ][:n_image]
        data_b = [ (os.path.join(path_b, file), 1) for file in os.listdir(path_b) if file.endswith(ext_b) ][:n_image]

        self._data = data_a + data_b
        self._size = size

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int):
        img_path, classification = self._data[idx]
        if img_path.endswith('png'):
            img = Image.open(img_path)
            img = transforms.PILToTensor()(img) / 255.

        elif img_path.endswith('pt'):
            img = torch.load(img_path)

        else:
            raise ValueError(f'Image format unsupported: {img_path}')

        return F.resize(img, list(self._size)), classification

class StegaStampSurrogateImageData(Dataset):
    def __init__(self, path: str, ext: str, secret_path: str, size: list = [256, 256]):
        self._path = path
        self._data = [ file for file in os.listdir(path) if file.endswith(ext) ]
        self._size = size

        with open(secret_path, 'r', encoding='utf-8') as f:
            self._screct_dict = json.load(f)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self._path, self._data[idx])
        img = F.resize(torch.load(img_path), self._size)
        secret = torch.tensor(self._screct_dict[self._data[idx].split('.')[0]], dtype=torch.float32)

        return img, secret

class ImageLoader:
    """ Data loader class that loads images from a given path.
    """

    def __init__(self, path: str, n_images: int = None, ext: str = 'png'):
        self._path = path

        if not os.path.exists(path):
            raise ValueError(f'{path} does not exists.')

        # TODO: Should we open to more image formats?
        self._image_files = sorted([ file for file in os.listdir(path) if file.endswith(ext) ])
        self._idx = 0
        self._size = min(n_images, len(self._image_files)) if n_images is not None else n_images
        self._ext = ext


    def __len__(self):
        return len(self._image_files) if self._size is None else self._size

    def load_batch(self, batch_size: int) -> Tuple[list[Image.Image], list[str]]:
        """ Loads a batch of images and return.
        """

        end_idx = min(self._idx + batch_size, len(self))

        batch_image = [
            (Image.open if self._ext == 'png' else np.load)(os.path.join(self._path, self._image_files[i])) for i in range(self._idx, end_idx)
        ]

        file_names = [self._image_files[i].split('.')[0] for i in range(self._idx, end_idx)]

        self._idx = end_idx
        return batch_image, file_names

    def position(self) ->int:
        return self._idx

    def has_next(self) -> bool:
        """ Returns true if there's more images to load.
        """
        return self._idx != len(self)

    def reset(self, n_images: int = None):
        self._idx = 0
        self._size = min(n_images, len(self._image_files)) if n_images is not None else n_images

""" Example use case

if __name__ == '__main__':
    loader = ImageLoader('images/main/dalle3/tree_ring')

    while loader.has_next():
        print(loader.load_batch(2))

"""
