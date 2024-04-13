import os

from PIL import Image

class ImageLoader:
    """ Data loader class that loads images from a given path.
    """

    def __init__(self, path: str):
        self._path = path

        if not os.path.exists(path):
            raise ValueError(f'{path} does not exists.')

        # TODO: Should we open to more image formats?
        self._image_files = sorted([ file for file in os.listdir(path) if file.endswith('.png') ])
        self._idx = 0


    def __len__(self):
        return len(self._image_files)

    def load_batch(self, batch_size: int) -> list[Image.Image]:
        """ Loads a batch of images and return.
        """

        end_idx = min(self._idx + batch_size, len(self))

        batch_image = [
            Image.open(os.path.join(self._path, self._image_files[i])) for i in range(self._idx, end_idx)
        ]

        self._idx = end_idx
        return batch_image

    def position(self) ->int:
        return self._idx

    def has_next(self) -> bool:
        """ Returns true if there's more images to load.
        """
        return self._idx != len(self)

    def reset(self):
        self._idx = 0

# Example use case
if __name__ == '__main__':
    loader = ImageLoader('images/main/dalle3/tree_ring')

    while loader.has_next():
        print(loader.load_batch(2))