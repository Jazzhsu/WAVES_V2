from abc import ABC, abstractmethod
from typing import Union, Optional
from enum import Enum

from PIL import Image

class WatermarkerType(Enum):
    POST_PROCESS = 0
    IN_PROCESS = 1

class Watermarker(ABC):

    def __init__(self, batch_size: int, type: WatermarkerType):
        self._batch_size = batch_size
        self._type = type

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def type(self):
        return self._type

    @abstractmethod
    def encode(self, images: Optional[list[Image.Image]], prompts: list[str], messages: list[list[Union[bool, float]]]) -> tuple[list[Image.Image], list[Image.Image]]:
        """ Encode message into given (image, prompt) or encode message into given prompt if the
        algorithm is in process watermarking.

        Args:
            images (list[Image.Image], *optional*): A list of PIL images to be watermarked. None if
                the watermarking is in-process.
            prompts (list[str]): A list of prompts corresponding to the clean images.
            messages (list[list[Union[bool, float]]]): A list of messages to be encoded in the images.
                The messages are generated by `generate_message`.

        Returns:
            tuple[list[Image.Image], list[Image.Image]]: The return is expected to be a tuple of a
                clean images (original images, or unwatermarked version of generated image if the
                algorithm is in-process) and watermarked images.
        """
        ...

    @abstractmethod
    def decode(self, images: list[Image.Image]) -> list[list[Union[bool, float]]]:
        """ Decode messages from a list of images.

        Args:
            images (list[Image.Image]): A listed of PIL images to decode.

        Returns:
            list[list[Union[bool, float]]]: A list of messages decoded fromt the images.
        """
        ...

    @abstractmethod
    def generate_messages(self, num_message: int) -> list[list[Union[bool, float]]]:
        """ Generate `num_message` messages.

        Args:
            num_message (int): number of messages to generate. The message should either be a
                bool array (binary) or float array.

        Returns:
            list[list[Union[bool, float]]]: The generated messages.
        """
        ...