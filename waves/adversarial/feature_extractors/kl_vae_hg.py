from .base_encoder import BaseEncoder
from diffusers.models import AutoencoderKL
import torchvision.transforms.functional as F


class VAEEmbedding(BaseEncoder):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoencoderKL.from_pretrained(model_name)

    def forward(self, images):
        images = F.resize(images, [256, 256])
        images = 2.0 * images - 1.0
        output = self.model.encode(images)
        z = output.latent_dist.mode()
        return z
