from typing import Tuple
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from diffusers import AutoencoderKL, StableDiffusionPipeline
from tqdm import tqdm
from peft import load_peft_weights, set_peft_model_state_dict, LoraConfig, get_peft_model

from conditional_BDIA_pipeline import BDIA_DDIM_pipeline
from waves.waves_benchmark import WavesBenchmark
from waves.watermarker import Watermarker, WatermarkerType
from waves.attacks import AttackMethods

def get_ring_kernels(dim: int, bits: int = 8, dtype: torch.dtype = torch.float):
    assert dim & 1

    kernels = torch.zeros((bits, 1, dim, dim), dtype=dtype)
    c_i = c_j = dim // 2
    for i in range(dim):
        for j in range(dim):
            bit = (dim // 2 - round(((c_i - i)**2 + (c_j - j)**2) ** 0.5))
            if -bits <= bit < bits:
                kernels[bit, :, i, j] = 1

    return kernels

def get_ring_correlations(imgs: torch.Tensor, kernels: torch.Tensor, channels: int = 1, ga_kernel: torch.Tensor = None, mean: bool = True):
    correlations = [ _get_ring_correlations(imgs[:, i:i+1, :, :], kernels, ga_kernel, mean) for i in range(channels) ]
    correlations = torch.stack(correlations, dim=-1)
    return correlations.mean(-1)


def _get_ring_correlations(imgs: torch.Tensor, kernels: torch.Tensor, ga_kernel: torch.Tensor = None, mean: bool = True):
    correlations = F.conv2d(imgs, kernels, stride=1)
    dim = correlations.shape[-1]
    offset = kernels.shape[-1] // 2

    correlations = correlations * imgs[:, :, offset:offset+dim, offset:offset+dim]
    correlations = correlations / torch.sum(kernels, dim=(-1, -2), keepdim=True).squeeze(-1)

    if (ga_kernel is None):
        return correlations.mean(dim=(-1, -2)) if mean else correlations

    return (correlations * ga_kernel).sum((-1, -2))

class NPCF(Watermarker):
    def __init__(self, batch_size: int = 16):
        super().__init__(batch_size=batch_size, type=WatermarkerType.POST_PROCESS)


        self._kernel = get_ring_kernels(33, 8).float()

    def load_pipe(self):
        self._pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float16).to('cuda')
        vae = self._pipe.vae

        def find_specific_layers(model):
            layers = []
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                    layers.append(name)
            return layers
        layers = find_specific_layers(vae.encoder)

        unet_lora_config = LoraConfig(
            r=4,
            lora_alpha=4,
            init_lora_weights="gaussian",
            target_modules=layers
        )
        vae = get_peft_model(vae, unet_lora_config)
        checkpoint = torch.load("./models/vae_ckpt_epoch_4_iter_40000_comb")
        set_peft_model_state_dict(vae, checkpoint['state_dict'])

        self._pipe.vae = vae
        self._pipe.vae.half()
        self._pipe.vae.eval()

    def unload_pipe(self):
        del self._pipe

    def encode(self, images: torch.Tensor, prompts: list[str], messages: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def decode(self, images: torch.Tensor) -> torch.Tensor:
        # images = images.clamp(0, 1)
        self.load_pipe()
        kmeans = KMeans(2)
        total_secret = []
        for i in range(images.size()[0]):
            image = 2*images[i]-1 #convert to [-1,1]
            latents = self._pipe.vae.encode(image.unsqueeze(0).to('cuda').type(torch.float16)).latent_dist.sample() * self._pipe.vae.config.scaling_factor
            correlation = get_ring_correlations(latents.mean(dim=1, keepdim=True).cpu().float(), self._kernel)
            kmeans.fit(correlation.detach().cpu().numpy().reshape(-1, 1))
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_
            mean_values = centers.mean(axis=1)
            if mean_values[0] > mean_values[1]:
                label_0 = 1
                label_1 = 0
            else:
                label_1 = 1
                label_0 = 0
            remapped_labels = [label_1 if label == 1 else label_0 for label in labels]
            total_secret.append(remapped_labels)
        self.unload_pipe()
        return torch.as_tensor(total_secret)

    def generate_messages(self, num_message: int) -> torch.Tensor:
        raise NotImplementedError()

if __name__ == '__main__':
    npcf = NPCF()
    benchmark = WavesBenchmark(npcf, image_src='.', cache_folder='./npcf_result/')

    benchmark.generate_attack_images(AttackMethods.COMB_DIST_ALL ^ AttackMethods.DIST_ERASE ^ AttackMethods.DIST_RESIZE_CROP ^ AttackMethods.DIST_ROTATION ^ AttackMethods.DIST_CONTRAST ^ AttackMethods.DIST_BRIGHT, 0.2)
