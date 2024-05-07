import torch
from .feature_extractors import (
    ResNet18Embedding,
    VAEEmbedding,
    ClipEmbedding,
    KLVAEEmbedding,
)
from tqdm import tqdm

EPS_FACTOR = 1 / 255
ALPHA_FACTOR = 0.05
N_STEPS = 200

STRENGTH = [2, 4, 6, 8]

class AdversarialEmbeddingAttacker:
    def __init__(self, encoder: str, strength: float, device=torch.device('cuda:0')):
        strength = min(int(strength * len(STRENGTH)), len(STRENGTH) - 1)
        strength = STRENGTH[strength]

        # load embedding model
        if encoder == "resnet18":
            # we use last layer's state as the embedding
            embedding_model = ResNet18Embedding("last")
        elif encoder == "clip":
            embedding_model = ClipEmbedding()
        elif encoder == "klvae8":
            # same vae as used in generator
            embedding_model = VAEEmbedding("stabilityai/sd-vae-ft-mse")
        elif encoder == "sdxlvae":
            embedding_model = VAEEmbedding("stabilityai/sdxl-vae")
        elif encoder == "klvae16":
            # embedding_model = KLVAEEmbedding("kl-f16")
            raise ValueError(f'KL-VAE-F16 model is not ready currently')
        else:
            raise ValueError(f"Unsupported encoder: {encoder}")

        embedding_model = embedding_model.to(device)
        embedding_model.eval()

        # Create an instance of the attack
        self._attacker = WarmupPGDEmbedding(
            model=embedding_model,
            eps=EPS_FACTOR * strength,
            alpha=ALPHA_FACTOR * EPS_FACTOR * strength,
            steps=N_STEPS,
            device=device,
        )

    def attack(self, imgs: torch.Tensor):
        return self._attacker.forward(imgs)

class WarmupPGDEmbedding:
    def __init__(
        self,
        model,
        device,
        eps=8 / 255,
        alpha=2 / 255,
        steps=10,
        loss_type="l2",
        random_start=True,
    ):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.loss_type = loss_type
        self.random_start = random_start
        self.device = device

        # Initialize the loss function
        if self.loss_type == "l1":
            self.loss_fn = torch.nn.L1Loss()
        elif self.loss_type == "l2":
            self.loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError("Unsupported loss type")

    def forward(self, images, init_delta=None):
        self.model.eval()
        images = images.clone().detach().to(self.device)

        # Get the original embeddings
        original_embeddings = self.model(images).detach()

        # initialize adv images
        if self.random_start:
            adv_images = images.clone().detach()
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        elif init_delta is not None:
            clamped_delta = torch.clamp(init_delta, min=-self.eps, max=self.eps)
            adv_images = images.clone().detach() + clamped_delta
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        else:
            assert False

        # PGD
        for _ in range(self.steps):
            self.model.zero_grad()
            adv_images.requires_grad = True
            adv_embeddings = self.model(adv_images)

            # Calculate loss
            cost = self.loss_fn(adv_embeddings, original_embeddings)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images.detach().cpu()
