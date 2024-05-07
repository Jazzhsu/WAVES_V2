import torch
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
import os
import argparse
from torchvision.utils import save_image
from torchvision.models import resnet18
from torchattacks.attack import Attack
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from tqdm import tqdm

EPS_FACTOR = 1 / 255
ALPHA_FACTOR = 0.05
N_STEPS = 200
BATCH_SIZE = 4

STRENGTH = [2, 4, 6, 8]

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self._convs = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        )

        self._ffn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 50 * 50, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 100),
        )

    def forward(self, x):
        x = self._convs(x)
        x = self._ffn(x)
        return x

class AdversarialSurrogateAttacker:
    def __init__(self, model_path: str, strength: float, model_input_size: int = 256, device = torch.device('cuda')):
        # Load classifier
        # self._model = resnet18(pretrained=False)
        # self._model.fc = nn.Linear(self._model.fc.in_features, 2)  # Binary classification: 2 classes
        # self._model.load_state_dict(torch.load(model_path))
        # self._model = self._model.to(device)
        # self._model.eval()
        self._model = MyModel()
        self._model.load_state_dict(torch.load(model_path))
        self._model = self._model.to(device)
        self._model.eval()

        self._device = device
        self._cls_input_size = [model_input_size, model_input_size]

        strength = min(int(strength * len(STRENGTH)), len(STRENGTH) - 1)
        strength = STRENGTH[strength]
        self._strength = strength
        self._init_delta = None

        self._attacker = pgd_attack_classifier(
            model=self._model,
            eps=EPS_FACTOR * strength,
            alpha=ALPHA_FACTOR * EPS_FACTOR * strength,
            steps=N_STEPS,
            model_input_size=self._cls_input_size,
        )

    def warmup(self, data_set: Dataset):
        loader = DataLoader(data_set, batch_size=64, shuffle=True, num_workers=4)

        average_deltas = []
        for i, (images, _) in tqdm(enumerate(loader), desc='Warm up'):
            images = images.to(self._device)
            target_labels = 1 - self._model(F.resize(images, self._cls_input_size)).argmax(dim=1)

            # Attack images
            images_adv = self._attacker(images, target_labels, init_delta=None)

            average_deltas.append((images_adv - images).mean(dim=0))

            if i >= 20:
                break

        self._init_delta = torch.cat(average_deltas, dim=0).mean(dim=0)

    def attack(self, imgs: torch.Tensor):
        imgs = imgs.to(self._device)
        target_labels = 1 - torch.round(
                                torch.nn.functional.sigmoid(self._model(F.resize(imgs, self._cls_input_size))))

        # Attack images
        return self._attacker(imgs, target_labels, init_delta=self._init_delta).detach().cpu()

class WarmupPGD(Attack):
    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, model_input_size=[256, 256]):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.supported_mode = ["default", "targeted"]
        self.loss = nn.CrossEntropyLoss()
        self._cls_size = model_input_size

    def forward(self, images, labels, init_delta=None):
        """
        Overridden.
        """
        self.model.eval()

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        if init_delta is None:
            adv_images = images.clone().detach()
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        else:
            clamped_delta = torch.clamp(init_delta, min=-self.eps, max=self.eps)
            adv_images = images.clone().detach() + clamped_delta
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()


        for _ in range(self.steps):
            self.model.zero_grad()
            adv_images.requires_grad = True
            outputs = self.model(F.resize(adv_images, self._cls_size))
            outputs = torch.nn.functional.sigmoid(outputs)

            # Calculate loss
            if self.targeted:
                cost = -self.loss(outputs, target_labels)
            else:
                cost = self.loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        # delta = torch.zeros_like(images, requires_grad=True, device='cuda')
        # optimizer = torch.optim.Adam([delta], lr=0.0001)
        # for _ in range(self.steps):
        #     optimizer.zero_grad()

        #     outputs = torch.nn.functional.sigmoid(self.model(torch.clamp(images + delta, 0, 1)))
        return adv_images


def pgd_attack_classifier(model, eps, alpha, steps, model_input_size):
    # Create an instance of the attack
    attack = WarmupPGD(
        model,
        eps=eps,
        alpha=alpha,
        steps=steps,
        model_input_size=model_input_size
    )

    # Set targeted mode
    attack.set_mode_targeted_by_label(quiet=True)

    return attack
