import os
import argparse
import torch
import torch.nn as nn
from torch.nn.modules import MaxPool2d
import torch.optim as optim
from torch.utils.data import random_split, ConcatDataset
from torchvision.models import resnet18
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from waves.utils.image_loader import SurrogateImageData, StegaStampSurrogateImageData
from stegastamp_bm import StegaStamp



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

def train_surrogate_classifier(
    train_data_cls_1_path: str, 
    train_data_cls_1_extension: str,
    train_data_cls_2_path: str,
    train_data_cls_2_extension: str,
    train_img_size: int,
    model_save_path: str, 
    model_save_name: str,
    learning_rate: float = 1e-3, 
    num_epochs: int = 10, 
    device: str = 'cuda', 

):
    # stega_stamp = StegaStamp('./stega_stamp.onnx')
    train_fraction = 0.9

    # Load datasets
    # Binary classifier
    # data_set = SurrogateImageData(
    #     path_a = train_data_cls_1_path,
    #     ext_a = train_data_cls_1_extension,
    #     path_b = train_data_cls_2_path,
    #     ext_b = train_data_cls_2_extension,
    #     size = (train_img_size, train_img_size),
    #     n_image = 1000,
    # )

    data_set = StegaStampSurrogateImageData(
        path='./result2/waves/wm',
        ext='pt',
        secret_path='./result2/waves/secret.json',
        size=[train_img_size, train_img_size]
    )

    data_set_2 = StegaStampSurrogateImageData(
        path='./result/waves/wm',
        ext='pt',
        secret_path='./result/waves/secret.json',
        size=[train_img_size, train_img_size]
    )

    data = ConcatDataset([data_set, data_set_2])


    training_set, validation_set = random_split(data, [train_fraction, 1 - train_fraction])
    train_loader = DataLoader(
        training_set,
        batch_size = 16,
        shuffle = True,
        num_workers = 16,
        pin_memory = True,
    )

    valid_loader = DataLoader(
        validation_set,
        batch_size = 16,
        shuffle = True,
        num_workers = 16,
        pin_memory = True,
    )

    print(f"Training on {len(data_set_2)} samples.")

    # Load pretrained ResNet18 and modify the final layer
    # model = resnet18(pretrained=False)

    # Modify the final layer only if the number of output features doesn't match
    # model.fc = nn.Linear(model.fc.in_features, 100)
    # model.fc = nn.Sequential(
    #     nn.Linear(model.fc.in_features, 20000),
    #     nn.LeakyReLU(),
    #     nn.Linear(20000, 512),
    #     nn.LeakyReLU(),
    #     nn.Linear(512, 100),
    #     nn.Sigmoid()
    # )
    model = MyModel()
    # model.load_state_dict(torch.load('./stegastamp_surrogate_model_50_cls.pth'))

    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    best_val_accuracy = 0.0
    best_model_state = None

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for images, labels in tqdm(train_loader):
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            # outputs = torch.nn.functional.sigmoid(outputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate predictions for training accuracy
            total += labels.size(0) * labels.size(1)
            predicted = torch.round(nn.functional.sigmoid(outputs))
            correct += (1 - torch.abs(predicted - labels)).sum().item()

        train_loss = total_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%"
        )

        # Evaluation on the validation set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(valid_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                outputs = torch.nn.functional.sigmoid(outputs)
                total += labels.size(0) * labels.size(1)
                predicted = torch.round(outputs)
                correct += (1 - torch.abs(predicted - labels)).sum().item()

        val_accuracy = 100 * correct / total
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

        # Update best validation accuracy and model state
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()  # Copy the model state
            print(
                f"New best model found at epoch {epoch + 1} with validation accuracy: {val_accuracy:.2f}%"
            )

    print("Training complete!")

    # Save the entire model
    # Save the best model based on validation accuracy
    if best_model_state is not None:
        save_path_best = os.path.join(
            model_save_path, model_save_name + ".pth"
        )
        torch.save(best_model_state, save_path_best)
        print(
            f"Best model saved to {save_path_best} with validation accuracy: {best_val_accuracy:.2f}%"
        )
    else:
        save_path_full = os.path.join(
            model_save_path, model_save_name + ".pth"
        )
        torch.save(model.state_dict(), save_path_full)
        print(f"Entire model saved to {save_path_full}")
    return


if __name__ == "__main__":
    train_surrogate_classifier(
        train_data_cls_1_path='./image_data',
        train_data_cls_1_extension='png',
        train_data_cls_2_path='./result/waves/wm',
        train_data_cls_2_extension='pt',
        train_img_size=400,
        model_save_path='./',
        model_save_name='stegastamp_surrogate_model',
        learning_rate=0.0001,
        num_epochs=10
    )
