import os
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
from dotenv import load_dotenv
from tqdm import tqdm


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class OxfordPetBinaryDataset(Dataset):
    def __init__(self, split_file: Path, images_dir: Path, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.samples = []

        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                image_name, class_id, species, breed_id = line.split()
                image_path = images_dir / f"{image_name}.jpg"

                # species = 1 for cat, 2 for dog
                label = 0 if int(species) == 1 else 1

                if image_path.exists():
                    self.samples.append((image_path, label))
                else:
                    print(f"Warning: missing image {image_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_transforms():
    weights = ResNet18_Weights.DEFAULT
    base_transform = weights.transforms()

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        base_transform,
    ])

    test_transform = base_transform

    return train_transform, test_transform


def get_data_loaders(dataset_dir: Path, batch_size: int = 32, num_workers: int = 0):
    images_dir = dataset_dir / "images"
    annotations_dir = dataset_dir / "annotations"

    train_file = annotations_dir / "trainval.txt"
    test_file = annotations_dir / "test.txt"

    train_transform, test_transform = get_transforms()

    train_dataset = OxfordPetBinaryDataset(train_file, images_dir, transform=train_transform)
    test_dataset = OxfordPetBinaryDataset(test_file, images_dir, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader


def set_batchnorm_eval(module):
    if isinstance(module, nn.BatchNorm2d):
        module.eval()


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(
    model,
    train_loader,
    test_loader,
    device,
    epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()

        # Important: keep BatchNorm layers in eval mode when backbone is frozen
        model.apply(set_batchnorm_eval)

        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs}", leave=False)
        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / total
        train_acc = correct / total

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4%} | "
            f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4%}"
        )


def main():
    start_time = time.time()
    load_dotenv()

    dataset_dir_str = os.getenv("DATASET_DIR")
    if not dataset_dir_str:
        raise ValueError("DATASET_DIR not found in .env")

    dataset_dir = Path(dataset_dir_str)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    set_seed(42)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    train_loader, test_loader = get_data_loaders(
        dataset_dir=dataset_dir,
        batch_size=32,
        num_workers=0,
    )

    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Freeze pretrained backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model = model.to(device)

    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=5,
        lr=1e-4,
        weight_decay=1e-4,
    )

    elapsed_seconds = time.time() - start_time
    print(f"Total runtime: {elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
