import os
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
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

                image_name, _class_id, species, _breed_id = line.split()
                image_path = images_dir / f"{image_name}.jpg"

                # species = 1 -> cat, 2 -> dog
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


def get_transform():
    weights = ResNet18_Weights.DEFAULT

    # Use the official preprocessing tied to the pretrained ResNet-18 weights.
    # No extra random augmentation here, to keep runs easier to compare.
    return weights.transforms()


def get_device_and_loader_settings():
    batch_size = 32
    num_workers = 0

    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Deterministic-minded CUDA settings
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

        pin_memory = True

    elif torch.backends.mps.is_available():
        device = torch.device("mps")

        pin_memory = False

    else:
        device = torch.device("cpu")

        pin_memory = False

    return device, batch_size, num_workers, pin_memory


def get_data_loaders(
    dataset_dir: Path,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
):
    images_dir = dataset_dir / "images"
    annotations_dir = dataset_dir / "annotations"

    train_file = annotations_dir / "trainval.txt"
    test_file = annotations_dir / "test.txt"

    transform = get_transform()

    train_dataset = OxfordPetBinaryDataset(train_file, images_dir, transform=transform)
    test_dataset = OxfordPetBinaryDataset(test_file, images_dir, transform=transform)

    generator = torch.Generator()
    generator.manual_seed(42)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
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
            images = images.to(device, non_blocking=(device.type == "cuda"))
            labels = labels.to(device, non_blocking=(device.type == "cuda"))

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
    epochs: int = 15,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()

        # Keep BatchNorm fixed when only training the final layer
        model.apply(set_batchnorm_eval)

        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs}", leave=False)

        for images, labels in progress_bar:
            images = images.to(device, non_blocking=(device.type == "cuda"))
            labels = labels.to(device, non_blocking=(device.type == "cuda"))

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

        tqdm.write(
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

    device, batch_size, num_workers, pin_memory = get_device_and_loader_settings()

    print(f"Using device: {device}")
    print(
        f"Settings | batch_size={batch_size}, num_workers={num_workers}, "
        f"pin_memory={pin_memory}, amp=False, random_aug=False"
    )

    train_loader, test_loader = get_data_loaders(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Freeze pretrained backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer for binary classification
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=5,
        lr=2e-3,
        weight_decay=1e-3,
    )

    elapsed_seconds = time.time() - start_time
    print(f"Total runtime: {elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
