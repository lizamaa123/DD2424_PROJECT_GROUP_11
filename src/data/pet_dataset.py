from pathlib import Path
from PIL import Image
from collections import defaultdict
import random

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models import ResNet18_Weights


class OxfordPetDataset(Dataset):
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

                # 37-class breed label, converted from 1-37 to 0-36
                label = int(_class_id) - 1

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


def make_stratified_subset(dataset, fraction: float, seed: int = 42):
    """
    Create a smaller class-balanced subset.

    fraction=1.0 means full dataset.
    fraction=0.1 means 10% of each class.
    fraction=0.01 means 1% of each class.
    """

    if fraction >= 1.0:
        return dataset

    random.seed(seed)

    class_to_indices = defaultdict(list)

    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices[label].append(idx)

    selected_indices = []

    for label, indices in class_to_indices.items():
        random.shuffle(indices)

        # Keep at least one image per class
        n_keep = max(1, int(len(indices) * fraction))
        selected_indices.extend(indices[:n_keep])

    random.shuffle(selected_indices)

    return Subset(dataset, selected_indices)


def make_imbalanced_subset(
    dataset,
    minority_classes,
    minority_fraction: float = 0.1,
    seed: int = 42,
):
    """
    Create an imbalanced dataset.

    minority_classes:
        List of class IDs that should have fewer training examples.
        Example: [0, 1, 2, 3, 4]

    minority_fraction:
        Fraction of examples to keep for the minority classes.
        Example: 0.1 keeps only 10% of those classes.

    All other classes keep 100% of their examples.
    """

    random.seed(seed)

    class_to_indices = defaultdict(list)

    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices[label].append(idx)

    selected_indices = []

    for label, indices in class_to_indices.items():
        random.shuffle(indices)

        if label in minority_classes:
            n_keep = max(1, int(len(indices) * minority_fraction))
            selected_indices.extend(indices[:n_keep])
        else:
            selected_indices.extend(indices)

    random.shuffle(selected_indices)

    return Subset(dataset, selected_indices)


def get_transform():
    weights = ResNet18_Weights.DEFAULT
    return weights.transforms()


def get_data_loaders(
    dataset_dir: Path,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    train_fraction: float = 1.0,
    imbalanced: bool = False,
    minority_classes=None,
    minority_fraction: float = 0.1,
):
    images_dir = dataset_dir / "images"
    annotations_dir = dataset_dir / "annotations"

    train_file = annotations_dir / "trainval.txt"
    test_file = annotations_dir / "test.txt"

    transform = get_transform()

    train_dataset = OxfordPetDataset(train_file, images_dir, transform=transform)
    test_dataset = OxfordPetDataset(test_file, images_dir, transform=transform)

    if imbalanced:
        if minority_classes is None:
            minority_classes = [0, 1, 2, 3, 4]

        train_dataset = make_imbalanced_subset(
            dataset=train_dataset,
            minority_classes=minority_classes,
            minority_fraction=minority_fraction,
        )

        print(
            f"Using imbalanced training set | "
            f"minority_classes={minority_classes}, "
            f"minority_fraction={minority_fraction}"
        )

    else:
        train_dataset = make_stratified_subset(
            dataset=train_dataset,
            fraction=train_fraction,
        )

    print(f"Training samples used: {len(train_dataset)}")

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