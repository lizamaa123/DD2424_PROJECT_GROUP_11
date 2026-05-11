from pathlib import Path
import random
from collections import defaultdict
from typing import Optional
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
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

                # now using the 37 category (0-36 indexed)
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


class OxfordPetUnlabeledDataset(Dataset):
    def __init__(self, split_file: Path, images_dir: Path, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.samples = []

        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                tokens = line.split()
                image_name = tokens[0]
                if not image_name.endswith(".jpg"):
                    image_name = f"{image_name}.jpg"

                image_path = images_dir / image_name
                if image_path.exists():
                    self.samples.append(image_path)
                else:
                    print(f"Warning: missing image {image_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, str(image_path)


class OxfordPetPseudoLabelDataset(Dataset):
    def __init__(self, pseudo_samples, transform=None):
        self.transform = transform
        self.samples = pseudo_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, pseudo_label, _confidence = self.samples[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, pseudo_label


class OxfordPetUnlabeledSubsetDataset(Dataset):
    def __init__(self, base_dataset: OxfordPetDataset, indices):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = base_dataset.transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        image_path, _label = self.base_dataset.samples[base_idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, str(image_path)
    
def get_transform():
    weights = ResNet18_Weights.DEFAULT

    # Use the official preprocessing tied to the pretrained ResNet-18 weights.
    # No extra random augmentation here, to keep runs easier to compare.
    return weights.transforms()

def get_data_loaders(
    dataset_dir: Path,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    return_datasets: bool = False,
):
    images_dir = dataset_dir / "images"
    annotations_dir = dataset_dir / "annotations"

    train_file = annotations_dir / "trainval.txt"
    test_file = annotations_dir / "test.txt"

    transform = get_transform()

    train_dataset = OxfordPetDataset(train_file, images_dir, transform=transform)
    test_dataset = OxfordPetDataset(test_file, images_dir, transform=transform)

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

    if return_datasets:
        return train_loader, test_loader, train_dataset, test_dataset

    return train_loader, test_loader


def get_unlabeled_dataset(
    dataset_dir: Path,
    split_filename: str = "unlabeled.txt",
) -> Optional[OxfordPetUnlabeledDataset]:
    images_dir = dataset_dir / "images"
    annotations_dir = dataset_dir / "annotations"
    split_file = annotations_dir / split_filename

    if not split_file.exists():
        return None

    transform = get_transform()
    return OxfordPetUnlabeledDataset(split_file, images_dir, transform=transform)


def stratified_labeled_unlabeled_indices(
    dataset: OxfordPetDataset,
    labeled_fraction: float,
    seed: int = 42,
):
    if not 0.0 < labeled_fraction < 1.0:
        raise ValueError("labeled_fraction must be in (0, 1)")

    class_to_indices = defaultdict(list)
    for idx, (_path, label) in enumerate(dataset.samples):
        class_to_indices[label].append(idx)

    labeled_indices = []
    unlabeled_indices = []
    rng = random.Random(seed)

    for label, indices in class_to_indices.items():
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        count = len(shuffled)
        labeled_count = max(1, int(round(count * labeled_fraction)))
        labeled_count = min(labeled_count, count)

        class_labeled = shuffled[:labeled_count]
        class_unlabeled = shuffled[labeled_count:]

        labeled_indices.extend(class_labeled)
        unlabeled_indices.extend(class_unlabeled)

    labeled_indices.sort()
    unlabeled_indices.sort()
    return labeled_indices, unlabeled_indices
