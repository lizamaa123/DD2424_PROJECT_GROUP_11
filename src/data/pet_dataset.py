from pathlib import Path
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
                label = int(species) - 1

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

    return train_loader, test_loader