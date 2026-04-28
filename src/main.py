import os
import random
import time
from pathlib import Path

import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights
from dotenv import load_dotenv

from data.pet_dataset import get_data_loaders
from training.engine import save_training_curves, train_model

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

    # Replace final layer for 37 classification
    model.fc = nn.Linear(model.fc.in_features, 37)
    model = model.to(device)

    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Catch the returned model and history
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=10,
        lr=2e-3,
        weight_decay=1e-3,
    )

    # Save the best model weights
    model_save_path = results_dir / "best_linear_probe_model.pth"
    torch.save(model.state_dict(), model_save_path)

    # Save the training curves
    plot_save_path = results_dir / "training_curves_linear_probe.png"
    save_training_curves(history, plot_save_path)

    elapsed_seconds = time.time() - start_time
    print(f"Total runtime: {elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    main()