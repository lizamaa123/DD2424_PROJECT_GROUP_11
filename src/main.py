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
from training.engine import (
    evaluate_per_class,
    save_per_class_metrics,
    save_training_curves,
    train_model,
)

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


def set_training_mode(model, mode: str) -> None:
    """
    linear_probe:
        Freeze ResNet18 body and train only final fc layer.

    finetune_layer4:
        Train ResNet18 layer4 + final fc layer.
    """

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    if mode == "linear_probe":
        for param in model.fc.parameters():
            param.requires_grad = True

    elif mode == "finetune_layer4":
        for param in model.layer4.parameters():
            param.requires_grad = True

        for param in model.fc.parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Unknown mode: {mode}")


def print_trainable_parameters(model) -> None:
    print("Trainable parameters:")

    trainable_count = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}")
            trainable_count += param.numel()

    print(f"Total trainable parameters: {trainable_count:,}")


def main():
    start_time = time.time()
    load_dotenv()

    train_fraction = 1.0
    mode = "linear_probe"

    imbalanced = True
    minority_classes = [0, 1, 2, 3, 4]
    minority_fraction = 0.1
    # Examples:
    # train_fraction = 1.0   # 100%
    # train_fraction = 0.10  # 10%
    # train_fraction = 0.01  # 1%
    #
    # mode = "linear_probe"
    # mode = "finetune_layer4"

    fraction_name = str(train_fraction).replace(".", "p")

    if imbalanced:
        minority_name = "_".join(str(c) for c in minority_classes)
        minority_fraction_name = str(minority_fraction).replace(".", "p")
        experiment_name = (
            f"{mode}_imbalanced_classes_{minority_name}_"
            f"minority_{minority_fraction_name}"
        )
    else:
        experiment_name = f"{mode}_fraction_{fraction_name}"


    dataset_dir_str = os.getenv("DATASET_DIR")
    if not dataset_dir_str:
        raise ValueError("DATASET_DIR not found in .env")

    dataset_dir = Path(dataset_dir_str)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    set_seed(42)

    device, batch_size, num_workers, pin_memory = get_device_and_loader_settings()

    print(f"Experiment: {experiment_name}")
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
    train_fraction=train_fraction,
    imbalanced=imbalanced,
    minority_classes=minority_classes,
    minority_fraction=minority_fraction,
)

    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Replace final layer for 37-class breed classification
    model.fc = nn.Linear(model.fc.in_features, 37)

    # Choose whether this run is linear probing or fine-tuning
    set_training_mode(model, mode=mode)

    print_trainable_parameters(model)

    model = model.to(device)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    if mode == "linear_probe":
        lr = 2e-3
        weight_decay = 1e-3
        freeze_batchnorm = True

    elif mode == "finetune_layer4":
        lr = 1e-4
        weight_decay = 1e-4
        freeze_batchnorm = False

    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(
        f"Training config | lr={lr}, weight_decay={weight_decay}, "
        f"freeze_batchnorm={freeze_batchnorm}"
    )


    model, history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=10,
        lr=lr,
        weight_decay=weight_decay,
        freeze_batchnorm=freeze_batchnorm,
    )

    model_save_path = results_dir / f"best_{experiment_name}_model.pth"
    torch.save(model.state_dict(), model_save_path)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_save_path = results_dir / f"training_curves_{experiment_name}.png"
    save_training_curves(history, plot_save_path)

    class_metrics = evaluate_per_class(
    model=model,
    loader=test_loader,
    device=device,
    num_classes=37,
    )

    print(f"Macro F1: {class_metrics['macro_f1']:.4f}")

    per_class_save_path = results_dir / f"per_class_metrics_{experiment_name}.csv"
    save_per_class_metrics(class_metrics, per_class_save_path)

    print(f"Saved per-class metrics to: {per_class_save_path}")

    elapsed_seconds = time.time() - start_time

    print(f"Saved model to: {model_save_path}")
    print(f"Saved curves to: {plot_save_path}")
    print(f"Total runtime: {elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    main()