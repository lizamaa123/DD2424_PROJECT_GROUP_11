import os
from pathlib import Path

import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights
from dotenv import load_dotenv

from data.pet_dataset import get_data_loaders
from training.engine import (
    compute_confusion_matrix,
    evaluate,
    evaluate_per_class,
    save_confusion_matrix_plot,
    save_normalized_confusion_matrix_plot,
    save_per_class_metrics,
)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(mode: str, model_path: Path, device):
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    model.fc = nn.Linear(model.fc.in_features, 37)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model


def main():
    load_dotenv()

    # Change these when needed for diffrent tests see the examples

    experiment_name = "linear_probe_imbalanced_classes_0_1_2_3_4_minority_0p1"
    model_path = Path("results/best_linear_probe_imbalanced_classes_0_1_2_3_4_minority_0p1_model.pth")

    # Examples:
    # experiment_name = "linear_probe_fraction_0p1"
    # model_path = Path("results/best_linear_probe_fraction_0p1_model.pth")
    #
    # experiment_name = "finetune_layer4_fraction_1p0"
    # model_path = Path("results/best_finetune_layer4_fraction_1p0_model.pth")


    dataset_dir_str = os.getenv("DATASET_DIR")
    if not dataset_dir_str:
        raise ValueError("DATASET_DIR not found in .env")

    dataset_dir = Path(dataset_dir_str)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    device = get_device()

    print(f"Evaluating experiment: {experiment_name}")
    print(f"Using device: {device}")
    print(f"Model path: {model_path}")

    _, test_loader = get_data_loaders(
        dataset_dir=dataset_dir,
        batch_size=32,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        train_fraction=1.0,
    )

    model = build_model(
        mode=experiment_name,
        model_path=model_path,
        device=device,
    )

    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4%}")

    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    class_metrics = evaluate_per_class(
        model=model,
        loader=test_loader,
        device=device,
        num_classes=37,
    )

    print(f"Macro F1: {class_metrics['macro_f1']:.4f}")

    per_class_path = results_dir / f"per_class_metrics_{experiment_name}.csv"
    save_per_class_metrics(class_metrics, per_class_path)

    confusion = compute_confusion_matrix(
        model=model,
        loader=test_loader,
        device=device,
        num_classes=37,
    )

    confusion_path = figures_dir / f"confusion_matrix_{experiment_name}.png"
    normalized_confusion_path = figures_dir / f"normalized_confusion_matrix_{experiment_name}.png"

    save_confusion_matrix_plot(confusion, confusion_path)
    save_normalized_confusion_matrix_plot(confusion, normalized_confusion_path)

    print(f"Saved per-class metrics to: {per_class_path}")
    print(f"Saved confusion matrix to: {confusion_path}")
    print(f"Saved normalized confusion matrix to: {normalized_confusion_path}")


if __name__ == "__main__":
    main()