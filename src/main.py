import csv
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from dotenv import load_dotenv
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import models
from torchvision.models import ResNet18_Weights

from data.pet_dataset import (
    OxfordPetPseudoLabelDataset,
    OxfordPetUnlabeledSubsetDataset,
    get_data_loaders,
    stratified_labeled_unlabeled_indices,
)
from training.engine import (
    evaluate_detailed,
    generate_pseudo_labels,
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


def get_env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_float_list_env(name: str, default: str):
    value = os.getenv(name, default)
    values = []
    for part in value.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    if not values:
        raise ValueError(f"{name} produced an empty list. Got: {value}")
    return values


def make_loader(dataset, batch_size, num_workers, pin_memory, shuffle, seed: int = 42):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
        kwargs["generator"] = generator
    return DataLoader(**kwargs)


def create_transfer_model(device, num_classes: int = 37, mode: str = "linear_probe"):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

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

    return model.to(device)


def count_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_pseudo_labels_csv(pseudo_samples, save_path: Path) -> None:
    with open(save_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_path", "pseudo_label", "confidence"])
        writer.writerows(pseudo_samples)


def load_pseudo_labels_csv(load_path: Path):
    pseudo_samples = []
    with open(load_path, "r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            pseudo_samples.append(
                (
                    row["image_path"],
                    int(row["pseudo_label"]),
                    float(row["confidence"]),
                )
            )
    return pseudo_samples


def filter_pseudo_samples_by_threshold(pseudo_samples, threshold: float):
    return [sample for sample in pseudo_samples if sample[2] >= threshold]


def get_pseudo_label_class_distribution(pseudo_samples, num_classes: int = 37):
    counts = [0 for _ in range(num_classes)]
    for _path, label, _confidence in pseudo_samples:
        counts[label] += 1
    return counts


def save_class_distribution_csv(class_counts, save_path: Path):
    with open(save_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["class_id", "count"])
        for class_id, count in enumerate(class_counts):
            writer.writerow([class_id, count])


def save_per_class_metrics_csv(metrics, save_path: Path):
    with open(save_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["class_id", "accuracy", "f1"])
        for class_id, (acc, f1) in enumerate(
            zip(metrics["per_class_accuracy"], metrics["per_class_f1"])
        ):
            writer.writerow([class_id, acc, f1])


def save_comparison_csv(rows, save_path: Path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(save_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fraction_to_tag(fraction: float) -> str:
    return f"{int(round(fraction * 100))}pct"


def threshold_to_tag(threshold: float) -> str:
    return f"{threshold:.2f}".replace(".", "p")


def run_full_supervised_training(
    dataset_dir,
    device,
    batch_size,
    num_workers,
    pin_memory,
    results_dir,
):
    train_loader, test_loader, _train_dataset, _test_dataset = get_data_loaders(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        return_datasets=True,
    )

    model = create_transfer_model(device=device, num_classes=37, mode="linear_probe")
    print(
        "Full supervised mode=linear_probe | "
        f"trainable_params={count_trainable_parameters(model):,}"
    )
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=10,
        lr=2e-3,
        weight_decay=1e-3,
    )

    model_save_path = results_dir / "best_linear_probe_model.pth"
    torch.save(model.state_dict(), model_save_path)

    plot_save_path = results_dir / "training_curves_linear_probe.png"
    save_training_curves(history, plot_save_path)

    metrics = evaluate_detailed(
        model=model,
        loader=test_loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        num_classes=37,
    )
    print(
        f"Final full-data metrics | acc={metrics['accuracy']:.4%}, "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )


def run_limited_label_experiments(
    dataset_dir,
    device,
    batch_size,
    num_workers,
    pin_memory,
    results_dir,
):
    figures_dir = results_dir / "figures"
    models_dir = results_dir / "models"
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    labeled_fractions = parse_float_list_env("LABELED_FRACTIONS", "0.1,0.01")
    pseudo_thresholds = parse_float_list_env("PSEUDO_LABEL_THRESHOLDS", "0.7,0.8,0.9")
    baseline_epochs = int(os.getenv("BASELINE_EPOCHS", "10"))
    pseudo_epochs = int(os.getenv("PSEUDO_EPOCHS", "10"))
    baseline_lr = float(os.getenv("BASELINE_LR", "2e-3"))
    pseudo_lr = float(os.getenv("PSEUDO_LR", "1e-3"))
    weight_decay = float(os.getenv("WEIGHT_DECAY", "1e-3"))
    model_mode = os.getenv("PSEUDO_MODEL_MODE", "linear_probe")
    print(f"Pseudo-label experiment model mode: {model_mode}")

    _train_loader, test_loader, train_dataset, _test_dataset = get_data_loaders(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        return_datasets=True,
    )

    criterion = nn.CrossEntropyLoss()
    comparison_rows = []
    baseline_by_fraction = {}

    for labeled_fraction in labeled_fractions:
        fraction_tag = fraction_to_tag(labeled_fraction)
        print(f"\n--- Running limited-label setup: {fraction_tag} ---")

        labeled_indices, unlabeled_indices = stratified_labeled_unlabeled_indices(
            train_dataset,
            labeled_fraction=labeled_fraction,
            seed=42,
        )

        labeled_dataset = Subset(train_dataset, labeled_indices)
        unlabeled_subset_dataset = OxfordPetUnlabeledSubsetDataset(train_dataset, unlabeled_indices)

        labeled_loader = make_loader(
            labeled_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,
            seed=42,
        )
        unlabeled_loader = make_loader(
            unlabeled_subset_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
        )

        baseline_model = create_transfer_model(
            device=device,
            num_classes=37,
            mode=model_mode,
        )
        print(
            f"[{fraction_tag}] Baseline mode={model_mode} | "
            f"trainable_params={count_trainable_parameters(baseline_model):,}"
        )
        baseline_model, baseline_history = train_model(
            model=baseline_model,
            train_loader=labeled_loader,
            test_loader=test_loader,
            device=device,
            epochs=baseline_epochs,
            lr=baseline_lr,
            weight_decay=weight_decay,
        )

        baseline_metrics = evaluate_detailed(
            model=baseline_model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            num_classes=37,
        )

        baseline_model_path = models_dir / f"baseline_{fraction_tag}.pth"
        torch.save(baseline_model.state_dict(), baseline_model_path)
        baseline_curves_path = figures_dir / f"training_curves_baseline_{fraction_tag}.png"
        save_training_curves(baseline_history, baseline_curves_path)
        baseline_per_class_path = results_dir / f"per_class_metrics_baseline_{fraction_tag}.csv"
        save_per_class_metrics_csv(baseline_metrics, baseline_per_class_path)
        print(
            f"[{fraction_tag}] Baseline metrics | "
            f"acc={baseline_metrics['accuracy']:.4%}, "
            f"macro_f1={baseline_metrics['macro_f1']:.4f}"
        )

        baseline_by_fraction[labeled_fraction] = baseline_metrics
        comparison_rows.append(
            {
                "scenario": "supervised_baseline",
                "labeled_fraction": labeled_fraction,
                "threshold": "",
                "labeled_samples": len(labeled_indices),
                "pseudo_samples": 0,
                "total_train_samples": len(labeled_indices),
                "test_accuracy": baseline_metrics["accuracy"],
                "test_macro_f1": baseline_metrics["macro_f1"],
                "delta_acc_vs_baseline": 0.0,
                "delta_macro_f1_vs_baseline": 0.0,
                "model_path": str(baseline_model_path),
                "curves_path": str(baseline_curves_path),
                "pseudo_labels_path": "",
            }
        )

        pseudo_samples_all, threshold_stats = generate_pseudo_labels(
            model=baseline_model,
            unlabeled_loader=unlabeled_loader,
            device=device,
            thresholds=pseudo_thresholds,
        )

        pseudo_all_path = results_dir / f"pseudo_labels_{fraction_tag}_all.csv"
        save_pseudo_labels_csv(pseudo_samples_all, pseudo_all_path)
        pseudo_samples_all = load_pseudo_labels_csv(pseudo_all_path)

        threshold_stats_path = results_dir / f"pseudo_label_threshold_stats_{fraction_tag}.csv"
        with open(threshold_stats_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["threshold", "total", "kept", "keep_ratio"])
            for threshold in pseudo_thresholds:
                stats = threshold_stats[float(threshold)]
                writer.writerow([threshold, stats["total"], stats["kept"], stats["keep_ratio"]])
                print(
                    f"[{fraction_tag}] Threshold {threshold:.2f} | "
                    f"kept={stats['kept']}/{stats['total']} "
                    f"({stats['keep_ratio']:.2%})"
                )

        for threshold in pseudo_thresholds:
            threshold_tag = threshold_to_tag(threshold)
            filtered_samples = filter_pseudo_samples_by_threshold(pseudo_samples_all, threshold)

            filtered_pseudo_path = (
                results_dir / f"pseudo_labels_{fraction_tag}_thr_{threshold_tag}.csv"
            )
            save_pseudo_labels_csv(filtered_samples, filtered_pseudo_path)

            class_dist = get_pseudo_label_class_distribution(filtered_samples, num_classes=37)
            class_dist_path = (
                results_dir / f"pseudo_label_class_distribution_{fraction_tag}_thr_{threshold_tag}.csv"
            )
            save_class_distribution_csv(class_dist, class_dist_path)

            if len(filtered_samples) == 0:
                comparison_rows.append(
                    {
                        "scenario": "pseudo_label_training",
                        "labeled_fraction": labeled_fraction,
                        "threshold": threshold,
                        "labeled_samples": len(labeled_indices),
                        "pseudo_samples": 0,
                        "total_train_samples": len(labeled_indices),
                        "test_accuracy": "",
                        "test_macro_f1": "",
                        "delta_acc_vs_baseline": "",
                        "delta_macro_f1_vs_baseline": "",
                        "model_path": "",
                        "curves_path": "",
                        "pseudo_labels_path": str(filtered_pseudo_path),
                    }
                )
                print(
                    f"Skipping threshold {threshold:.2f} for {fraction_tag}: no pseudo labels kept."
                )
                continue

            pseudo_dataset = OxfordPetPseudoLabelDataset(
                pseudo_samples=filtered_samples,
                transform=train_dataset.transform,
            )
            combined_train_dataset = ConcatDataset([labeled_dataset, pseudo_dataset])
            combined_train_loader = make_loader(
                combined_train_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                shuffle=True,
                seed=42,
            )

            pseudo_model = create_transfer_model(
                device=device,
                num_classes=37,
                mode=model_mode,
            )
            print(
                f"[{fraction_tag}] Student mode={model_mode}, threshold={threshold:.2f} | "
                f"trainable_params={count_trainable_parameters(pseudo_model):,}"
            )
            pseudo_model, pseudo_history = train_model(
                model=pseudo_model,
                train_loader=combined_train_loader,
                test_loader=test_loader,
                device=device,
                epochs=pseudo_epochs,
                lr=pseudo_lr,
                weight_decay=weight_decay,
            )

            pseudo_metrics = evaluate_detailed(
                model=pseudo_model,
                loader=test_loader,
                criterion=criterion,
                device=device,
                num_classes=37,
            )

            pseudo_model_path = models_dir / f"pseudo_{fraction_tag}_thr_{threshold_tag}.pth"
            torch.save(pseudo_model.state_dict(), pseudo_model_path)
            pseudo_curves_path = (
                figures_dir / f"training_curves_pseudo_{fraction_tag}_thr_{threshold_tag}.png"
            )
            save_training_curves(pseudo_history, pseudo_curves_path)
            pseudo_per_class_path = (
                results_dir / f"per_class_metrics_pseudo_{fraction_tag}_thr_{threshold_tag}.csv"
            )
            save_per_class_metrics_csv(pseudo_metrics, pseudo_per_class_path)

            baseline_metrics_ref = baseline_by_fraction[labeled_fraction]
            print(
                f"[{fraction_tag}] Threshold {threshold:.2f} metrics | "
                f"baseline_acc={baseline_metrics_ref['accuracy']:.4%}, "
                f"baseline_macro_f1={baseline_metrics_ref['macro_f1']:.4f}, "
                f"pseudo_acc={pseudo_metrics['accuracy']:.4%}, "
                f"pseudo_macro_f1={pseudo_metrics['macro_f1']:.4f}"
            )
            comparison_rows.append(
                {
                    "scenario": "pseudo_label_training",
                    "labeled_fraction": labeled_fraction,
                    "threshold": threshold,
                    "labeled_samples": len(labeled_indices),
                    "pseudo_samples": len(filtered_samples),
                    "total_train_samples": len(combined_train_dataset),
                    "test_accuracy": pseudo_metrics["accuracy"],
                    "test_macro_f1": pseudo_metrics["macro_f1"],
                    "delta_acc_vs_baseline": (
                        pseudo_metrics["accuracy"] - baseline_metrics_ref["accuracy"]
                    ),
                    "delta_macro_f1_vs_baseline": (
                        pseudo_metrics["macro_f1"] - baseline_metrics_ref["macro_f1"]
                    ),
                    "model_path": str(pseudo_model_path),
                    "curves_path": str(pseudo_curves_path),
                    "pseudo_labels_path": str(filtered_pseudo_path),
                }
            )

    comparison_path = results_dir / "semi_supervised_comparison.csv"
    save_comparison_csv(comparison_rows, comparison_path)

    grouped_rows = defaultdict(list)
    for row in comparison_rows:
        grouped_rows[row["labeled_fraction"]].append(row)

    conclusion_lines = []
    for labeled_fraction, rows in grouped_rows.items():
        pseudo_rows = [
            row
            for row in rows
            if row["scenario"] == "pseudo_label_training" and row["test_macro_f1"] != ""
        ]
        if not pseudo_rows:
            conclusion_lines.append(
                f"{fraction_to_tag(labeled_fraction)}: no pseudo-label threshold produced trainable data."
            )
            continue

        best_row = max(pseudo_rows, key=lambda row: float(row["test_macro_f1"]))
        delta_f1 = float(best_row["delta_macro_f1_vs_baseline"])
        delta_acc = float(best_row["delta_acc_vs_baseline"])
        threshold = float(best_row["threshold"])
        if delta_f1 > 0:
            conclusion_lines.append(
                f"{fraction_to_tag(labeled_fraction)}: best threshold {threshold:.2f} "
                f"improved macro-F1 by {delta_f1:.4f} and accuracy by {delta_acc:.4f}."
            )
        else:
            conclusion_lines.append(
                f"{fraction_to_tag(labeled_fraction)}: best threshold {threshold:.2f} "
                f"did not improve macro-F1 (delta {delta_f1:.4f})."
            )

    conclusion_path = results_dir / "semi_supervised_conclusion.txt"
    with open(conclusion_path, "w") as f:
        f.write("\n".join(conclusion_lines) + "\n")

    print(f"Saved comparison table to: {comparison_path}")
    print(f"Saved conclusion to: {conclusion_path}")


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
    enable_pseudo_labeling = get_env_flag("ENABLE_PSEUDO_LABELING", default=False)

    print(f"Using device: {device}")
    print(
        f"Settings | batch_size={batch_size}, num_workers={num_workers}, "
        f"pin_memory={pin_memory}, amp=False, random_aug=False, "
        f"pseudo_labeling={enable_pseudo_labeling}"
    )

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    if enable_pseudo_labeling:
        run_limited_label_experiments(
            dataset_dir=dataset_dir,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            results_dir=results_dir,
        )
    else:
        run_full_supervised_training(
            dataset_dir=dataset_dir,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            results_dir=results_dir,
        )

    elapsed_seconds = time.time() - start_time
    print(f"Total runtime: {elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
