import copy
import csv
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
import numpy as np


def set_batchnorm_eval(module):
    """
    Put BatchNorm layers in eval mode.

    Useful for linear probing, where the pretrained backbone is frozen.
    Usually do not use this for fine-tuning.
    """
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
    freeze_batchnorm: bool = False,
):
    criterion = nn.CrossEntropyLoss()

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]

    if len(trainable_parameters) == 0:
        raise ValueError("No trainable parameters found. Check requires_grad settings.")

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=lr,
        weight_decay=weight_decay,
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()

        # For linear probing, keep BatchNorm fixed.
        # For fine-tuning, leave BatchNorm trainable/normal.
        if freeze_batchnorm:
            model.apply(set_batchnorm_eval)

        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:02d}/{epochs}",
            leave=False,
        )

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

        test_loss, test_acc = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        tqdm.write(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4%} | "
            f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4%}"
        )

    print(f"Best test accuracy: {best_acc:.4%}")

    model.load_state_dict(best_model_wts)

    return model, history
def evaluate_per_class(
    model,
    loader,
    device,
    num_classes: int = 37,
):
    """
    Compute per-class accuracy and per-class F1.
    """

    model.eval()

    true_positives = torch.zeros(num_classes)
    false_positives = torch.zeros(num_classes)
    false_negatives = torch.zeros(num_classes)
    correct_per_class = torch.zeros(num_classes)
    total_per_class = torch.zeros(num_classes)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=(device.type == "cuda"))
            labels = labels.to(device, non_blocking=(device.type == "cuda"))

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            labels_cpu = labels.cpu()
            preds_cpu = preds.cpu()

            for true_label, pred_label in zip(labels_cpu, preds_cpu):
                true_label = int(true_label)
                pred_label = int(pred_label)

                total_per_class[true_label] += 1

                if true_label == pred_label:
                    correct_per_class[true_label] += 1
                    true_positives[true_label] += 1
                else:
                    false_positives[pred_label] += 1
                    false_negatives[true_label] += 1

    per_class_accuracy = correct_per_class / torch.clamp(total_per_class, min=1)

    precision = true_positives / torch.clamp(true_positives + false_positives, min=1)
    recall = true_positives / torch.clamp(true_positives + false_negatives, min=1)

    per_class_f1 = 2 * precision * recall / torch.clamp(precision + recall, min=1e-8)
    macro_f1 = per_class_f1.mean().item()

    return {
        "per_class_accuracy": per_class_accuracy.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "macro_f1": macro_f1,
        "total_per_class": total_per_class.tolist(),
    }


def save_per_class_metrics(metrics, save_path):
    """
    Save per-class accuracy and F1 to a CSV file.
    """

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["class_id", "test_samples", "accuracy", "f1"])

        for class_id, (acc, f1, total) in enumerate(
            zip(
                metrics["per_class_accuracy"],
                metrics["per_class_f1"],
                metrics["total_per_class"],
            )
        ):
            writer.writerow(
                [
                    class_id,
                    int(total),
                    f"{acc:.4f}",
                    f"{f1:.4f}",
                ]
            )

        writer.writerow([])
        writer.writerow(["macro_f1", f"{metrics['macro_f1']:.4f}"])

def save_training_curves(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    ax1.plot(epochs, history["test_loss"], label="Test Loss", marker="o")
    ax1.set_title("Training and Test Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history["train_acc"], label="Train Accuracy", marker="o")
    ax2.plot(epochs, history["test_acc"], label="Test Accuracy", marker="o")
    ax2.set_title("Training and Test Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compute_confusion_matrix(model, loader, device, num_classes: int = 37):
    """
    Compute confusion matrix.

    Rows = true labels
    Columns = predicted labels
    """
    model.eval()

    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=(device.type == "cuda"))
            labels = labels.to(device, non_blocking=(device.type == "cuda"))

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            labels_cpu = labels.cpu()
            preds_cpu = preds.cpu()

            for true_label, pred_label in zip(labels_cpu, preds_cpu):
                confusion[int(true_label), int(pred_label)] += 1

    return confusion


def save_confusion_matrix_plot(confusion_matrix, save_path):
    """
    Save confusion matrix as a figure.
    """
    matrix = confusion_matrix.numpy()

    plt.figure(figsize=(12, 10))
    plt.imshow(matrix)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_normalized_confusion_matrix_plot(confusion_matrix, save_path):
    """
    Save row-normalized confusion matrix.

    This is often easier to read because each row sums to 1.
    """
    matrix = confusion_matrix.numpy().astype(float)

    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = matrix / np.maximum(row_sums, 1)

    plt.figure(figsize=(12, 10))
    plt.imshow(normalized)
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()