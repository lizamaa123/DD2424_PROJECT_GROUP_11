import torch
from torch import nn
from tqdm import tqdm
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score

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

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def evaluate_detailed(model, loader, criterion, device, num_classes: int = 37):
    model.eval()
    total_loss = 0.0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=(device.type == "cuda"))
            labels = labels.to(device, non_blocking=(device.type == "cuda"))

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            total += labels.size(0)

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels))) if total > 0 else 0.0
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) if total > 0 else 0.0
    per_class_f1 = (
        f1_score(all_labels, all_preds, labels=list(range(num_classes)), average=None, zero_division=0)
        if total > 0
        else np.zeros(num_classes)
    )

    cm = (
        confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
        if total > 0
        else np.zeros((num_classes, num_classes), dtype=int)
    )
    class_totals = cm.sum(axis=1)
    diagonal = np.diag(cm)
    per_class_acc = np.divide(
        diagonal,
        class_totals,
        out=np.zeros_like(diagonal, dtype=float),
        where=class_totals > 0,
    )

    return {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class_accuracy": per_class_acc.tolist(),
        "per_class_f1": per_class_f1.tolist(),
    }


def generate_pseudo_labels(model, unlabeled_loader, device, thresholds=None):
    model.eval()
    pseudo_samples = []
    total = 0
    thresholds = thresholds or [0.95]

    with torch.no_grad():
        progress_bar = tqdm(unlabeled_loader, desc="Pseudo-labeling", leave=False)
        for images, image_paths in progress_bar:
            images = images.to(device, non_blocking=(device.type == "cuda"))

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = probabilities.max(dim=1)

            for image_path, prediction, confidence in zip(
                image_paths,
                predictions.cpu().tolist(),
                confidences.cpu().tolist(),
            ):
                total += 1
                pseudo_samples.append((image_path, int(prediction), float(confidence)))

    threshold_stats = {}
    for threshold in thresholds:
        kept = sum(1 for _path, _label, conf in pseudo_samples if conf >= threshold)
        keep_ratio = (kept / total) if total > 0 else 0.0
        threshold_stats[float(threshold)] = {
            "total": total,
            "kept": kept,
            "keep_ratio": keep_ratio,
        }
        tqdm.write(
            f"Pseudo-label stats | threshold={threshold:.2f}, "
            f"kept {kept}/{total} ({keep_ratio:.2%})"
        )

    return pseudo_samples, threshold_stats


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
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]

    if len(trainable_parameters) == 0:
        raise ValueError("No trainable parameters found. Check requires_grad settings.")

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=lr,
        weight_decay=weight_decay,
    )

    # track metrics and best model state
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "test_macro_f1": [],
    }
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

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
        detailed_metrics = evaluate_detailed(model, test_loader, criterion, device)
        test_loss = detailed_metrics["loss"]
        test_acc = detailed_metrics["accuracy"]
        test_macro_f1 = detailed_metrics["macro_f1"]

        # save metrics to history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["test_macro_f1"].append(test_macro_f1)

        # check if this is the best model so far
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        tqdm.write(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4%} | "
            f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4%}, "
            f"Test macro-F1: {test_macro_f1:.4f}"
        )

    # load the best weights back into the model before returning
    print(f"Best test accuracy: {best_acc:.4%}")
    model.load_state_dict(best_model_wts)
    return model, history


# generate and save training curves
def save_training_curves(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss plots
    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    ax1.plot(epochs, history["test_loss"], label="Test Loss", marker="o")
    ax1.set_title("Training and Test Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plots
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
