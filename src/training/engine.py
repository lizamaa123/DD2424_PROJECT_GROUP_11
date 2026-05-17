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


def train_consistency_regularized_model(
    model,
    labeled_loader,
    unlabeled_loader,
    test_loader,
    device,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    confidence_threshold: float = 0.8,
    lambda_u: float = 1.0,
    freeze_batchnorm: bool = True,
):
    if len(labeled_loader) == 0:
        raise ValueError("labeled_loader is empty.")
    if len(unlabeled_loader) == 0:
        raise ValueError("unlabeled_loader is empty.")

    supervised_criterion = nn.CrossEntropyLoss()
    unsupervised_criterion = nn.CrossEntropyLoss(reduction="none")
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
        "train_supervised_loss": [],
        "train_unsupervised_loss": [],
        "pseudo_keep_ratio": [],
        "test_loss": [],
        "test_acc": [],
        "test_macro_f1": [],
    }
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        if freeze_batchnorm:
            model.apply(set_batchnorm_eval)

        running_total_loss = 0.0
        running_supervised_loss = 0.0
        running_unsupervised_loss = 0.0
        running_keep = 0
        running_unlabeled = 0
        step_count = 0

        unlabeled_iterator = iter(unlabeled_loader)
        progress_bar = tqdm(labeled_loader, desc=f"Consistency epoch {epoch:02d}/{epochs}", leave=False)

        for labeled_images, labels in progress_bar:
            try:
                unlabeled_weak, unlabeled_strong = next(unlabeled_iterator)
            except StopIteration:
                unlabeled_iterator = iter(unlabeled_loader)
                unlabeled_weak, unlabeled_strong = next(unlabeled_iterator)

            labeled_images = labeled_images.to(device, non_blocking=(device.type == "cuda"))
            labels = labels.to(device, non_blocking=(device.type == "cuda"))
            unlabeled_weak = unlabeled_weak.to(device, non_blocking=(device.type == "cuda"))
            unlabeled_strong = unlabeled_strong.to(device, non_blocking=(device.type == "cuda"))

            optimizer.zero_grad()

            supervised_logits = model(labeled_images)
            supervised_loss = supervised_criterion(supervised_logits, labels)

            with torch.no_grad():
                weak_logits = model(unlabeled_weak)
                weak_probs = torch.softmax(weak_logits, dim=1)
                confidences, pseudo_labels = weak_probs.max(dim=1)
                mask = confidences >= confidence_threshold

            strong_logits = model(unlabeled_strong)
            unsupervised_loss_per_sample = unsupervised_criterion(strong_logits, pseudo_labels)
            mask_float = mask.float()
            unsupervised_loss = (
                (unsupervised_loss_per_sample * mask_float).sum()
                / mask_float.sum().clamp(min=1.0)
            )

            loss = supervised_loss + lambda_u * unsupervised_loss
            loss.backward()
            optimizer.step()

            running_total_loss += loss.item()
            running_supervised_loss += supervised_loss.item()
            running_unsupervised_loss += unsupervised_loss.item()
            running_keep += int(mask.sum().item())
            running_unlabeled += int(mask.numel())
            step_count += 1

            keep_ratio_step = float(mask_float.mean().item())
            progress_bar.set_postfix(
                sup=f"{supervised_loss.item():.4f}",
                unsup=f"{unsupervised_loss.item():.4f}",
                keep=f"{keep_ratio_step:.2%}",
            )

        train_loss = running_total_loss / max(1, step_count)
        train_supervised_loss = running_supervised_loss / max(1, step_count)
        train_unsupervised_loss = running_unsupervised_loss / max(1, step_count)
        pseudo_keep_ratio = running_keep / max(1, running_unlabeled)

        detailed_metrics = evaluate_detailed(model, test_loader, supervised_criterion, device)
        test_loss = detailed_metrics["loss"]
        test_acc = detailed_metrics["accuracy"]
        test_macro_f1 = detailed_metrics["macro_f1"]

        history["train_loss"].append(train_loss)
        history["train_supervised_loss"].append(train_supervised_loss)
        history["train_unsupervised_loss"].append(train_unsupervised_loss)
        history["pseudo_keep_ratio"].append(pseudo_keep_ratio)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["test_macro_f1"].append(test_macro_f1)

        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        tqdm.write(
            f"Consistency epoch {epoch:02d}/{epochs} | "
            f"threshold={confidence_threshold:.2f}, lambda_u={lambda_u:.2f} | "
            f"Train loss: {train_loss:.4f}, sup: {train_supervised_loss:.4f}, "
            f"unsup: {train_unsupervised_loss:.4f}, keep_ratio: {pseudo_keep_ratio:.2%} | "
            f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4%}, "
            f"Test macro-F1: {test_macro_f1:.4f}"
        )

    print(
        f"Best consistency test accuracy @ threshold={confidence_threshold:.2f}: "
        f"{best_acc:.4%}"
    )
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
