import torch
from torch import nn
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt

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
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr, weight_decay=weight_decay)

    # track metrics and best model state
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
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
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # save metrics to history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        # check if this is the best model so far
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        tqdm.write(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4%} | "
            f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4%}"
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