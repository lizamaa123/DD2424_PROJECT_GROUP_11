import torch
from torch import nn
from tqdm import tqdm

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

        tqdm.write(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4%} | "
            f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4%}"
        )