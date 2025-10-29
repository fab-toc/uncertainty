import random
import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchvision.models import ResNet, ResNet18_Weights, resnet18


def seed_everything(seed: int = 0) -> None:
    """
    Sets random seeds for reproducibility across all libraries.

    Ensures deterministic behavior for random number generation in Python's random module, NumPy, PyTorch (CPU and CUDA), and cuDNN operations.

    Args:
        seed (int, optional): Desired seed value. Defaults to 0.

    Example:
        >>> seed_everything(42)
        >>> torch.rand(3)  # Will always produce the same output
        tensor([0.8823, 0.9150, 0.3829])
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(
    model: Module,
    train_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
    criterion: Module,
    optimizer: Optimizer,
    epochs: int = 50,
    scheduler: Optional[LRScheduler] = None,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
) -> tuple[Module, list[float], list[float], list[float]]:
    """
    Trains a PyTorch model with validation monitoring and early stopping.

    Trains the model for a specified number of epochs, evaluates on validation data after each epoch, and saves the model state with the best validation accuracy.

    The model is trained in-place and the best weights are loaded before returning.

    Args:
        model (Module): Neural network model to train. Will be moved to the specified device.
        train_loader (DataLoader): DataLoader providing batches of training data as (inputs, labels).
        validation_loader (DataLoader): DataLoader providing batches of validation data.
        criterion (Module): Loss function (e.g. CrossEntropyLoss()). Must be compatible with the model's output and target format. Defaults to `CrossEntropyLoss()`.
        optimizer (Optimizer): Optimization algorithm (e.g., torch.optim.Adam(model.parameters())).
        epochs (int, optional): Number of training epochs to run. Defaults to 50.
        scheduler (LRScheduler, optional): Optional learning rate scheduler that updates after each epoch. Defaults to None.
        device (torch.device, optional): Device to perform training on. Defaults to `torch.device("cpu")`.
        verbose (bool, optional): Whether to print training progress. Defaults to True.

    Returns:
        tuple: A 4-tuple containing:
            - model (Module): Trained model with best validation weights loaded.
            - train_losses (list[float]): Average training loss per epoch.
            - validation_losses (list[float]): Average validation loss per epoch.
            - validation_accuracies (list[float]): Validation accuracy per epoch.

    Example:
        >>> model = MyModel()
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
        >>> criterion = nn.CrossEntropyLoss()
        >>> trained_model, train_loss, val_loss, val_acc = train_model(
        ...     model, train_loader, val_loader, epochs=10,
        ...     criterion=criterion, optimizer=optimizer
        ... )
        Epoch [1/10] | Train Loss: 0.6543 | Val Loss: 0.5234 | Val Acc: 78.45% | Time: 12.34s
    """

    model = model.to(device)
    criterion = criterion.to(device)  # some criterion have parameters

    # Track the best model state
    best_model_state: Optional[dict[str, torch.Tensor]] = None
    best_accuracy: float = 0.0

    # Initialize lists to store losses and accuracies over epochs
    train_losses: list[float] = []
    validation_losses: list[float] = []
    validation_accuracies: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()  # set the model in training mode

        epoch_loss: float = 0.0
        n_samples: int = 0

        t0: float = time.time()

        # Iterate over batches
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()  # zero the gradients for safety
            logits = model(x)  # forward pass
            loss = criterion(
                logits, y
            )  # compute loss between the predicted and true labels

            loss.backward()  # backpropagate the gradients
            optimizer.step()  # update model parameters with gradient descent

            # Update the current epoch loss
            # "loss.item()" is the loss averaged over the batch, so multiply it with the current batch size to get the total batch loss
            epoch_loss += loss.item() * x.size(0)
            n_samples += x.size(0)

        # Update learning rate scheduler if provided
        if scheduler is not None:
            scheduler.step()

        # Calculate average training loss
        avg_train_loss = epoch_loss / n_samples
        train_losses.append(avg_train_loss)

        # Evaluate on validation set
        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, criterion, device
        )
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)

        if verbose:
            print(
                f"Epoch [{epoch}/{epochs}] | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {validation_loss:.4f} | "
                f"Val Acc: {validation_accuracy * 100:.2f}% | "
                f"Time: {time.time() - t0:.2f}s"
            )

        # Save best model based on validation accuracy
        if validation_accuracy > best_accuracy:
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            best_accuracy = validation_accuracy

    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, validation_losses, validation_accuracies


@torch.no_grad()
def evaluate(
    model: Module,
    loader: DataLoader[Any],
    criterion: Module = nn.CrossEntropyLoss(),
    device: torch.device = torch.device("cpu"),
) -> tuple[float, float]:
    """
    Evaluates a PyTorch model on a dataset without computing gradients.

    Computes the average loss and classification accuracy over all batches in the provided DataLoader.

    The model is set to evaluation mode and no gradients are computed (via @torch.no_grad() decorator).

    Args:
        model: Neural network model to evaluate. Will be moved to the specified device.
        loader: DataLoader providing batches of evaluation data as (inputs, labels).
        criterion: Loss function used to compute the loss. Must match the training criterion. Defaults to `CrosEntropyLoss()`.
        device: Device to perform evaluation on. Defaults to torch.device("cpu").

    Returns:
        tuple: A 2-tuple containing:
            - avg_loss (float): Average loss over all samples.
            - accuracy (float): Classification accuracy in [0, 1].

    Example:
        >>> model = MyModel()
        >>> criterion = nn.CrossEntropyLoss()
        >>> test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        >>> print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc*100:.2f}%")
        Test Loss: 0.3421, Test Accuracy: 89.34%

    Note:
        This function does not modify the model's state. It automatically sets the model to eval mode but does not restore the previous mode.
    """

    model = model.to(device)
    criterion = criterion.to(device)  # some criterion have parameters

    model.eval()  # set the model in evaluation mode

    # Initialize metrics
    total_loss: float = 0.0
    total_correct: int = 0
    n_samples: int = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)  # forward pass
        loss = criterion(
            logits, y
        )  # compute loss between the predicted and true labels

        # "loss.item()" is the loss averaged over the batch, so multiply it with the current batch size to get the total batch loss
        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        n_samples += y.size(0)

    avg_loss = total_loss / n_samples
    accuracy = total_correct / n_samples

    return avg_loss, accuracy


def make_resnet18(n_classes: int, pretrained: bool = True) -> ResNet:
    """
    Creates a ResNet18 model with a custom classification head.

    Loads a ResNet18 architecture and replaces the final fully connected layer to output the specified number of classes. Optionally initializes with ImageNet pre-trained weights.

    Args:
        n_classes: Number of output classes for the classification task.
        pretrained: Whether to use ImageNet-1K pre-trained weights. If False, uses random initialization. Defaults to True.

    Returns:
        ResNet: ResNet18 model with modified final layer. The model is returned on CPU and should be moved to the desired device before use.

    Example:
        >>> # Create a ResNet18 for 10-class classification with pre-trained weights
        >>> model = make_resnet18(n_classes=10, pretrained=True)
        >>> model = model.to('cuda')
        >>>
        >>> # Create from scratch (no pre-training)
        >>> model_scratch = make_resnet18(n_classes=100, pretrained=False)

    Note:
        The pre-trained weights are trained on ImageNet-1K with 1000 classes. Only the final layer is replaced; all other layers retain their weights if pretrained = True.
    """

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None

    model = resnet18(weights=weights, progress=True)
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    return model
