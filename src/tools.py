import csv
import json
import random
import time
from pathlib import Path
from typing import Any, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from matplotlib.figure import Figure
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchvision.models import ResNet, ResNet18_Weights, resnet18


def seed_everything(seed: int = 0) -> None:
    """Set random seeds for reproducibility across all libraries.

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
    file_path: str,
    epochs: int = 50,
    scheduler: Optional[LRScheduler] = None,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
    save_plots: bool = True,
    config: Optional[dict[str, Any]] = None,
) -> tuple[Module, list[float], list[float], list[float]]:
    """Train a PyTorch model with validation monitoring and early stopping.

    Trains the model for a specified number of epochs, evaluates on validation data after each epoch, and saves the model state with the best validation accuracy.

    The model is trained in-place and the best weights are loaded before returning.

    Args:
        model (Module): Neural network model to train. Will be moved to the specified device.
        train_loader (DataLoader): DataLoader providing batches of training data as (inputs, labels).
        validation_loader (DataLoader): DataLoader providing batches of validation data.
        criterion (Module): Loss function (e.g. CrossEntropyLoss()). Must be compatible with the model's output and target format. Defaults to `CrossEntropyLoss()`.
        optimizer (Optimizer): Optimization algorithm (e.g., torch.optim.Adam(model.parameters())).
        filename (str): Model filename to save the model dict.
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
        ...     criterion=criterion, optimizer=optimizer,
        ...     filename="model.pt", device=torch.device("cuda")
        ... )
        Epoch [1/10] | Train Loss: 0.6543 | Val Loss: 0.5234 | Val Acc: 78.45% | Time: 12.34s
    """

    model = model.to(device)
    criterion = criterion.to(device)  # some criterion have parameters

    # Setup metrics tracker
    run_dir = Path(file_path).parent
    tracker = MetricsTracker(run_dir, config=config)

    # Track the best model state
    best_model_state: Optional[dict[str, torch.Tensor]] = {
        k: v.cpu().clone() for k, v in model.state_dict().items()
    }
    best_loss: float = float("inf")

    # Initialize lists to store losses and accuracies over epochs
    train_losses: list[float] = []
    validation_losses: list[float] = []
    validation_accuracies: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()  # set the model in training mode

        epoch_loss: float = 0.0
        epoch_correct: int = 0
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
            epoch_correct += (logits.argmax(1) == y).sum().item()
            n_samples += x.size(0)

        # Compute training metrics
        avg_train_loss = epoch_loss / n_samples
        train_accuracy = epoch_correct / n_samples
        train_losses.append(avg_train_loss)

        # Evaluate on validation set
        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, criterion, device
        )
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Update learning rate scheduler if provided
        if scheduler is not None:
            scheduler.step()

        # Time taken
        epoch_time = time.time() - t0

        # Log to CSV
        tracker.log_epoch(
            epoch=epoch,
            train_loss=avg_train_loss,
            train_acc=train_accuracy,
            val_loss=validation_loss,
            val_acc=validation_accuracy,
            learning_rate=current_lr,
            time_seconds=epoch_time,
        )

        if verbose:
            print(
                f"Epoch [{epoch}/{epochs}] | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Train Acc: {train_accuracy * 100:.2f}% | "
                f"Val Loss: {validation_loss:.4f} | "
                f"Val Acc: {validation_accuracy * 100:.2f}% | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.2f}s"
            )

        # Save best model based on validation accuracy
        if validation_loss < best_loss:
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            best_loss = validation_loss

    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, file_path)

    # Close tracker and generate plots
    tracker.close()

    if save_plots:
        tracker.save_plots()

    if verbose:
        print("\n" + "=" * 70)
        print("Training completed!")
        print(f"Best validation loss: {best_loss:.4f}")
        print(f"Best validation accuracy: {max(validation_accuracies) * 100:.2f}%")
        print(f"Model saved to: {file_path}")
        print(f"Metrics saved to: {tracker.csv_path}")
        print(f"Plots saved to: {tracker.plots_dir}/")
        print("=" * 70 + "\n")

    return model, train_losses, validation_losses, validation_accuracies


@torch.no_grad()
def evaluate(
    model: Module,
    loader: DataLoader[Any],
    criterion: Module,
    device: torch.device = torch.device("cpu"),
) -> tuple[float, float]:
    """Evaluate a PyTorch model on a dataset without computing gradients.

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

    model = model.eval()  # set the model in evaluation mode

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


def make_resnet18(num_classes: int, pretrained: bool = True) -> ResNet:
    """Create a ResNet18 model with a custom classification head.

    Loads a ResNet18 architecture and replaces the final fully connected layer to output the specified number of classes. Optionally initializes with ImageNet pre-trained weights.

    Args:
        num_classes: Number of output classes for the classification task.
        pretrained: Whether to use ImageNet-1K pre-trained weights. If False, uses random initialization. Defaults to True.

    Returns:
        ResNet: ResNet18 model with modified final layer. The model is returned on CPU and should be moved to the desired device before use.

    Example:
        >>> # Create a ResNet18 for 10-class classification with pre-trained weights
        >>> model = make_resnet18(num_classes=10, pretrained=True)
        >>> model = model.to('cuda')
        >>>
        >>> # Create from scratch (no pre-training)
        >>> model_scratch = make_resnet18(num_classes=100, pretrained=False)

    Note:
        The pre-trained weights are trained on ImageNet-1K with 1000 classes. Only the final layer is replaced; all other layers retain their weights if pretrained = True.
    """

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None

    model = resnet18(weights=weights, progress=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def get_model_name(
    model: str = "resnet18",
    pretrained: bool = False,
    shuffle: bool = True,
    seed: int = 0,
    normalization: Literal["MNIST", "ImageNet"] = "MNIST",
    model_number: Optional[int] = None,
) -> str:
    """Generate a standardized model filename based on training configuration.

    Creates a descriptive filename that encodes all important hyperparameters for reproducibility and experiment tracking.

    Args:
        model (str, optional): Model architecture name. Defaults to "resnet18".
        pretrained (bool, optional): Whether pretrained weights were used.
            Defaults to False.
        shuffle (bool, optional): Whether data was shuffled during training.
            Defaults to True.
        seed (int, optional): Random seed used for training. Defaults to 0.
        normalization (Literal["MNIST", "ImageNet"], optional): Normalization
            strategy used. Defaults to "MNIST".

    Returns:
        str: Formatted model name encoding all parameters.

    Example:
        >>> get_model_name(pretrained=True, shuffle=False, seed=42, normalization="MNIST")
        'resnet18_pt_normMNIST_seed42'
        >>> get_model_name(pretrained=False, shuffle=True, seed=0, normalization="ImageNet")
        'resnet18_normImageNet_shuffle_seed0'

    Note:
        The filename format is: {model}[_pre-trained]_norm{normalization}[_shuffle]_seed{seed}
    """
    return f"{model}{'_pre-trained' if pretrained else ''}_norm{normalization}_{'shuffle' if shuffle else 'no-shuffle'}_seed{seed}{f'_{model_number}' if model_number is not None else ''}"


def init_weights(module: nn.Module, mode: str = "default") -> None:
    """Initialize weights of a neural network module.

    Applies different weight initialization strategies to convolutional and linear layers. Biases are always initialized to zero.

    Args:
        module (nn.Module): PyTorch module to initialize.
        mode (str, optional): Initialization strategy. Options are:
            - "default": No initialization (keep PyTorch defaults or pretrained)
            - "kaiming": Kaiming/He initialization for ReLU networks
            - "xavier": Xavier/Glorot initialization for tanh/sigmoid networks
            - "orthogonal": Orthogonal initialization for RNNs
            Defaults to "default".

    Raises:
        ValueError: If mode is not one of the supported strategies.

    Example:
        >>> model = resnet18()
        >>> model.apply(lambda m: init_weights(m, mode="kaiming"))

    Note:
        Only Conv2d and Linear layers are affected. Other layer types are ignored.
    """

    # default = ne rien toucher (poids pré-entraînés ou init PyTorch)
    if mode == "default":
        return

    if isinstance(module, (nn.Conv2d, nn.Linear)):
        if mode == "kaiming":
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif mode == "xavier":
            nn.init.xavier_uniform_(module.weight)
        elif mode == "orthogonal":
            nn.init.orthogonal_(module.weight, gain=1.0)
        else:
            raise ValueError(f"init_mode inconnu pour init_weights: {mode}")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def load_model(model: nn.Module, file_path: str, device: torch.device) -> nn.Module:
    """Load a saved model from disk.

    Loads the model's state dictionary from a file and moves it to the specified device.

    Args:
        model (nn.Module): Model instance to load weights into.
        file_path (str): Path to the saved model state dict (.pt file).
        device (torch.device): Device to load the model onto.

    Returns:
        nn.Module: Model with loaded weights on the specified device.

    Example:
        >>> model = resnet18(num_classes=10)
        >>> model = load_model(model, "models/best_model.pt", torch.device("cuda"))

    Note:
        The model architecture must match the saved state dict.
    """
    model.load_state_dict(torch.load(file_path))
    return model.to(device)


def get_dataset_estimates(set: torch.utils.data.Dataset):
    """Compute mean and standard deviation of a dataset.

    Computes the global mean and standard deviation across all samples in the dataset. Useful for data normalization.

    Args:
        dataset (torch.utils.data.Dataset): PyTorch dataset that returns (image, label) tuples where images are tensors.

    Returns:
        tuple: A 2-tuple containing:
            - mean (float): Mean pixel value across the entire dataset.
            - std (float): Standard deviation of pixel values.

    Example:
        >>> mnist_train = datasets.MNIST(root, train=True, transform=ToTensor())
        >>> mean, std = get_dataset_estimates(mnist_train)
        >>> print(f"MNIST mean={mean:.4f}, std={std:.4f}")
        MNIST mean=0.1307, std=0.3081

    Note:
        This function loads all data into memory. For large datasets, consider sampling or computing statistics incrementally.
    """
    data = torch.cat([x for x, _ in set], dim=0)  # concatenate all data
    mean = data.mean().item()
    std = data.std().item()
    return mean, std


def get_data_transforms(
    data_root: str,
    resize_value: int,
    normalization: Literal["MNIST", "ImageNet"] = "MNIST",
):
    """Create data transformation pipeline for MNIST preprocessing.

    Builds a transformation pipeline that resizes MNIST images to `resize_value x resize_value`, converts grayscale to RGB (3 channels), and applies normalization.

    Supports both MNIST-specific normalization and ImageNet normalization for transfer learning.

    Args:
        data_root (str): Root directory containing the MNIST dataset.
        resize_value (int): Target size to resize images to (e.g., 32 for 32x32).
        normalization (Literal["MNIST", "ImageNet"], optional): Normalization strategy to use. "MNIST" computes statistics from the dataset, "ImageNet" uses ImageNet statistics for transfer learning. Defaults to "MNIST".

    Returns:
        transforms.Compose: Composed transformation pipeline.

    Example:
        >>> transforms = get_data_transforms("./data", resize_value=32, normalization="MNIST")
        >>> dataset = datasets.MNIST("./data", train=True, transform=transforms)
    """

    # Compute mean & std of MNIST dataset
    mean, std = get_dataset_estimates(
        torchvision.datasets.MNIST(
            data_root,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
    )

    # Duplicate on 3 channels for the model
    MNIST_MEAN: list[float] = [mean, mean, mean]
    MNIST_STD: list[float] = [std, std, std]

    IMAGENET_MEAN: list[float] = ResNet18_Weights.IMAGENET1K_V1.transforms().mean
    IMAGENET_STD: list[float] = ResNet18_Weights.IMAGENET1K_V1.transforms().std

    tf = [
        transforms.Resize((resize_value, resize_value)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]

    tf.append(
        transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD)
        if normalization == "MNIST"
        else transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    )

    return transforms.Compose(tf)


def get_loaders(
    train_data: torch.utils.data.Dataset,
    validation_data: torch.utils.data.Dataset,
    test_data: torch.utils.data.Dataset,
    shuffle: bool,
    batch_size: int,
    drop_last: bool,
    num_workers: int = 0,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    """Create DataLoaders for training, validation, and test datasets.

    Creates three DataLoaders with consistent settings for batch size, shuffling, and parallel data loading.

    Automatically enables pin_memory for GPU training and persistent_workers for efficiency.

    Args:
        train_data (torch.utils.data.Dataset): Training dataset.
        validation_data (torch.utils.data.Dataset): Validation dataset.
        test_data (torch.utils.data.Dataset): Test dataset.
        shuffle (bool): Whether to shuffle the data at every epoch.
        batch_size (int): Number of samples per batch.
        drop_last (bool): Whether to drop the last incomplete batch.
        num_workers (int, optional): Number of subprocesses for data loading.
            Defaults to 0 (main process only).

    Returns:
        tuple: A 3-tuple containing:
            - train_loader: DataLoader for training data.
            - validation_loader: DataLoader for validation data.
            - test_loader: DataLoader for test data.

    Example:
        >>> train_loader, val_loader, test_loader = get_loaders(
        ...     train_data, val_data, test_data,
        ...     shuffle=True, batch_size=128, drop_last=True, num_workers=4
        ... )

    Note:
        pin_memory is always True for faster GPU transfers. persistent_workers is enabled when num_workers > 0 to avoid recreating workers each epoch.
    """
    train_loader = DataLoader(
        train_data,
        shuffle=shuffle,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    validation_loader = DataLoader(
        validation_data,
        shuffle=shuffle,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_data,
        shuffle=shuffle,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return train_loader, validation_loader, test_loader


def visualize_predictions(
    model: Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device = torch.device("cpu"),
    num_samples: int = 20,
    seed: Optional[int] = None,
    figsize: tuple[int, int] = (20, 8),
) -> Figure:
    """Visualize model predictions on random samples from a dataset.

    Displays a grid of images with their true labels and predicted labels.
    Correct predictions are shown in green, incorrect ones in red.

    Args:
        model (Module): Trained PyTorch model to use for predictions.
        dataset (torch.utils.data.Dataset): Dataset to sample from (e.g., test set).
        device (torch.device, optional): Device to perform inference on.
            Defaults to torch.device("cpu").
        num_samples (int, optional): Number of random samples to visualize.
            Defaults to 20.
        seed (int, optional): Random seed for reproducible sampling. If None,
            sampling is non-deterministic. Defaults to None.
        figsize (tuple[int, int], optional): Figure size as (width, height) in inches.
            Defaults to (20, 8).

    Returns:
        Figure: Matplotlib figure containing the visualization grid.

    Example:
        >>> model = load_model(model, "models/best_model.pt", device)
        >>> fig = visualize_predictions(model, test_data, device, num_samples=20, seed=42)
        >>> plt.show()
        >>> # Or save the figure
        >>> fig.savefig("predictions.png", dpi=150, bbox_inches='tight')

    Note:
        - The model is automatically set to eval mode.
        - Images are denormalized for display if they were normalized during training.
        - Predictions show both the predicted class and confidence (softmax probability).
    """
    model = model.to(device)
    model.eval()

    # Utiliser get_random_samples pour échantillonnage cohérent
    if seed is not None:
        samples, indices = get_random_samples(
            dataset=dataset,
            set_size=len(dataset),
            seed=seed,
            num_samples=num_samples,
        )
    else:
        # Fallback sans seed
        indices = random.sample(range(len(dataset)), num_samples)
        samples = [dataset[idx] for idx in indices]

    # Create figure
    num_cols = 5
    num_rows = (num_samples + num_cols - 1) // num_cols  # Ceiling division
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten() if num_samples > 1 else [axes]

    with torch.no_grad():
        for idx, (ax, (image, true_label)) in enumerate(zip(axes, samples)):
            # Prepare image for model (add batch dimension)
            image_tensor = image.unsqueeze(0).to(device)

            # Get prediction
            logits = model(image_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_label = logits.argmax(dim=1).item()
            confidence = probs[0, pred_label].item()

            # Denormalize image for display
            img_display = image.permute(1, 2, 0).cpu().numpy()
            img_display = (img_display - img_display.min()) / (
                img_display.max() - img_display.min()
            )

            # Convert RGB back to grayscale for MNIST display
            if img_display.shape[2] == 3:
                img_display = img_display.mean(axis=2)

            # Display image
            ax.imshow(img_display, cmap="gray")
            ax.axis("off")

            # Set title with true and predicted labels
            is_correct = pred_label == true_label
            color = "green" if is_correct else "red"
            title = f"True: {true_label}\nPred: {pred_label} ({confidence:.2%})"
            ax.set_title(title, color=color, fontsize=10, fontweight="bold")

    # Hide unused subplots
    for ax in axes[len(samples) :]:
        ax.axis("off")

    plt.tight_layout()
    return fig


def visualize_predictions_with_uncertainty(
    models: list[Module],
    dataset: torch.utils.data.Dataset,
    device: torch.device = torch.device("cpu"),
    num_samples: int = 20,
    seed: Optional[int] = None,
    figsize: tuple[int, int] = (20, 10),
) -> Figure:
    """Visualize ensemble predictions with uncertainty estimation.

    Shows predictions from an ensemble of models with uncertainty bars representing
    the standard deviation across model predictions.

    Args:
        models (list[Module]): List of trained models (ensemble).
        dataset (torch.utils.data.Dataset): Dataset to sample from.
        device (torch.device, optional): Device for inference. Defaults to cpu.
        num_samples (int, optional): Number of images to display. Defaults to 20.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        figsize (tuple[int, int], optional): Figure size. Defaults to (20, 10).

    Returns:
        Figure: Matplotlib figure with predictions and uncertainty bars.

    Example:
        >>> fig = visualize_predictions_with_uncertainty(
        ...     models, test_data, device, num_samples=20, seed=42
        ... )
        >>> plt.show()

    Note:
        - Uses ensemble of models instead of MC Dropout.
        - Uncertainty comes from disagreement between models.
        - Higher standard deviation = higher uncertainty.
    """
    # Mettre tous les modèles en eval mode
    for model in models:
        model.to(device).eval()

    # Utiliser get_random_samples pour échantillonnage cohérent
    if seed is not None:
        samples, indices = get_random_samples(
            dataset=dataset,
            set_size=len(dataset),
            seed=seed,
            num_samples=num_samples,
        )
    else:
        # Fallback sans seed
        indices = random.sample(range(len(dataset)), num_samples)
        samples = [dataset[idx] for idx in indices]

    # Create figure with subplots for images and uncertainty bars
    num_cols = 5
    num_rows = (num_samples + num_cols - 1) // num_cols
    fig = plt.figure(figsize=figsize)

    # Create gridspec for better layout control
    gs = fig.add_gridspec(num_rows * 2, num_cols, hspace=0.4, wspace=0.3)

    with torch.no_grad():
        for plot_idx, (image, true_label) in enumerate(samples):
            row = (plot_idx // num_cols) * 2
            col = plot_idx % num_cols

            image_tensor = image.unsqueeze(0).to(device)

            # Prédictions de tous les modèles de l'ensemble
            ensemble_predictions = []
            for model in models:
                logits = model(image_tensor)
                probs = torch.softmax(logits, dim=1)
                ensemble_predictions.append(probs.cpu().numpy()[0])

            ensemble_predictions = np.array(
                ensemble_predictions
            )  # Shape: (num_models, num_classes)

            # Calculate statistics
            mean_probs = ensemble_predictions.mean(axis=0)
            std_probs = ensemble_predictions.std(axis=0)
            pred_label = mean_probs.argmax()
            confidence = mean_probs[pred_label]
            uncertainty = std_probs[pred_label]

            # Plot image
            ax_img = fig.add_subplot(gs[row, col])
            img_display = image.permute(1, 2, 0).cpu().numpy()
            img_display = (img_display - img_display.min()) / (
                img_display.max() - img_display.min()
            )
            if img_display.shape[2] == 3:
                img_display = img_display.mean(axis=2)

            ax_img.imshow(img_display, cmap="gray")
            ax_img.axis("off")

            is_correct = pred_label == true_label
            color = "green" if is_correct else "red"
            title = f"True: {true_label} | Pred: {pred_label}\n({confidence:.2%} ± {uncertainty:.2%})"
            ax_img.set_title(title, color=color, fontsize=9, fontweight="bold")

            # Plot uncertainty bars
            ax_bar = fig.add_subplot(gs[row + 1, col])
            num_classes = len(mean_probs)
            x_pos = np.arange(num_classes)

            bars = ax_bar.bar(
                x_pos,
                mean_probs,
                yerr=std_probs,
                capsize=3,
                alpha=0.7,
                color="steelblue",
                error_kw={"elinewidth": 1},
            )

            # Highlight predicted class
            bars[pred_label].set_color("green" if is_correct else "red")

            ax_bar.set_ylim(0, 1)
            ax_bar.set_xticks(x_pos)
            ax_bar.set_xticklabels(x_pos, fontsize=7)
            ax_bar.set_ylabel("Prob", fontsize=7)
            ax_bar.tick_params(axis="y", labelsize=7)
            ax_bar.grid(axis="y", alpha=0.3)

    plt.suptitle(
        f"Ensemble Predictions with Uncertainty ({len(models)} models)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    return fig


class MetricsTracker:
    """Track and save training metrics to CSV and generate plots.

    Automatically creates a structured directory with metrics CSV, config JSON,
    and visualization plots for a training run.

    Args:
        run_dir (str | Path): Directory to save all metrics and plots.
        config (dict[str, Any], optional): Hyperparameters and config to save
            as JSON. Defaults to None.

    Example:
        >>> tracker = MetricsTracker("models/resnet18_run1", config={"lr": 0.001})
        >>> for epoch in range(10):
        ...     tracker.log_epoch(epoch, train_loss=0.5, val_loss=0.4, val_acc=0.9)
        >>> tracker.save_plots()

    Attributes:
        run_dir (Path): Root directory for this training run.
        csv_path (Path): Path to metrics.csv file.
        plots_dir (Path): Directory containing generated plots.
        metrics (list[dict]): List of metric dictionaries for each epoch.
    """

    def __init__(self, run_dir: str | Path, config: Optional[dict[str, Any]] = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # CSV file for metrics
        self.csv_path = self.run_dir / "metrics.csv"
        self.csv_file = None
        self.csv_writer = None

        # Plots directory
        self.plots_dir = self.run_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

        # Save config
        if config is not None:
            config_path = self.run_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

        # Internal storage
        self.metrics: list[dict[str, Any]] = []
        self._init_csv()

    def _init_csv(self) -> None:
        """Initialize CSV file with headers."""
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "learning_rate",
                "time_seconds",
            ],
        )
        self.csv_writer.writeheader()
        self.csv_file.flush()

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        learning_rate: float = 0.0,
        time_seconds: float = 0.0,
    ) -> None:
        """Log metrics for one epoch.

        Args:
            epoch (int): Current epoch number (1-indexed).
            train_loss (float): Average training loss.
            train_acc (float): Training accuracy in [0, 1].
            val_loss (float): Average validation loss.
            val_acc (float): Validation accuracy in [0, 1].
            learning_rate (float, optional): Current learning rate. Defaults to 0.0.
            time_seconds (float, optional): Time taken for this epoch in seconds.
                Defaults to 0.0.

        Example:
            >>> tracker.log_epoch(1, 0.6, 0.75, 0.5, 0.82, lr=0.001, time_seconds=45.2)
        """
        row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "train_acc": f"{train_acc:.4f}",
            "val_loss": f"{val_loss:.6f}",
            "val_acc": f"{val_acc:.4f}",
            "learning_rate": f"{learning_rate:.6e}",
            "time_seconds": f"{time_seconds:.2f}",
        }

        self.csv_writer.writerow(row)
        self.csv_file.flush()
        self.metrics.append(row)

    def close(self) -> None:
        """Close CSV file."""
        if self.csv_file is not None:
            self.csv_file.close()

    def save_plots(self) -> None:
        """Generate and save all training plots.

        Creates three plots:
            - Loss curves (train + validation)
            - Accuracy curves (train + validation)
            - Learning rate schedule

        Example:
            >>> tracker.save_plots()  # After training completes
        """
        if not self.metrics:
            return

        df = pd.DataFrame(self.metrics)
        df["epoch"] = df["epoch"].astype(int)
        for col in ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]:
            df[col] = df[col].astype(float)

        # Plot 1: Loss curves
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df["epoch"], df["train_loss"], label="Train Loss", linewidth=2)
        ax.plot(df["epoch"], df["val_loss"], label="Val Loss", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(self.plots_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Plot 2: Accuracy curves
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df["epoch"], df["train_acc"] * 100, label="Train Accuracy", linewidth=2)
        ax.plot(df["epoch"], df["val_acc"] * 100, label="Val Accuracy", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(
            self.plots_dir / "accuracy_curves.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

        # Plot 3: Learning rate
        if df["learning_rate"].nunique() > 1:  # Only if LR changes
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df["epoch"], df["learning_rate"], linewidth=2, color="orange")
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("Learning Rate", fontsize=12)
            ax.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            fig.savefig(
                self.plots_dir / "learning_rate.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig)


def load_metrics(run_dir: str | Path) -> pd.DataFrame:
    """Load metrics from a training run's CSV file.

    Args:
        run_dir (str | Path): Directory containing metrics.csv.

    Returns:
        pd.DataFrame: DataFrame with columns: epoch, train_loss, train_acc,
            val_loss, val_acc, learning_rate, time_seconds.

    Example:
        >>> df = load_metrics("models/resnet18_run1")
        >>> print(df['val_acc'].max())  # Best validation accuracy
        0.9876

    Raises:
        FileNotFoundError: If metrics.csv does not exist in run_dir.
    """
    run_dir = Path(run_dir)
    csv_path = run_dir / "metrics.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"No metrics.csv found in {run_dir}")

    df = pd.read_csv(csv_path)

    # Convert types
    if "epoch" in df.columns:
        df["epoch"] = df["epoch"].astype(int)
    for col in [
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "learning_rate",
        "time_seconds",
    ]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return df


def load_config(run_dir: str | Path) -> dict[str, Any]:
    """Load configuration from a training run's JSON file.

    Args:
        run_dir (str | Path): Directory containing config.json.

    Returns:
        dict[str, Any]: Configuration dictionary with hyperparameters.

    Example:
        >>> config = load_config("models/resnet18_run1")
        >>> print(config['learning_rate'])
        0.0001

    Raises:
        FileNotFoundError: If config.json does not exist in run_dir.
    """
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found in {run_dir}")

    with open(config_path, "r") as f:
        return json.load(f)


def plot_metric_comparison(
    run_dirs: list[str | Path],
    metric: str = "val_acc",
    labels: Optional[list[str]] = None,
    title: Optional[str] = None,
    smooth_window: Optional[int] = None,
    save_path: Optional[str | Path] = None,
    figsize: tuple[int, int] = (12, 6),
) -> Figure:
    """Compare a metric across multiple training runs.

    Args:
        run_dirs (list[str | Path]): List of run directories to compare.
        metric (str, optional): Metric to plot. Must be a column in metrics.csv.
            Options: 'train_loss', 'train_acc', 'val_loss', 'val_acc'.
            Defaults to 'val_acc'.
        labels (list[str], optional): Labels for each run in the legend. If None,
            uses directory names. Defaults to None.
        title (str, optional): Plot title. If None, auto-generates. Defaults to None.
        smooth_window (int, optional): Window size for rolling average smoothing.
            If None, no smoothing is applied. Defaults to None.
        save_path (str | Path, optional): Path to save the figure. If None, figure
            is not saved. Defaults to None.
        figsize (tuple[int, int], optional): Figure size as (width, height).
            Defaults to (12, 6).

    Returns:
        Figure: Matplotlib figure object.

    Example:
        >>> fig = plot_metric_comparison(
        ...     run_dirs=["models/run1", "models/run2", "models/run3"],
        ...     metric='val_acc',
        ...     labels=["Pretrained", "From Scratch", "Fine-tuned"],
        ...     title="Validation Accuracy Comparison",
        ...     smooth_window=3,
        ...     save_path="comparison.png"
        ... )
        >>> plt.show()

    Raises:
        ValueError: If metric is not found in any of the metrics.csv files.
    """
    if labels is None:
        labels = [Path(d).name for d in run_dirs]

    if len(labels) != len(run_dirs):
        raise ValueError("labels and run_dirs must have the same length")

    fig, ax = plt.subplots(figsize=figsize)

    metric_names = {
        "train_loss": "Training Loss",
        "val_loss": "Validation Loss",
        "train_acc": "Training Accuracy (%)",
        "val_acc": "Validation Accuracy (%)",
    }

    for run_dir, label in zip(run_dirs, labels):
        df = load_metrics(run_dir)

        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in {run_dir}/metrics.csv")

        x = df["epoch"]
        y = df[metric].copy()

        # Apply smoothing if requested
        if smooth_window is not None and smooth_window > 1:
            y = y.rolling(window=smooth_window, min_periods=1, center=False).mean()

        # Convert accuracy to percentage
        if "acc" in metric:
            y = y * 100

        ax.plot(x, y, label=label, linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(metric_names.get(metric, metric), fontsize=12)

    if title is None:
        title = f"{metric_names.get(metric, metric)} - Comparison"
    ax.set_title(title, fontsize=14, fontweight="bold")

    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


@torch.no_grad()
def get_mean_probs(
    x: torch.Tensor,
    models: list[nn.Module],
    num_classes: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Retourne la moyenne des softmax sur les 7 modèles pour un batch x."""

    probs_sum = torch.zeros(x.shape[0], num_classes, device=device)

    for model in models:
        model = model.to(device=device).eval()

        logits = model(x.to(device=device))
        probs = torch.softmax(logits, dim=1)
        probs_sum += probs

    return probs_sum / len(models)


@torch.no_grad()
def get_mean_probs_fast(
    x: torch.Tensor,
    models: list[nn.Module],
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Version optimisée avec stack des prédictions."""
    x = x.to(device)

    # Collecter toutes les probs en une fois
    all_probs = torch.stack(
        [torch.softmax(model(x), dim=1) for model in models]
    )  # Shape: (num_models, batch_size, num_classes)

    # Moyenne sur la dimension des modèles
    mean_probs = all_probs.mean(dim=0)  # Shape: (batch_size, num_classes)

    return mean_probs


def load_or_train_ensemble(
    num_models: int,
    num_classes: int,
    train_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
    test_loader: DataLoader[Any],
    criterion: Module,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    device: torch.device,
    models_root: str,
    pretrained: bool = False,
    shuffle: bool = False,
    normalization: Literal["MNIST", "ImageNet"] = "MNIST",
    force_retrain: bool = False,
    partial_load: bool = True,
    verbose: bool = True,
) -> tuple[list[Module], list[str]]:
    """Load or train an ensemble of models for uncertainty estimation.

    Automatically checks if models exist on disk. Supports three modes:
        1. All models exist → Load all from disk
        2. No models exist → Train all from scratch
        3. Some models exist → Load existing + train missing (if partial_load=True)

    Args:
        num_models (int): Number of models in the ensemble (e.g., 7).
        num_classes (int): Number of output classes.
        train_loader (DataLoader): Training data loader.
        validation_loader (DataLoader): Validation data loader.
        test_loader (DataLoader): Test data loader.
        criterion (Module): Loss function.
        epochs (int): Number of training epochs per model.
        learning_rate (float): Learning rate for optimizer.
        weight_decay (float): Weight decay for optimizer.
        batch_size (int): Batch size (for logging in config).
        device (torch.device): Device to use for training/loading.
        models_root (str): Root directory for saving/loading models.
        pretrained (bool, optional): Use pretrained weights. Defaults to False.
        shuffle (bool, optional): Shuffle training data. Defaults to False.
        normalization (Literal["MNIST", "ImageNet"], optional): Normalization
            strategy. Defaults to "MNIST".
        force_retrain (bool, optional): Force retraining even if models exist.
            If True, all existing models are deleted and retrained.
            Defaults to False.
        partial_load (bool, optional): Allow loading subset of models. If True
            and only some models exist, loads existing ones and trains missing.
            If False and not all models exist, trains all from scratch.
            Defaults to True.
        verbose (bool, optional): Print progress information. Defaults to True.

    Returns:
        tuple[list[Module], list[str]]: A 2-tuple containing:
            - models (list[Module]): List of trained models on the specified device.
            - model_paths (list[str]): List of paths to saved model files.

    Example:
        >>> # Case 1: Load all existing models
        >>> models, paths = load_or_train_ensemble(
        ...     num_models=7, num_classes=10, train_loader=train_loader,
        ...     validation_loader=val_loader, test_loader=test_loader,
        ...     criterion=nn.CrossEntropyLoss(), epochs=20,
        ...     learning_rate=1e-4, weight_decay=1e-4, batch_size=512,
        ...     device=torch.device("cuda"), models_root="../models"
        ... )
        ✓ Found 7 existing models. Loading from disk...

        >>> # Case 2: Train all from scratch
        >>> models, paths = load_or_train_ensemble(
        ...     num_models=7, force_retrain=True, ...
        ... )
        ⚠ force_retrain=True. Training 7 models from scratch...

        >>> # Case 3: Load 3 existing, train 4 missing
        >>> models, paths = load_or_train_ensemble(
        ...     num_models=7, partial_load=True, ...
        ... )
        ⚠ Found 3/7 existing models. Loading existing and training missing...

    Note:
        - Models are named using get_model_name() for consistency.
        - Each model uses a different random seed (1 to num_models).
        - Test accuracy is computed and logged after training each model.
        - If force_retrain=True, existing model directories are deleted.
    """
    models: list[Module] = []
    model_paths: list[str] = []
    model_names: list[str] = []

    # Generate model names and paths
    for seed in range(1, num_models + 1):
        model_name = get_model_name(
            pretrained=pretrained,
            shuffle=shuffle,
            seed=seed,
            normalization=normalization,
            model_number=seed,
        )
        model_dir = Path(models_root) / model_name
        model_path = model_dir / f"{model_name}.pt"

        model_names.append(model_name)
        model_paths.append(str(model_path))

    # Check which models exist
    existing_mask = [Path(p).exists() for p in model_paths]
    num_existing = sum(existing_mask)
    all_exist = num_existing == num_models
    none_exist = num_existing == 0

    # Handle force_retrain: delete all existing models
    if force_retrain and num_existing > 0:
        if verbose:
            print(f"\n{'=' * 70}")
            print(
                f"⚠️  force_retrain=True. Deleting {num_existing} existing model(s)..."
            )
            print(f"{'=' * 70}\n")

        for model_path in model_paths:
            model_dir = Path(model_path).parent
            if model_dir.exists():
                import shutil

                shutil.rmtree(model_dir)
                if verbose:
                    print(f"  🗑️  Deleted: {model_dir.name}")

        existing_mask = [False] * num_models
        num_existing = 0
        all_exist = False
        none_exist = True

        if verbose:
            print()

    # Determine action based on what exists
    if all_exist and not force_retrain:
        # Case 1: All models exist → Load all
        if verbose:
            print(f"\n{'=' * 70}")
            print(
                f"✓ Found {num_models}/{num_models} existing models. Loading from disk..."
            )
            print(f"{'=' * 70}\n")

        for seed, model_path in enumerate(model_paths, start=1):
            model = make_resnet18(num_classes, pretrained=pretrained)
            model = load_model(model, model_path, device)
            model.eval()
            models.append(model)

            if verbose:
                print(
                    f"  ✓ Loaded model {seed}/{num_models}: {Path(model_path).parent.name}"
                )

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"✓ All {num_models} models loaded successfully!")
            print(f"{'=' * 70}\n")

    elif none_exist or not partial_load:
        # Case 2: No models exist OR partial_load disabled → Train all
        if verbose:
            print(f"\n{'=' * 70}")
            if none_exist:
                print("⚠️  No existing models found.")
            else:
                print(
                    f"⚠️  Found {num_existing}/{num_models} models but partial_load=False."
                )
            print(f"Training all {num_models} models from scratch...")
            print(f"{'=' * 70}\n")

        models, model_paths = _train_all_models(
            num_models=num_models,
            num_classes=num_classes,
            train_loader=train_loader,
            validation_loader=validation_loader,
            test_loader=test_loader,
            criterion=criterion,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            device=device,
            models_root=models_root,
            pretrained=pretrained,
            shuffle=shuffle,
            normalization=normalization,
            verbose=verbose,
        )

    else:
        # Case 3: Some models exist + partial_load=True → Load existing + train missing
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"⚠️  Found {num_existing}/{num_models} existing models.")
            print("Loading existing and training missing models...")
            print(f"{'=' * 70}\n")

        for seed, (exists, model_path) in enumerate(
            zip(existing_mask, model_paths), start=1
        ):
            if exists:
                # Load existing model
                model = make_resnet18(num_classes, pretrained=pretrained)
                model = load_model(model, model_path, device)
                model.eval()
                models.append(model)

                if verbose:
                    print(
                        f"  ✓ Loaded existing model {seed}/{num_models}: {Path(model_path).parent.name}"
                    )

            else:
                # Train missing model
                if verbose:
                    print(f"\n  {'─' * 66}")
                    print(
                        f"  ⚙️  Training missing model {seed}/{num_models} (seed={seed})"
                    )
                    print(f"  {'─' * 66}\n")

                model = _train_single_model(
                    seed=seed,
                    num_classes=num_classes,
                    train_loader=train_loader,
                    validation_loader=validation_loader,
                    test_loader=test_loader,
                    criterion=criterion,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    batch_size=batch_size,
                    device=device,
                    models_root=models_root,
                    pretrained=pretrained,
                    shuffle=shuffle,
                    normalization=normalization,
                    num_models=num_models,
                    verbose=verbose,
                )
                models.append(model)

        if verbose:
            print(f"\n{'=' * 70}")
            print(
                f"✓ Ensemble ready: {num_existing} loaded + {num_models - num_existing} trained!"
            )
            print(f"{'=' * 70}\n")

    return models, model_paths


def _train_single_model(
    seed: int,
    num_classes: int,
    train_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
    test_loader: DataLoader[Any],
    criterion: Module,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    device: torch.device,
    models_root: str,
    pretrained: bool,
    shuffle: bool,
    normalization: Literal["MNIST", "ImageNet"],
    num_models: int,
    verbose: bool,
) -> Module:
    """Train a single model in the ensemble (internal helper function).

    Args:
        seed (int): Random seed for this specific model.
        num_classes (int): Number of output classes.
        train_loader (DataLoader): Training data loader.
        validation_loader (DataLoader): Validation data loader.
        test_loader (DataLoader): Test data loader.
        criterion (Module): Loss function.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate.
        weight_decay (float): Weight decay.
        batch_size (int): Batch size (for config logging).
        device (torch.device): Training device.
        models_root (str): Root directory for models.
        pretrained (bool): Use pretrained weights.
        shuffle (bool): Shuffle training data.
        normalization (Literal["MNIST", "ImageNet"]): Normalization strategy.
        num_models (int): Total number of models in ensemble (for config).
        verbose (bool): Print progress.

    Returns:
        Module: Trained model on the specified device.
    """
    # Set seed for this model
    seed_everything(seed=seed)

    # Create model
    model = make_resnet18(num_classes, pretrained=pretrained)

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Prepare paths
    model_name = get_model_name(
        pretrained=pretrained,
        shuffle=shuffle,
        seed=seed,
        normalization=normalization,
        model_number=seed,
    )
    model_dir = Path(models_root) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_name}.pt"

    # Config for logging
    config = {
        "model": "resnet18",
        "pretrained": pretrained,
        "shuffle": shuffle,
        "seed": seed,
        "normalization": normalization,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "optimizer": "Adam",
        "criterion": "CrossEntropyLoss",
        "ensemble_id": seed,
        "total_ensemble_size": num_models,
    }

    # Train
    model, _, _, _ = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        file_path=str(model_path),
        verbose=verbose,
        save_plots=True,
        config=config,
    )

    # Evaluate on test set
    test_loss, test_accuracy = evaluate(
        model, test_loader, criterion=criterion, device=device
    )

    if verbose:
        print(
            f"\n  ✓ Model {seed}/{num_models} completed | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy * 100:.2f}%\n"
        )

    return model


def _train_all_models(
    num_models: int,
    num_classes: int,
    train_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
    test_loader: DataLoader[Any],
    criterion: Module,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    device: torch.device,
    models_root: str,
    pretrained: bool,
    shuffle: bool,
    normalization: Literal["MNIST", "ImageNet"],
    verbose: bool,
) -> tuple[list[Module], list[str]]:
    """Train all models in the ensemble from scratch (internal helper function).

    Returns:
        tuple[list[Module], list[str]]: Trained models and their paths.
    """
    models: list[Module] = []
    model_paths: list[str] = []

    for seed in range(1, num_models + 1):
        if verbose:
            print(f"\n{'─' * 70}")
            print(f"Training Model {seed}/{num_models} (seed={seed})")
            print(f"{'─' * 70}\n")

        model = _train_single_model(
            seed=seed,
            num_classes=num_classes,
            train_loader=train_loader,
            validation_loader=validation_loader,
            test_loader=test_loader,
            criterion=criterion,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            device=device,
            models_root=models_root,
            pretrained=pretrained,
            shuffle=shuffle,
            normalization=normalization,
            num_models=num_models,
            verbose=verbose,
        )

        model_name = get_model_name(
            pretrained=pretrained,
            shuffle=shuffle,
            seed=seed,
            normalization=normalization,
            model_number=seed,
        )
        model_dir = Path(models_root) / model_name
        model_path = str(model_dir / f"{model_name}.pt")

        models.append(model)
        model_paths.append(model_path)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"✓ All {num_models} models trained successfully!")
        print(f"{'=' * 70}\n")

    return models, model_paths


def get_random_samples(
    dataset: torch.utils.data.Dataset,
    set_size: int,
    seed: int,
    num_samples: int = 20,
) -> tuple[list[tuple[torch.Tensor, int]], list[int]]:
    """Select random samples from a dataset with reproducible seeding.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to sample from.
        set_size (int): Total size of the dataset (len(dataset)).
        seed (int): Random seed for reproducibility.
        num_samples (int, optional): Number of samples to select. Defaults to 20.

    Returns:
        tuple[list[tuple[torch.Tensor, int]], list[int]]: A 2-tuple containing:
            - samples: List of (image, label) tuples.
            - indices: List of selected indices (useful for tracking).

    Example:
        >>> samples, indices = get_random_samples(
        ...     test_data, len(test_data), seed=42, num_samples=20
        ... )
        >>> print(f"Selected indices: {indices}")
        >>> imgs = torch.stack([img for img, _ in samples])
        >>> labels = torch.tensor([y for _, y in samples])

    Note:
        Indices are returned sorted for reproducibility and easier tracking.
    """
    rng = random.Random(seed)
    sel_idx = sorted(rng.sample(range(set_size), num_samples))
    samples = [dataset[idx] for idx in sel_idx]

    return samples, sel_idx
