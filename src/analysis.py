from pathlib import Path

import matplotlib.pyplot as plt

import tools

# Root directory
MODELS_ROOT = Path("../models")

# Load all runs
run1 = MODELS_ROOT / "resnet18_normMNIST_no-shuffle_seed0"
run2 = MODELS_ROOT / "resnet18_pre-trained_normMNIST_no-shuffle_seed0"
run3 = MODELS_ROOT / "resnet18_pre-trained_normImageNet_shuffle_seed42"

# Compare validation accuracy
fig = tools.plot_metric_comparison(
    run_dirs=[run1, run2, run3],
    metric="val_acc",
    labels=["From Scratch", "Pretrained (MNIST norm)", "Pretrained (ImageNet norm)"],
    title="MNIST Classification - Validation Accuracy Comparison",
    smooth_window=3,
    save_path="../figures/val_acc_comparison.png",
)
plt.show()

# Compare training loss
fig = tools.plot_metric_comparison(
    run_dirs=[run1, run2, run3],
    metric="train_loss",
    labels=["From Scratch", "Pretrained (MNIST norm)", "Pretrained (ImageNet norm)"],
    title="Training Loss Comparison",
    save_path="../figures/train_loss_comparison.png",
)
plt.show()

# Analyze a specific run
df = tools.load_metrics(run2)
print(f"Best validation accuracy: {df['val_acc'].max() * 100:.2f}%")
print(f"Final training loss: {df['train_loss'].iloc[-1]:.4f}")
print(f"Total training time: {df['time_seconds'].sum() / 60:.1f} minutes")

# Load configuration
config = tools.load_config(run2)
print(f"Learning rate used: {config['learning_rate']}")
print(f"Batch size: {config['batch_size']}")
