"""
utils.py - Reusable training and evaluation utilities for neural network exploration.

This module provides clean abstractions for training loops, evaluation,
and visualization so the main notebook can focus on experiments and analysis.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from tqdm import tqdm
import copy
import time


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch. Returns average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

    return running_loss / total, 100.0 * correct / total


def evaluate(model, test_loader, criterion, device):
    """Evaluate model. Returns average loss, accuracy, all predictions, and all labels."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)

            running_loss += loss.item() * X.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return running_loss / total, 100.0 * correct / total, np.array(all_preds), np.array(all_labels)


def train_model(model, train_loader, test_loader, optimizer, criterion, device,
                epochs=30, scheduler=None, verbose=True):
    """
    Full training loop with history tracking.
    
    Returns:
        dict with keys: train_loss, train_acc, val_loss, val_acc, best_model, best_epoch, elapsed_time
    """
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.1f}% | "
                  f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.1f}%")

    elapsed = time.time() - start_time
    history['best_model'] = best_model_state
    history['best_epoch'] = best_epoch
    history['best_val_acc'] = best_val_acc
    history['elapsed_time'] = elapsed

    if verbose:
        print(f"  ✓ Best val acc: {best_val_acc:.2f}% at epoch {best_epoch} ({elapsed:.1f}s)")

    return history


def plot_training_curves(history, title="Training Curves"):
    """Plot loss and accuracy curves side-by-side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], label='Train', linewidth=2)
    ax1.plot(epochs, history['val_loss'], label='Validation', linewidth=2)
    ax1.axvline(history['best_epoch'], color='red', linestyle='--', alpha=0.5, label=f"Best (epoch {history['best_epoch']})")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} — Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_acc'], label='Train', linewidth=2)
    ax2.plot(epochs, history['val_acc'], label='Validation', linewidth=2)
    ax2.axvline(history['best_epoch'], color='red', linestyle='--', alpha=0.5, label=f"Best (epoch {history['best_epoch']})")
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{title} — Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, test_loader, device, class_names, title="Confusion Matrix"):
    """Generate and plot a normalized confusion matrix."""
    _, _, preds, labels = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    cm = confusion_matrix(labels, preds, normalize='true')

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

    return cm


def extract_features(model, loader, device, layer_hook=None):
    """
    Extract intermediate representations from a model.
    By default, extracts the output of the second-to-last layer.
    
    Returns: features (np.array), labels (np.array)
    """
    model.eval()
    features = []
    labels = []
    hook_output = []

    def hook_fn(module, input, output):
        hook_output.append(output.detach().cpu())

    # Register hook on second-to-last layer if no specific layer given
    if layer_hook is None:
        # Find the last linear layer and hook its input
        modules = list(model.modules())
        linear_layers = [m for m in modules if isinstance(m, nn.Linear)]
        if len(linear_layers) >= 2:
            handle = linear_layers[-2].register_forward_hook(hook_fn)
        else:
            handle = linear_layers[-1].register_forward_hook(
                lambda m, i, o: hook_output.append(i[0].detach().cpu())
            )
    else:
        handle = layer_hook.register_forward_hook(hook_fn)

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            hook_output.clear()
            _ = model(X)
            if hook_output:
                features.append(hook_output[0])
            labels.extend(y.numpy())

    handle.remove()
    features = torch.cat(features, dim=0).numpy()
    labels = np.array(labels)
    return features, labels


def plot_tsne(features, labels, class_names, title="t-SNE of Learned Representations",
              n_samples=2000, perplexity=30):
    """Visualize learned representations using t-SNE."""
    # Subsample for speed
    if len(features) > n_samples:
        idx = np.random.choice(len(features), n_samples, replace=False)
        features = features[idx]
        labels = labels[idx]

    print(f"  Running t-SNE on {len(features)} samples...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    embedded = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(embedded[:, 0], embedded[:, 1],
                         c=labels, cmap='tab10', s=8, alpha=0.6)

    # Add legend
    handles = []
    for i, name in enumerate(class_names):
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=plt.cm.tab10(i / 10),
                                  markersize=8, label=name))
    ax.legend(handles=handles, loc='best', fontsize=8)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def compare_experiments(results_dict, metric='val_acc'):
    """Bar chart comparing final val accuracy across experiments."""
    names = list(results_dict.keys())
    values = [results_dict[n]['best_val_acc'] for n in names]
    times = [results_dict[n]['elapsed_time'] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = sns.color_palette("viridis", len(names))
    bars = ax1.barh(names, values, color=colors)
    ax1.set_xlabel('Best Validation Accuracy (%)')
    ax1.set_title('Accuracy Comparison')
    for bar, val in zip(bars, values):
        ax1.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                 f'{val:.1f}%', va='center', fontsize=9)
    ax1.set_xlim(0, max(values) * 1.1)

    bars2 = ax2.barh(names, times, color=colors)
    ax2.set_xlabel('Training Time (seconds)')
    ax2.set_title('Training Time Comparison')
    for bar, val in zip(bars2, times):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 f'{val:.1f}s', va='center', fontsize=9)

    plt.tight_layout()
    plt.show()


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
