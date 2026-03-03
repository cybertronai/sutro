#!/usr/bin/env python3
# --- Colab/pip compatibility: install PyTorch if not already available ---
try:
    import torch
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "--quiet"])
    import torch

"""
Sparse Parity Benchmark: Energy Efficiency Testbed for Learning Algorithms
==========================================================================

This script implements a baseline benchmark for evaluating the "energy cost"
(optimization steps to perfect generalization) of neural network training
algorithms on the (n, k)-sparse parity problem.

**The Sparse Parity Problem:**
Given an input x ∈ {-1, 1}^n, the target is y = ∏_{i ∈ S} x_i, where S is a
hidden subset of k indices. A network must discover S from data alone.

**Why this is a good benchmark:**
Standard SGD exhibits "grokking" on this task—it memorizes training data quickly
(~100% train accuracy) but sits at ~50% test accuracy for thousands of epochs
before suddenly generalizing. The "energy cost" of this plateau is the metric
to beat with improved algorithms.

**Modularity:**
The optimizer/update rule is isolated in a clearly marked section so that
experimental AI-generated learning algorithms can be swapped in easily.

Literature hyperparameters for reliable grokking dynamics (can scale up from
the toy defaults below):
  n=40, k=3, N=1000, hidden=1000, lr=0.1, weight_decay=0.01

Usage:
    python sparse_parity_benchmark.py
"""

import torch.nn as nn
import numpy as np
import time
from typing import Tuple, Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Problem Parameters ---
N_BITS = 3            # n: total number of input bits
K_SPARSE = 3          # k: number of relevant bits (parity subset size)
N_TRAIN = 20          # number of training samples (doubled; steps/epoch also doubles)
N_TEST = 20           # number of test samples
SEED = 42             # random seed for reproducibility

# --- Model Parameters ---
HIDDEN_DIM = 1000     # width of the hidden layer

# --- Training Parameters ---
LEARNING_RATE = 0.1
WEIGHT_DECAY = 0.01   # L2 regularization — critical for grokking transition
MAX_EPOCHS = 10       # 10 epochs × 20 samples = 200 optimizer steps (same as before)
LOG_INTERVAL = 1      # print metrics every epoch

# --- Energy Tracking ---
PATIENCE = 10         # epochs of no improvement before declaring failure


# =============================================================================
# 1. DATA GENERATION
# =============================================================================

def generate_sparse_parity_data(
    n: int,
    k: int,
    num_samples: int,
    secret_indices: torch.Tensor,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate data for the (n, k)-sparse parity task.

    Args:
        n: Total number of input bits.
        k: Number of relevant (parity) bits.
        num_samples: How many samples to generate.
        secret_indices: 1-D tensor of k indices defining the hidden subset S.
        rng: NumPy random generator for reproducibility.

    Returns:
        x: Tensor of shape (num_samples, n) with entries in {-1, +1}.
        y: Tensor of shape (num_samples,) with entries in {-1, +1}.
           y_i = product of x_i at the secret indices.
    """
    # Sample uniformly from {-1, +1}^n
    x_np = rng.choice([-1, 1], size=(num_samples, n)).astype(np.float32)
    x = torch.from_numpy(x_np)

    # Compute parity: y = ∏_{i ∈ S} x_i
    x_relevant = x[:, secret_indices]          # (num_samples, k)
    y = x_relevant.prod(dim=1)                 # (num_samples,)

    return x, y


def create_datasets(
    n: int = N_BITS,
    k: int = K_SPARSE,
    n_train: int = N_TRAIN,
    n_test: int = N_TEST,
    seed: int = SEED,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create train and test datasets with a randomly chosen secret subset S.

    Returns:
        x_train, y_train, x_test, y_test, secret_indices
    """
    rng = np.random.default_rng(seed)

    # Choose the hidden subset S of k indices from {0, ..., n-1}
    secret_indices = torch.from_numpy(
        rng.choice(n, size=k, replace=False).astype(np.int64)
    )
    secret_indices, _ = secret_indices.sort()

    print(f"[DATA] Secret parity indices S = {secret_indices.tolist()}")
    print(f"[DATA] Problem: n={n}, k={k}, N_train={n_train}, N_test={n_test}")

    x_train, y_train = generate_sparse_parity_data(n, k, n_train, secret_indices, rng)
    x_test, y_test = generate_sparse_parity_data(n, k, n_test, secret_indices, rng)

    return x_train, y_train, x_test, y_test, secret_indices


# =============================================================================
# 2. MODEL DEFINITION
# =============================================================================

class ParityMLP(nn.Module):
    """
    2-layer MLP for the sparse parity task.

    Architecture:
        Input (n) → Linear (hidden_dim) → ReLU → Linear (1) → scalar output

    The output is a raw score f(x); the sign of f(x) is the predicted label.
    """

    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns shape (batch,) — the raw score for each input."""
        return self.net(x).squeeze(-1)


# =============================================================================
# 3. LOSS FUNCTION
# =============================================================================

def hinge_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Hinge loss: L = mean(max(0, 1 - f(x) · y))

    Args:
        predictions: Raw model outputs, shape (batch,).
        targets: True labels in {-1, +1}, shape (batch,).

    Returns:
        Scalar loss.
    """
    return torch.clamp(1.0 - predictions * targets, min=0.0).mean()


# =============================================================================
# 4. METRICS
# =============================================================================

def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Classification accuracy: fraction of samples where sign(f(x)) == y.
    """
    predicted_labels = predictions.sign()
    correct = (predicted_labels == targets).float().mean()
    return correct.item()


# =============================================================================
# 5. OPTIMIZER FACTORY
# =============================================================================
# ┌─────────────────────────────────────────────────────────────────────────┐
# │                    🔧 OPTIMIZER INJECTION POINT 🔧                     │
# │                                                                         │
# │  To swap in an experimental learning algorithm:                         │
# │                                                                         │
# │  Option A: Replace the body of `create_optimizer()` to return your      │
# │            custom torch.optim.Optimizer subclass.                       │
# │                                                                         │
# │  Option B: Replace the `optimizer_step()` function below to implement   │
# │            a completely custom update rule (e.g., evolutionary,         │
# │            gradient normalization, SVD-based updates, etc.).            │
# │            In that case, set the optimizer to None.                     │
# │                                                                         │
# │  The training loop calls `optimizer_step()` once per sample per epoch.  │
# └─────────────────────────────────────────────────────────────────────────┘

def create_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """
    Create the optimizer for training.

    **DEFAULT**: Standard SGD with weight decay.
    **TO CUSTOMIZE**: Return your own Optimizer or None (if using a fully
    custom update rule in `optimizer_step`).
    """
    return torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


def optimizer_step(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    loss: torch.Tensor,
) -> None:
    """
    Perform one optimization step.

    **DEFAULT**: Standard backprop + optimizer.step().
    **TO CUSTOMIZE**: Replace this function body with your custom update rule.
    You have full access to the model parameters, gradients, and the loss.

    Example custom update (gradient normalization):
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= lr * p.grad / (p.grad.norm() + 1e-8)

    Args:
        model: The neural network.
        optimizer: The optimizer (may be None for fully custom rules).
        loss: The scalar loss tensor (not yet backpropagated).
    """
    # --- DEFAULT: standard gradient descent ---
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# =============================================================================
# 6. ENERGY TRACKER
# =============================================================================

class EnergyTracker:
    """
    Tracks the "energy cost" of training: the total number of optimization
    steps (epochs) required for the test accuracy to reach 100%.

    Also records per-step train and test losses for plotting.
    """

    def __init__(self):
        self.train_accs: list[float] = []
        self.test_accs: list[float] = []
        self.losses: list[float] = []
        self.train_losses: list[float] = []   # full-batch train loss, recorded per step
        self.test_losses: list[float] = []    # full-batch test loss, recorded per step
        self.generalization_epoch: Optional[int] = None  # epoch of 100% test acc
        self.step_count: int = 0              # total optimizer steps taken

    def update(self, step: int, epoch: int, train_acc: float, test_acc: float,
               train_loss: float, test_loss: float):
        self.train_accs.append(train_acc)
        self.test_accs.append(test_acc)
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)

        if test_acc >= 1.0 and self.generalization_epoch is None:
            self.generalization_epoch = epoch

    def record_step(self):
        """Increment the global optimizer step counter."""
        self.step_count += 1

    @property
    def has_generalized(self) -> bool:
        return self.generalization_epoch is not None

    def report(self) -> str:
        """Generate a final energy cost report."""
        lines = [
            "",
            "=" * 65,
            "  ENERGY COST REPORT",
            "=" * 65,
            f"  Total optimizer steps: {self.step_count}",
        ]
        if self.has_generalized:
            lines.append(
                f"  ✅ Perfect generalization reached at epoch {self.generalization_epoch}"
            )
            lines.append(
                f"  ⚡ ENERGY COST = {self.generalization_epoch} epochs "
                f"({self.step_count} optimizer steps)"
            )
        else:
            lines.append(
                f"  ❌ Did NOT reach 100% test accuracy within {len(self.train_accs)} epochs"
            )
            best_test = max(self.test_accs)
            best_epoch = self.test_accs.index(best_test)
            lines.append(f"  Best test accuracy: {best_test:.2%} at epoch {best_epoch}")
            lines.append(f"  ⚡ ENERGY COST = ∞ (failed to generalize)")

        lines.append(f"  Final train loss:     {self.train_losses[-1]:.6f}")
        lines.append(f"  Final test loss:      {self.test_losses[-1]:.6f}")
        lines.append(f"  Final train accuracy: {self.train_accs[-1]:.2%}")
        lines.append(f"  Final test accuracy:  {self.test_accs[-1]:.2%}")
        lines.append("=" * 65)
        return "\n".join(lines)


# =============================================================================
# 7. TRAINING LOOP
# =============================================================================

def train(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    max_epochs: int = MAX_EPOCHS,
    log_interval: int = LOG_INTERVAL,
    patience: int = PATIENCE,
) -> EnergyTracker:
    """
    Single-sample cyclic training loop with energy tracking.

    Each epoch iterates through ALL training samples one at a time in fixed
    order (index 0, 1, 2, ..., N-1), performing one optimizer step per sample.
    After each epoch, full-batch train/test losses and accuracies are evaluated.

    Args:
        model: The neural network to train.
        optimizer: The optimizer (or None for custom rules).
        x_train, y_train: Training data.
        x_test, y_test: Test data.
        max_epochs: Maximum number of training epochs.
        log_interval: Print metrics every this many epochs.
        patience: Stop if test accuracy hasn't improved in this many epochs.

    Returns:
        EnergyTracker with the full training history.
    """
    tracker = EnergyTracker()
    best_test_acc = 0.0
    epochs_without_improvement = 0
    n_train = len(x_train)
    x_all = torch.cat([x_train, x_test], dim=0)  # pre-concat for batched eval
    print(f"\n[TRAIN] Starting training for up to {max_epochs} epochs...")
    print(f"[TRAIN] Mode: single-sample cyclic (batch_size=1, fixed order)")
    print(f"[TRAIN] Steps per epoch: {n_train}")
    print(f"[TRAIN] Optimizer: {optimizer.__class__.__name__ if optimizer else 'Custom'}")
    print(f"[TRAIN] Loss: Hinge Loss")
    print(f"[TRAIN] Log interval: every {log_interval} epochs\n")

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        # --- Cycle through each training sample in fixed order ---
        for i in range(n_train):
            model.train()
            # Single-sample forward pass
            x_i = x_train[i:i+1]   # shape (1, n)
            y_i = y_train[i:i+1]   # shape (1,)
            pred_i = model(x_i)
            loss_i = hinge_loss(pred_i, y_i)

            # --- Optimization step (THE PART TO CUSTOMIZE) ---
            # Skip if loss is already 0: avoids weight-decay updates that would
            # spuriously change test loss after train loss has fully converged.
            if loss_i.item() > 0:
                optimizer_step(model, optimizer, loss_i)
            else:
                # Still need to clear any stale gradients
                if optimizer is not None:
                    optimizer.zero_grad()
            tracker.record_step()

            # --- Evaluate full-batch train & test after EVERY step ---
            # Combine into one forward pass for efficiency
            model.eval()
            with torch.no_grad():
                all_preds  = model(x_all)
                train_preds = all_preds[:n_train]
                test_preds  = all_preds[n_train:]
                train_loss  = hinge_loss(train_preds, y_train).item()
                test_loss   = hinge_loss(test_preds,  y_test).item()
                train_acc   = compute_accuracy(train_preds, y_train)
                test_acc    = compute_accuracy(test_preds,  y_test)

            tracker.update(tracker.step_count, epoch,
                           train_acc, test_acc, train_loss, test_loss)

            # --- Logging (once per sample when log_interval==1) ---
            if tracker.step_count % (log_interval * n_train) == 0 or tracker.step_count == 1:
                print(f"\n{'─' * 65}")
                print(f"[STEP {tracker.step_count}] epoch={epoch}, sample={i}  "
                      f"train_loss={train_loss:.6f}  test_loss={test_loss:.6f}  "
                      f"train_acc={train_acc:.2%}  test_acc={test_acc:.2%}")

                # Per-sample predictions
                print(f"  Train predictions (raw → sign vs target):")
                for j in range(n_train):
                    raw = train_preds[j].item()
                    pred_label = 1 if raw >= 0 else -1
                    target = int(y_train[j].item())
                    match = "✓" if pred_label == target else "✗"
                    print(f"    sample {j:>2}: f(x)={raw:+.4f}  "
                          f"pred={pred_label:+d}  target={target:+d}  {match}")

                print(f"  Test predictions (raw → sign vs target):")
                for j in range(len(x_test)):
                    raw = test_preds[j].item()
                    pred_label = 1 if raw >= 0 else -1
                    target = int(y_test[j].item())
                    match = "✓" if pred_label == target else "✗"
                    print(f"    sample {j:>2}: f(x)={raw:+.4f}  "
                          f"pred={pred_label:+d}  target={target:+d}  {match}")

                # Weight norms and gradient norms per layer
                print(f"  Weight/gradient norms:")
                for name, param in model.named_parameters():
                    w_norm = param.data.norm().item()
                    g_norm = param.grad.norm().item() if param.grad is not None else 0.0
                    print(f"    {name:>20s}: |W|={w_norm:.4f}  |∇W|={g_norm:.6f}")

        # --- Early stopping on perfect generalization (checked after every epoch) ---
        if tracker.has_generalized:
            elapsed = time.time() - start_time
            print(f"\n  🎉 GENERALIZED at epoch {epoch}! "
                  f"({tracker.step_count} steps, {elapsed:.1f}s)")
            break

        # --- Patience-based stopping ---
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\n[TRAIN] Stopping: no test improvement in {patience} epochs.")
            break

    elapsed = time.time() - start_time
    print(f"[TRAIN] Finished in {elapsed:.1f}s "
          f"({epoch} epochs, {tracker.step_count} total steps)")

    return tracker


# =============================================================================
# 8. PLOTTING
# =============================================================================

def plot_losses(tracker: EnergyTracker, save_path: str = "loss_plot.png"):
    """
    Plot training and test loss curves and save to a file.

    Args:
        tracker: EnergyTracker with recorded train_losses and test_losses.
        save_path: File path to save the plot.
    """
    # Lazy import — avoids ~0.7s startup cost when matplotlib isn't needed yet
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "--quiet"])
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

    steps = list(range(1, len(tracker.train_losses) + 1))
    n_steps = len(steps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss plot ---
    ax1.plot(steps, tracker.train_losses, label='Train Loss', linewidth=1.2, color='#2196F3')
    ax1.plot(steps, tracker.test_losses,  label='Test Loss',  linewidth=1.2, color='#F44336')
    ax1.set_xlabel('Optimizer Step')
    ax1.set_ylabel('Hinge Loss')
    ax1.set_title('Training & Test Loss (per step)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Accuracy plot ---
    ax2.plot(steps, tracker.train_accs, label='Train Accuracy', linewidth=1.2, color='#2196F3')
    ax2.plot(steps, tracker.test_accs,  label='Test Accuracy',  linewidth=1.2, color='#F44336')
    ax2.set_xlabel('Optimizer Step')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training & Test Accuracy (per step)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)

    # Mark generalization step if it happened
    if tracker.generalization_epoch is not None:
        # convert epoch → approximate step number
        gen_step = tracker.generalization_epoch  # stored as step count now
        for ax in (ax1, ax2):
            ax.axvline(x=gen_step, color='green',
                       linestyle='--', alpha=0.7, label='Generalized')

    fig.suptitle('Sparse Parity Benchmark — Loss & Accuracy Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[PLOT] Saved loss/accuracy plot to: {save_path}")
    plt.close(fig)


# =============================================================================
# 9. MAIN
# =============================================================================

def main():
    """
    Run the sparse parity benchmark with the default (SGD) optimizer.
    """
    total_start = time.time()

    print("=" * 65)
    print("  SPARSE PARITY BENCHMARK — Energy Efficiency Testbed")
    print("=" * 65)

    # Reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # --- Initialization (data + model + optimizer) ---
    init_start = time.time()

    x_train, y_train, x_test, y_test, secret = create_datasets()
    print(f"[DATA] Train labels: {y_train.tolist()}")
    print(f"[DATA] Test labels:  {y_test.tolist()}")

    model = ParityMLP(input_dim=N_BITS, hidden_dim=HIDDEN_DIM)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] ParityMLP: {N_BITS} → {HIDDEN_DIM} → 1  ({num_params:,} parameters)")

    optimizer = create_optimizer(model)

    init_elapsed = time.time() - init_start

    # --- Simulation (training loop only) ---
    sim_start = time.time()
    tracker = train(model, optimizer, x_train, y_train, x_test, y_test)
    sim_elapsed = time.time() - sim_start

    # --- Report ---
    print(tracker.report())

    # --- Plot losses ---
    plot_start = time.time()
    plot_losses(tracker)
    plot_elapsed = time.time() - plot_start

    total_elapsed = time.time() - total_start
    print(f"\n[TIMING] Initialization:  {init_elapsed:.3f}s  (data + model + optimizer)")
    print(f"[TIMING] Simulation:      {sim_elapsed:.3f}s  (training loop)")
    print(f"[TIMING] Plotting:        {plot_elapsed:.3f}s  (matplotlib)")
    print(f"[TIMING] Total wall time: {total_elapsed:.3f}s")

    return tracker


if __name__ == "__main__":
    main()
