#!/usr/bin/env python3
"""
Sparse Parity Benchmark: Energy Efficiency Testbed for Learning Algorithms
==========================================================================

This script implements a baseline benchmark for evaluating the "energy cost"
(optimization steps to perfect generalization) of neural network training
algorithms on the (n, k)-sparse parity problem.

NO AUTODIFF. All gradients are computed by hand.
Inspired by Karpathy's microGPT — everything is explicit.

The Sparse Parity Problem:
  Given x ∈ {-1,1}^n, the target is y = ∏_{i∈S} x_i where S is a hidden
  subset of k indices. The network must discover S from data alone.

Architecture:
  x ∈ R^n  →  Linear(W1,b1)  →  ReLU  →  Linear(W2,b2)  →  scalar output f(x)
  Loss: Hinge loss = max(0, 1 - f(x)·y)

Usage:
    python sparse_parity_benchmark.py
"""

import torch
import numpy as np
import time

# =============================================================================
# CONFIGURATION
# =============================================================================

N_BITS = 3            # n: total input bits
K_SPARSE = 3          # k: relevant bits (parity subset size)
N_TRAIN = 20          # training samples
N_TEST = 20           # test samples
SEED = 42

HIDDEN_DIM = 1000     # hidden layer width

LEARNING_RATE = 0.5
WEIGHT_DECAY = 0.01   # L2 regularization — critical for grokking
MAX_EPOCHS = 10       # 10 epochs × 20 samples = 200 optimizer steps
LOG_INTERVAL = 1      # print every epoch
PATIENCE = 10


# =============================================================================
# 1. DATA GENERATION
# =============================================================================

def create_datasets(n=N_BITS, k=K_SPARSE, n_train=N_TRAIN, n_test=N_TEST, seed=SEED):
    """Create (n,k)-sparse parity train/test datasets."""
    rng = np.random.default_rng(seed)

    # Choose the hidden subset S
    secret = np.sort(rng.choice(n, size=k, replace=False))
    print(f"[DATA] Secret parity indices S = {secret.tolist()}")
    print(f"[DATA] Problem: n={n}, k={k}, N_train={n_train}, N_test={n_test}")

    def make_data(num):
        x = torch.from_numpy(rng.choice([-1, 1], size=(num, n)).astype(np.float32))
        y = x[:, secret].prod(dim=1)
        return x, y

    x_train, y_train = make_data(n_train)
    x_test, y_test = make_data(n_test)
    return x_train, y_train, x_test, y_test, secret


# =============================================================================
# 2. MODEL: raw parameter tensors (no nn.Module, no autograd)
# =============================================================================

def init_params(n_in, n_hidden):
    """
    Initialize a 2-layer MLP as raw tensors.
    Kaiming initialization for ReLU layers.
    Returns a dict of {name: tensor}.
    """
    # Layer 1: (n_hidden, n_in)
    W1 = torch.randn(n_hidden, n_in) * (2.0 / n_in) ** 0.5
    b1 = torch.zeros(n_hidden)
    # Layer 2: (1, n_hidden)
    W2 = torch.randn(1, n_hidden) * (2.0 / n_hidden) ** 0.5
    b2 = torch.zeros(1)

    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    total = sum(p.numel() for p in params.values())
    print(f"[MODEL] MLP: {n_in} → {n_hidden} → 1  ({total:,} parameters)")
    return params


# =============================================================================
# 3. FORWARD PASS (manual)
# =============================================================================

def forward(x, params):
    """
    Forward pass through 2-layer ReLU MLP.

    Args:
        x: input tensor, shape (batch, n_in) or (n_in,)
        params: dict with W1, b1, W2, b2

    Returns:
        out: scalar output per sample, shape (batch,)
        cache: intermediates needed for backward
    """
    # Layer 1: linear + ReLU
    h_pre = x @ params['W1'].T + params['b1']     # (batch, hidden)
    h = torch.clamp(h_pre, min=0)                  # ReLU

    # Layer 2: linear
    out = h @ params['W2'].T + params['b2']        # (batch, 1)
    out = out.squeeze(-1)                           # (batch,)

    cache = (x, h_pre, h)
    return out, cache


# =============================================================================
# 4. LOSS (manual)
# =============================================================================

def hinge_loss(out, y):
    """Hinge loss: mean(max(0, 1 - out * y))"""
    return torch.clamp(1.0 - out * y, min=0.0).mean().item()


def compute_accuracy(out, y):
    """Fraction of samples where sign(out) == y."""
    return (out.sign() == y).float().mean().item()


# =============================================================================
# 5. BACKWARD PASS + SGD UPDATE (manual — no .backward(), no autograd)
# =============================================================================
# ┌─────────────────────────────────────────────────────────────────────────┐
# │                    🔧 OPTIMIZER INJECTION POINT 🔧                     │
# │                                                                         │
# │  To swap in an experimental learning algorithm, replace the body of     │
# │  `backward_and_update()`. You have full access to:                      │
# │    - x, y: the single training sample                                   │
# │    - out, cache: forward pass outputs and intermediates                 │
# │    - params: all model weights as raw tensors                           │
# │                                                                         │
# │  The math below is fully explicit — modify any part of the gradient     │
# │  computation or update rule (e.g. gradient normalization, SVD-based     │
# │  updates, evolutionary perturbation, etc.)                              │
# └─────────────────────────────────────────────────────────────────────────┘

def backward_and_update(x, y, out, cache, params, lr=LEARNING_RATE, wd=WEIGHT_DECAY):
    """
    Manual backward pass + SGD update for a single sample.

    Architecture:  x → [W1, b1] → ReLU → [W2, b2] → out
    Loss:          L = max(0, 1 - out * y)

    Gradients are derived by hand and applied inline.
    Weight decay: param -= lr * (grad + wd * param)
    """
    x_i, h_pre, h = cache  # x_i: (1, n), h_pre: (1, hidden), h: (1, hidden)

    margin = (out * y).item()
    if margin >= 1.0:
        return  # hinge loss is 0, no gradient

    # ── dL/dout ──────────────────────────────────────────────────────────
    # L = max(0, 1 - out·y),  dL/dout = -y  (when margin < 1)
    dout = -y                                      # (1,)

    # ── Layer 2 backward: out = W2 @ h + b2 ─────────────────────────────
    dW2 = dout.unsqueeze(1) * h                    # (1, hidden)
    db2 = dout                                     # (1,)
    dh  = params['W2'].T * dout                    # (hidden, 1) → broadcast

    # ── ReLU backward: h = relu(h_pre) ──────────────────────────────────
    dh_pre = dh.T * (h_pre > 0).float()            # (1, hidden)

    # ── Layer 1 backward: h_pre = W1 @ x + b1 ──────────────────────────
    dW1 = dh_pre.T @ x_i                           # (hidden, n)
    db1 = dh_pre.squeeze(0)                        # (hidden,)

    # ── SGD update with weight decay ────────────────────────────────────
    grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
    for name in params:
        params[name] -= lr * (grads[name] + wd * params[name])


# =============================================================================
# 6. ENERGY TRACKER
# =============================================================================

class EnergyTracker:
    """Tracks steps to perfect generalization + per-step losses for plotting."""

    def __init__(self):
        self.train_accs = []
        self.test_accs = []
        self.train_losses = []
        self.test_losses = []
        self.generalization_epoch = None
        self.step_count = 0

    def record_step(self):
        self.step_count += 1

    def update(self, epoch, train_acc, test_acc, train_loss, test_loss):
        self.train_accs.append(train_acc)
        self.test_accs.append(test_acc)
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        if test_acc >= 1.0 and self.generalization_epoch is None:
            self.generalization_epoch = epoch

    @property
    def has_generalized(self):
        return self.generalization_epoch is not None

    def report(self):
        lines = [
            "",
            "=" * 65,
            "  ENERGY COST REPORT",
            "=" * 65,
            f"  Total optimizer steps: {self.step_count}",
        ]
        if self.has_generalized:
            lines.append(f"  ✅ Perfect generalization at epoch {self.generalization_epoch}")
            lines.append(f"  ⚡ ENERGY COST = {self.generalization_epoch} epochs "
                         f"({self.step_count} optimizer steps)")
        else:
            lines.append(f"  ❌ Did NOT reach 100% test accuracy within "
                         f"{len(self.train_accs)} steps")
            best_test = max(self.test_accs)
            best_step = self.test_accs.index(best_test)
            lines.append(f"  Best test accuracy: {best_test:.2%} at step {best_step}")
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

def train(params, x_train, y_train, x_test, y_test,
          max_epochs=MAX_EPOCHS, log_interval=LOG_INTERVAL, patience=PATIENCE):
    """
    Single-sample cyclic training loop with manual backprop.
    Evaluates full-batch train/test loss after every optimizer step.
    """
    tracker = EnergyTracker()
    best_test_acc = 0.0
    epochs_without_improvement = 0
    n_train = len(x_train)
    x_all = torch.cat([x_train, x_test], dim=0)
    y_all = torch.cat([y_train, y_test], dim=0)

    print(f"\n[TRAIN] Starting training for up to {max_epochs} epochs...")
    print(f"[TRAIN] Mode: single-sample cyclic (batch_size=1, fixed order)")
    print(f"[TRAIN] Steps per epoch: {n_train}")
    print(f"[TRAIN] Loss: Hinge Loss")
    print(f"[TRAIN] Log interval: every {log_interval} epochs\n")

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        for i in range(n_train):
            # Single-sample forward
            x_i = x_train[i:i+1]
            y_i = y_train[i:i+1]
            out_i, cache = forward(x_i, params)

            # Backward + update (manual — no autograd)
            backward_and_update(x_i, y_i, out_i, cache, params)
            tracker.record_step()

            # Evaluate full-batch train & test after every step
            with torch.no_grad():
                all_out, _ = forward(x_all, params)
                train_out = all_out[:n_train]
                test_out  = all_out[n_train:]
                train_loss = hinge_loss(train_out, y_train)
                test_loss  = hinge_loss(test_out,  y_test)
                train_acc  = compute_accuracy(train_out, y_train)
                test_acc   = compute_accuracy(test_out,  y_test)

            tracker.update(epoch, train_acc, test_acc, train_loss, test_loss)

            # Logging
            if tracker.step_count % (log_interval * n_train) == 0 or tracker.step_count == 1:
                print(f"\n{'─' * 65}")
                print(f"[STEP {tracker.step_count}] epoch={epoch}, sample={i}  "
                      f"train_loss={train_loss:.6f}  test_loss={test_loss:.6f}  "
                      f"train_acc={train_acc:.2%}  test_acc={test_acc:.2%}")

                print(f"  Train predictions (raw → sign vs target):")
                for j in range(n_train):
                    raw = train_out[j].item()
                    pred = 1 if raw >= 0 else -1
                    tgt = int(y_train[j].item())
                    print(f"    sample {j:>2}: f(x)={raw:+.4f}  "
                          f"pred={pred:+d}  target={tgt:+d}  "
                          f"{'✓' if pred == tgt else '✗'}")

                print(f"  Test predictions (raw → sign vs target):")
                for j in range(len(x_test)):
                    raw = test_out[j].item()
                    pred = 1 if raw >= 0 else -1
                    tgt = int(y_test[j].item())
                    print(f"    sample {j:>2}: f(x)={raw:+.4f}  "
                          f"pred={pred:+d}  target={tgt:+d}  "
                          f"{'✓' if pred == tgt else '✗'}")

                print(f"  Weight norms:")
                for name, p in params.items():
                    print(f"    {name:>4s}: |W|={p.norm().item():.4f}")

        # Early stopping on perfect generalization
        if tracker.has_generalized:
            elapsed = time.time() - start_time
            print(f"\n  🎉 GENERALIZED at epoch {epoch}! "
                  f"({tracker.step_count} steps, {elapsed:.1f}s)")
            break

        # Patience
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

def plot_losses(tracker, save_path="loss_plot.png"):
    """Plot training and test loss/accuracy curves."""
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(steps, tracker.train_losses, label='Train Loss', lw=1.2, color='#2196F3')
    ax1.plot(steps, tracker.test_losses,  label='Test Loss',  lw=1.2, color='#F44336')
    ax1.set(xlabel='Optimizer Step', ylabel='Hinge Loss', title='Training & Test Loss (per step)')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(steps, tracker.train_accs, label='Train Accuracy', lw=1.2, color='#2196F3')
    ax2.plot(steps, tracker.test_accs,  label='Test Accuracy',  lw=1.2, color='#F44336')
    ax2.set(xlabel='Optimizer Step', ylabel='Accuracy', title='Training & Test Accuracy (per step)')
    ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_ylim(-0.05, 1.05)

    if tracker.generalization_epoch is not None:
        for ax in (ax1, ax2):
            ax.axvline(x=tracker.generalization_epoch, color='green',
                       linestyle='--', alpha=0.7, label='Generalized')

    fig.suptitle('Sparse Parity Benchmark — Loss & Accuracy Curves',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[PLOT] Saved loss/accuracy plot to: {save_path}")
    plt.close(fig)


# =============================================================================
# 9. MAIN
# =============================================================================

def main():
    total_start = time.time()

    print("=" * 65)
    print("  SPARSE PARITY BENCHMARK — Energy Efficiency Testbed")
    print("  (No autodiff — all gradients computed by hand)")
    print("=" * 65)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # --- Initialization ---
    init_start = time.time()
    x_train, y_train, x_test, y_test, secret = create_datasets()
    print(f"[DATA] Train labels: {y_train.tolist()}")
    print(f"[DATA] Test labels:  {y_test.tolist()}")
    params = init_params(N_BITS, HIDDEN_DIM)
    init_elapsed = time.time() - init_start

    # --- Simulation ---
    sim_start = time.time()
    tracker = train(params, x_train, y_train, x_test, y_test)
    sim_elapsed = time.time() - sim_start

    # --- Report ---
    print(tracker.report())

    # --- Plot ---
    plot_start = time.time()
    plot_losses(tracker)
    plot_elapsed = time.time() - plot_start

    total_elapsed = time.time() - total_start
    print(f"\n[TIMING] Initialization:  {init_elapsed:.3f}s  (data + model)")
    print(f"[TIMING] Simulation:      {sim_elapsed:.3f}s  (training loop)")
    print(f"[TIMING] Plotting:        {plot_elapsed:.3f}s  (matplotlib)")
    print(f"[TIMING] Total wall time: {total_elapsed:.3f}s")

    return tracker


if __name__ == "__main__":
    main()
