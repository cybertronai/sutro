#!/usr/bin/env python3
"""
Sparse Parity Benchmark — Pure Python, No Dependencies
=======================================================

The most atomic way to train a neural network on sparse parity.
This file is the complete algorithm. Everything else is just efficiency.

No torch. No numpy. No autograd. Just Python, math, and random.
Inspired by Karpathy's microGPT.

Architecture: x ∈ {-1,1}^n → Linear(W1,b1) → ReLU → Linear(W2,b2) → scalar
Loss:         Hinge loss = max(0, 1 - f(x)·y)
Optimizer:    SGD with weight decay
"""

import math
import random
import time

random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================

N_BITS    = 3       # total input bits
K_SPARSE  = 3       # parity subset size
N_TRAIN   = 20      # training samples
N_TEST    = 20      # test samples
HIDDEN    = 1000    # hidden layer width
LR        = 0.5     # learning rate
WD        = 0.01    # weight decay
MAX_EPOCHS = 10     # 10 epochs × 20 samples = 200 steps
LOG_EVERY  = 1      # log every N epochs
PATIENCE   = 10

# =============================================================================
# MATRIX HELPERS — the "linear algebra library" in pure Python
# =============================================================================

def zeros(rows, cols=None):
    """Create a zero vector (if cols=None) or matrix."""
    if cols is None:
        return [0.0] * rows
    return [[0.0] * cols for _ in range(rows)]

def randn(rows, cols=None, std=1.0):
    """Sample from N(0, std^2). Returns vector or matrix."""
    if cols is None:
        return [random.gauss(0, std) for _ in range(rows)]
    return [[random.gauss(0, std) for _ in range(cols)] for _ in range(rows)]

def matvec(M, v):
    """Matrix-vector product: M @ v. M is (m, n), v is (n,) → (m,)."""
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]

def vecmat(v, M):
    """Vector-matrix product: v @ M. v is (m,), M is (m, n) → (n,)."""
    n = len(M[0])
    return [sum(v[i] * M[i][j] for i in range(len(v))) for j in range(n)]

def outer(u, v):
    """Outer product: u ⊗ v. u is (m,), v is (n,) → (m, n)."""
    return [[u[i] * v[j] for j in range(len(v))] for i in range(len(u))]

def dot(u, v):
    """Dot product."""
    return sum(a * b for a, b in zip(u, v))

def norm(v):
    """L2 norm of a vector."""
    return math.sqrt(sum(x * x for x in v))

def mat_norm(M):
    """Frobenius norm of a matrix."""
    return math.sqrt(sum(x * x for row in M for x in row))

# =============================================================================
# 1. DATA GENERATION
# =============================================================================

def create_datasets():
    """Generate (n,k)-sparse parity train/test data."""
    # Choose secret subset S
    secret = sorted(random.sample(range(N_BITS), K_SPARSE))
    print(f"[DATA] Secret parity indices S = {secret}")
    print(f"[DATA] Problem: n={N_BITS}, k={K_SPARSE}, N_train={N_TRAIN}, N_test={N_TEST}")

    def make_data(num):
        xs, ys = [], []
        for _ in range(num):
            x = [random.choice([-1.0, 1.0]) for _ in range(N_BITS)]
            y = 1.0
            for idx in secret:
                y *= x[idx]
            xs.append(x)
            ys.append(y)
        return xs, ys

    x_train, y_train = make_data(N_TRAIN)
    x_test, y_test = make_data(N_TEST)
    return x_train, y_train, x_test, y_test, secret

# =============================================================================
# 2. MODEL: raw lists, Kaiming init
# =============================================================================

def init_params():
    """Initialize 2-layer MLP parameters as plain Python lists."""
    std1 = math.sqrt(2.0 / N_BITS)
    std2 = math.sqrt(2.0 / HIDDEN)
    W1 = randn(HIDDEN, N_BITS, std=std1)    # (hidden, n)
    b1 = zeros(HIDDEN)                       # (hidden,)
    W2 = randn(1, HIDDEN, std=std2)          # (1, hidden)
    b2 = zeros(1)                            # (1,)
    total = HIDDEN * N_BITS + HIDDEN + HIDDEN + 1
    print(f"[MODEL] MLP: {N_BITS} → {HIDDEN} → 1  ({total:,} parameters)")
    return W1, b1, W2, b2

# =============================================================================
# 3. FORWARD PASS
# =============================================================================

def forward(x, W1, b1, W2, b2):
    """
    Forward pass for a single sample.
    x → W1·x + b1 → ReLU → W2·h + b2 → scalar output
    Returns (out, h_pre, h)
    """
    # Layer 1
    h_pre = [sum(W1[j][i] * x[i] for i in range(len(x))) + b1[j]
             for j in range(len(W1))]
    # ReLU
    h = [max(0.0, hp) for hp in h_pre]
    # Layer 2
    out = sum(W2[0][j] * h[j] for j in range(len(h))) + b2[0]
    return out, h_pre, h

def forward_batch(xs, W1, b1, W2, b2):
    """Forward pass for a batch. Returns list of outputs."""
    return [forward(x, W1, b1, W2, b2)[0] for x in xs]

# =============================================================================
# 4. LOSS & ACCURACY
# =============================================================================

def hinge_loss_batch(outs, ys):
    """Mean hinge loss over a batch."""
    return sum(max(0.0, 1.0 - o * y) for o, y in zip(outs, ys)) / len(ys)

def accuracy(outs, ys):
    """Fraction where sign(out) == y."""
    correct = sum(1 for o, y in zip(outs, ys)
                  if (1.0 if o >= 0 else -1.0) == y)
    return correct / len(ys)

# =============================================================================
# 5. BACKWARD + SGD UPDATE (manual — the part to customize)
# =============================================================================
# ┌─────────────────────────────────────────────────────────────────────────┐
# │                    🔧 OPTIMIZER INJECTION POINT 🔧                     │
# │                                                                         │
# │  To swap in an experimental learning algorithm, replace the body of     │
# │  backward_and_update(). You have full access to:                        │
# │    - x, y: the single training sample                                   │
# │    - out, h_pre, h: forward pass intermediates                         │
# │    - W1, b1, W2, b2: all model weights as plain Python lists           │
# │                                                                         │
# │  Modify any part of the gradient computation or update rule.            │
# └─────────────────────────────────────────────────────────────────────────┘

def backward_and_update(x, y, out, h_pre, h, W1, b1, W2, b2):
    """
    Manual backward pass + SGD with weight decay for one sample.

    x → [W1, b1] → ReLU → [W2, b2] → out
    L = max(0, 1 - out·y)
    """
    margin = out * y
    if margin >= 1.0:
        return  # hinge loss is 0, no gradient

    # dL/dout = -y
    dout = -y

    # Layer 2 backward: out = W2·h + b2
    dW2_0 = [dout * h[j] for j in range(HIDDEN)]
    db2_0 = dout
    dh = [W2[0][j] * dout for j in range(HIDDEN)]

    # ReLU backward
    dh_pre = [dh[j] * (1.0 if h_pre[j] > 0 else 0.0) for j in range(HIDDEN)]

    # Layer 1 backward: h_pre = W1·x + b1
    # dW1[j][i] = dh_pre[j] * x[i]
    # db1[j] = dh_pre[j]

    # SGD update with weight decay: p -= lr * (grad + wd * p)
    for j in range(HIDDEN):
        for i in range(N_BITS):
            grad = dh_pre[j] * x[i]
            W1[j][i] -= LR * (grad + WD * W1[j][i])
        b1[j] -= LR * (dh_pre[j] + WD * b1[j])

    for j in range(HIDDEN):
        W2[0][j] -= LR * (dW2_0[j] + WD * W2[0][j])
    b2[0] -= LR * (db2_0 + WD * b2[0])

# =============================================================================
# 6. TRAINING LOOP
# =============================================================================

def train(W1, b1, W2, b2, x_train, y_train, x_test, y_test):
    """Single-sample cyclic training with per-step evaluation."""

    # History for plotting
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    step = 0
    gen_epoch = None
    best_test_acc = 0.0
    epochs_no_improve = 0

    print(f"\n[TRAIN] Starting training for up to {MAX_EPOCHS} epochs...")
    print(f"[TRAIN] Mode: single-sample cyclic (batch_size=1, fixed order)")
    print(f"[TRAIN] Steps per epoch: {N_TRAIN}")
    print(f"[TRAIN] Loss: Hinge Loss\n")

    start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        for i in range(N_TRAIN):
            # Forward on single sample
            out, h_pre, h = forward(x_train[i], W1, b1, W2, b2)

            # Backward + update
            backward_and_update(x_train[i], y_train[i], out, h_pre, h,
                                W1, b1, W2, b2)
            step += 1

            # Evaluate full-batch after every step
            tr_outs = forward_batch(x_train, W1, b1, W2, b2)
            te_outs = forward_batch(x_test, W1, b1, W2, b2)
            tr_loss = hinge_loss_batch(tr_outs, y_train)
            te_loss = hinge_loss_batch(te_outs, y_test)
            tr_acc  = accuracy(tr_outs, y_train)
            te_acc  = accuracy(te_outs, y_test)

            train_losses.append(tr_loss)
            test_losses.append(te_loss)
            train_accs.append(tr_acc)
            test_accs.append(te_acc)

            if te_acc >= 1.0 and gen_epoch is None:
                gen_epoch = epoch

            # Logging
            if step % (LOG_EVERY * N_TRAIN) == 0 or step == 1:
                print(f"[STEP {step:>4}] epoch={epoch}  "
                      f"train_loss={tr_loss:.4f}  test_loss={te_loss:.4f}  "
                      f"train_acc={tr_acc:.0%}  test_acc={te_acc:.0%}")

                # Per-sample predictions
                for label, outs, ys in [("Train", tr_outs, y_train),
                                         ("Test",  te_outs, y_test)]:
                    print(f"  {label} predictions:")
                    for j, (o, t) in enumerate(zip(outs, ys)):
                        pred = 1 if o >= 0 else -1
                        tgt = int(t)
                        ok = "✓" if pred == tgt else "✗"
                        print(f"    {j:>2}: f(x)={o:+.4f}  "
                              f"pred={pred:+d}  target={tgt:+d}  {ok}")

                print(f"  Weight norms: |W1|={mat_norm(W1):.4f}  "
                      f"|b1|={norm(b1):.4f}  |W2|={mat_norm(W2):.4f}  "
                      f"|b2|={abs(b2[0]):.4f}")

        # Early stopping
        if gen_epoch is not None:
            elapsed = time.time() - start
            print(f"\n  🎉 GENERALIZED at epoch {epoch}! "
                  f"({step} steps, {elapsed:.1f}s)")
            break

        if te_acc > best_test_acc:
            best_test_acc = te_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\n[TRAIN] Stopping: no improvement in {PATIENCE} epochs.")
            break

    elapsed = time.time() - start
    print(f"[TRAIN] Finished in {elapsed:.1f}s "
          f"({epoch} epochs, {step} total steps)")

    return {
        'train_losses': train_losses, 'test_losses': test_losses,
        'train_accs': train_accs, 'test_accs': test_accs,
        'step_count': step, 'gen_epoch': gen_epoch,
    }

# =============================================================================
# 7. REPORT
# =============================================================================

def print_report(r):
    """Print energy cost report."""
    print(f"\n{'=' * 65}")
    print(f"  ENERGY COST REPORT")
    print(f"{'=' * 65}")
    print(f"  Total optimizer steps: {r['step_count']}")
    if r['gen_epoch'] is not None:
        print(f"  ✅ Perfect generalization at epoch {r['gen_epoch']}")
        print(f"  ⚡ ENERGY COST = {r['gen_epoch']} epochs "
              f"({r['step_count']} optimizer steps)")
    else:
        print(f"  ❌ Did NOT reach 100% test accuracy")
        best = max(r['test_accs'])
        best_step = r['test_accs'].index(best)
        print(f"  Best test accuracy: {best:.0%} at step {best_step}")
        print(f"  ⚡ ENERGY COST = ∞ (failed to generalize)")
    print(f"  Final train loss:     {r['train_losses'][-1]:.6f}")
    print(f"  Final test loss:      {r['test_losses'][-1]:.6f}")
    print(f"  Final train accuracy: {r['train_accs'][-1]:.0%}")
    print(f"  Final test accuracy:  {r['test_accs'][-1]:.0%}")
    print(f"{'=' * 65}")

# =============================================================================
# 8. PLOTTING (the only dependency beyond stdlib)
# =============================================================================

def plot_losses(r, save_path="loss_plot.png"):
    """Plot training and test loss/accuracy curves."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "matplotlib", "--quiet"])
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

    steps = list(range(1, len(r['train_losses']) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(steps, r['train_losses'], label='Train', lw=1.2, color='#2196F3')
    ax1.plot(steps, r['test_losses'],  label='Test',  lw=1.2, color='#F44336')
    ax1.set(xlabel='Step', ylabel='Hinge Loss', title='Loss (per step)')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(steps, r['train_accs'], label='Train', lw=1.2, color='#2196F3')
    ax2.plot(steps, r['test_accs'],  label='Test',  lw=1.2, color='#F44336')
    ax2.set(xlabel='Step', ylabel='Accuracy', title='Accuracy (per step)')
    ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_ylim(-0.05, 1.05)

    if r['gen_epoch'] is not None:
        for ax in (ax1, ax2):
            ax.axvline(x=r['gen_epoch'], color='green', ls='--', alpha=0.7)

    fig.suptitle('Sparse Parity Benchmark — Pure Python',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[PLOT] Saved to: {save_path}")
    plt.close(fig)

# =============================================================================
# 9. MAIN
# =============================================================================

def main():
    total_start = time.time()

    print("=" * 65)
    print("  SPARSE PARITY BENCHMARK — Pure Python, No Dependencies")
    print("=" * 65)

    # --- Init ---
    init_start = time.time()
    x_train, y_train, x_test, y_test, secret = create_datasets()
    print(f"[DATA] Train labels: {[int(y) for y in y_train]}")
    print(f"[DATA] Test labels:  {[int(y) for y in y_test]}")
    W1, b1, W2, b2 = init_params()
    init_elapsed = time.time() - init_start

    # --- Simulation ---
    sim_start = time.time()
    results = train(W1, b1, W2, b2, x_train, y_train, x_test, y_test)
    sim_elapsed = time.time() - sim_start

    # --- Report ---
    print_report(results)

    # --- Plot ---
    plot_start = time.time()
    plot_losses(results)
    plot_elapsed = time.time() - plot_start

    total_elapsed = time.time() - total_start
    print(f"\n[TIMING] Initialization:  {init_elapsed:.3f}s")
    print(f"[TIMING] Simulation:      {sim_elapsed:.3f}s  (training loop)")
    print(f"[TIMING] Plotting:        {plot_elapsed:.3f}s  (matplotlib)")
    print(f"[TIMING] Total wall time: {total_elapsed:.3f}s")

if __name__ == "__main__":
    main()
