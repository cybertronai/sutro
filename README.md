# sutro

**Sparse Parity Benchmark** — an energy-efficiency testbed for neural network learning algorithms.

The [k-sparse parity problem](https://en.wikipedia.org/wiki/Parity_function) is the "Drosophila" of algorithmic benchmarking: simple enough to be tractable, rich enough to exhibit **grokking** (delayed generalization after memorization). The goal is to measure and minimize the *energy cost* — optimizer steps to 100% test accuracy — of different learning algorithms.

## Features

- **Pure Python** — no PyTorch, no NumPy; only `math`, `random`, `time`, and `matplotlib`
- **Manual forward/backward** — all gradients computed by hand; fully transparent
- **Memory reuse distance tracking** — measures how cache-friendly the algorithm is
- **Modular optimizer** — swap in any learning algorithm at the marked injection point

## Architecture

```
x ∈ {-1,1}^n  →  Linear(W1, b1)  →  ReLU  →  Linear(W2, b2)  →  scalar f(x)
Loss: Hinge loss = max(0, 1 - f(x)·y)
Optimizer: SGD with weight decay
```

## Usage

```bash
python sparse_parity_benchmark.py
```

No installation needed beyond Python 3 and `matplotlib` (auto-installed if missing).

## Sample Output

```
=================================================================
  SPARSE PARITY BENCHMARK — Pure Python + Reuse Distance
=================================================================
[DATA] Secret parity indices S = [0, 1, 2]
[DATA] Problem: n=3, k=3, N_train=20, N_test=20
[MODEL] MLP: 3 → 1000 → 1  (5,001 parameters)

[TRAIN] Starting training for up to 2 epochs...
[TRAIN] Mode: single-sample cyclic (batch_size=1, fixed order)
[TRAIN] Steps per epoch: 20

[STEP    1] epoch=1  train_loss=103.8993  test_loss=64.8248  train_acc=60%  test_acc=75%
[STEP   20] epoch=1  train_loss=11.9400  test_loss=28.7414   train_acc=90%  test_acc=70%
[STEP   40] epoch=2  train_loss=0.0000   test_loss=0.0000   train_acc=100%  test_acc=100%
  🎉 GENERALIZED at epoch 2! (40 steps, 2.2s)

=================================================================
  ENERGY COST REPORT
=================================================================
  Total optimizer steps: 40
  ✅ Perfect generalization at epoch 2
  ⚡ ENERGY COST = 2 epochs (40 optimizer steps)
  Final train loss:     0.000000
  Final test loss:      0.000000
  Final train accuracy: 100%
  Final test accuracy:  100%
=================================================================

[TIMING] Initialization:  0.004s
[TIMING] Simulation:      2.236s  (training loop)
[TIMING] Plotting:        1.240s  (matplotlib)
[TIMING] Total wall time: 3.480s
```

## Memory Reuse Distance

The benchmark instruments the **first optimizer step** to measure *memory reuse distance*: the number of floats that flow through memory between when a buffer is written and when it is next read.

**Small distance → still in cache → energy efficient. Large distance → cache miss → expensive.**

The clock advances by buffer size (floats), so a 3,000-float matrix contributes 3,000 to the distance, while a scalar contributes 1. This reflects real cache eviction behavior.

### Per-buffer summary (first step, n=3, hidden=1000)

```
Buffer            Size  Reads    Avg Dist       Min       Max  Cache?
────────────  ────────  ─────  ──────────  ────────  ────────  ──────
x                    3      2       8,510         4    17,015       ✅
W1               3,000      2      13,514     5,008    22,019       ❌
b1               1,000      2      15,514     5,008    26,019       ❌
h_pre            1,000      2       5,504     1,000    10,008       ✅
h                1,000      2       2,003     1,000     3,006       ✅
W2               1,000      3      16,347     9,008    28,019       ❌
b2                   1      2      19,014     9,008    29,020       ❌
out                  1      1           1         1         1       ✅
y                    1      1       9,007     9,007     9,007       ✅
dout                 1      2       1,502         1     3,003       ✅
dh               1,000      1       1,000     1,000     1,000       ✅
dh_pre           1,000      2       4,502     1,000     8,003       ✅
dW2              1,000      1      16,005    16,005    16,005       ❌
db2                  1      1      18,005    18,005    18,005       ❌

Average reuse distance (per-read):        9,716 floats
Average reuse distance (per-float-read):  10,640 floats
Total working set:                        10,008 floats
```

**Key insight:** Parameters (`W1`, `b1`, `W2`, `b2`) all have ❌ — they are written at init, read during forward, then not consumed again until the SGD update at the end of backward. Activations and partial gradients (`h`, `h_pre`, `dh`, `dh_pre`) have ✅ — they are created and immediately consumed.

The size-weighted average reuse distance is the headline metric for comparing algorithms: a better algorithm reorders computation to reduce this number.

## Loss & Accuracy Curves

![Loss & Accuracy Curves](loss_plot.png)

## Customizing the Optimizer

Find the marked section in the code:

```python
# ┌────────────────────────────────────────────────────────┐
# │              🔧 OPTIMIZER INJECTION POINT 🔧           │
# │                                                        │
# │  Replace backward_and_update() with your custom        │
# │  learning algorithm. All buffers are plain Python      │
# │  lists — fully transparent.                            │
# └────────────────────────────────────────────────────────┘
```

Replace the body of `backward_and_update()` with any update rule — gradient normalization, evolutionary perturbation, SVD-based updates, etc.
