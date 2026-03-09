#!/usr/bin/env python3
"""
compare_reuse_distance.py — Side-by-side energy efficiency comparison

Runs both SGD MLP and GF(2) Gaussian Elimination on the same sparse parity
problem and compares: reuse distance, runtime, FLOPs, and memory reads.

Usage:
    python3 compare_reuse_distance.py
"""

import math
import random
import time

# =============================================================================
# CONFIG
# =============================================================================

# Reuse distance comparison (per-step)
N_BITS   = 20
K_SPARSE = 3
N_TRAIN  = 21
HIDDEN   = 100

# Convergence timing (small enough for SGD to converge in pure Python)
CONV_N      = 3
CONV_K      = 3
CONV_TRAIN  = 20
CONV_TEST   = 20
CONV_HIDDEN = 1000

LR = 0.5
WD = 0.01
SEED = 42


# =============================================================================
# MEMORY REUSE DISTANCE TRACKER
# =============================================================================

class MemTracker:
    def __init__(self):
        self.clock = 0
        self._last_access = {}
        self._write_size = {}
        self._events = []

    def write(self, name, size):
        self._last_access[name] = self.clock
        self._write_size[name] = size
        self._events.append(('W', name, size, self.clock, None))
        self.clock += size

    def read(self, name, size=None):
        if size is None:
            size = self._write_size.get(name, 0)
        dist = self.clock - self._last_access[name] if name in self._last_access else -1
        self._events.append(('R', name, size, self.clock, dist))
        self._last_access[name] = self.clock
        self.clock += size
        return dist

    def stats(self):
        reads = [(n, s, d) for t, n, s, _, d in self._events if t == 'R' and d >= 0]
        writes = [e for e in self._events if e[0] == 'W']
        if not reads:
            return {}
        total_float_dist = sum(s * d for _, s, d in reads)
        total_floats = sum(s for _, s, _ in reads)
        weighted_ard = total_float_dist / total_floats if total_floats else 0
        per_read_ard = sum(d for _, _, d in reads) / len(reads)
        buf = {}
        for n, s, d in reads:
            buf.setdefault(n, {'size': s, 'dists': []})['dists'].append(d)
        working_set = sum(self._write_size.values())
        return {
            'weighted_ard': weighted_ard, 'per_read_ard': per_read_ard,
            'total_accessed': self.clock, 'total_read': total_floats,
            'working_set': working_set, 'n_reads': len(reads), 'n_writes': len(writes),
            'per_buffer': buf,
        }

    def report(self, title):
        s = self.stats()
        if not s:
            print(f"  No data for {title}")
            return s
        ws = s['working_set']
        print(f"\n{'═' * 72}")
        print(f"  {title}")
        print(f"{'═' * 72}")
        print(f"  Total elements accessed: {s['total_accessed']:>10,}")
        print(f"  Total elements read:     {s['total_read']:>10,}")
        print(f"  Working set:             {s['working_set']:>10,}")

        print(f"\n  {'Buffer':<14} {'Size':>8} {'Reads':>5} {'Avg Dist':>10}"
              f" {'Min':>8} {'Max':>8} {'Cache?':>6}")
        print(f"  {'─'*14} {'─'*8} {'─'*5} {'─'*10} {'─'*8} {'─'*8} {'─'*6}")
        for name, info in s['per_buffer'].items():
            dists = info['dists']
            sz = info['size']
            avg = sum(dists) / len(dists)
            mn, mx = min(dists), max(dists)
            cached = "✅" if avg < ws else "❌"
            print(f"  {name:<14} {sz:>8,} {len(dists):>5} {avg:>10,.0f}"
                  f" {mn:>8,} {mx:>8,} {cached:>6}")

        print(f"\n  Avg reuse distance (per-read):       {s['per_read_ard']:>10,.0f} elements")
        print(f"  Avg reuse distance (per-elem-read):  {s['weighted_ard']:>10,.0f} elements")
        print(f"{'═' * 72}")
        return s


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_data(n, k, n_samples, seed):
    rng = random.Random(seed)
    secret = sorted(rng.sample(range(n), k))
    xs = [[rng.choice([-1.0, 1.0]) for _ in range(n)] for _ in range(n_samples)]
    ys = []
    for x in xs:
        val = 1.0
        for idx in secret:
            val *= x[idx]
        ys.append(val)
    return xs, ys, secret


# =============================================================================
# ALGORITHM 1: SGD MLP — reuse distance (1 step)
# =============================================================================

def run_sgd_step(xs, ys, n, hidden, lr, wd, seed):
    mem = MemTracker()
    rng = random.Random(seed)

    std1 = math.sqrt(2.0 / n)
    std2 = math.sqrt(2.0 / hidden)
    W1 = [[rng.gauss(0, std1) for _ in range(n)] for _ in range(hidden)]
    b1 = [0.0] * hidden
    W2 = [[rng.gauss(0, std2) for _ in range(hidden)]]
    b2 = [0.0]
    x, y = xs[0], ys[0]

    mem.write('W1', hidden * n); mem.write('b1', hidden)
    mem.write('W2', hidden);    mem.write('b2', 1)
    mem.write('x', n);          mem.write('y', 1)

    # Forward
    mem.read('x', n); mem.read('W1', hidden * n); mem.read('b1', hidden)
    h_pre = [sum(W1[j][i] * x[i] for i in range(n)) + b1[j] for j in range(hidden)]
    mem.write('h_pre', hidden)
    mem.read('h_pre', hidden)
    h = [max(0.0, hp) for hp in h_pre]
    mem.write('h', hidden)
    mem.read('h', hidden); mem.read('W2', hidden); mem.read('b2', 1)
    out = sum(W2[0][j] * h[j] for j in range(hidden)) + b2[0]
    mem.write('out', 1)

    # Backward (fused)
    mem.read('out', 1); mem.read('y', 1)
    dout = -y
    mem.write('dout', 1)
    mem.read('dout', 1); mem.read('h', hidden)
    dW2_0 = [dout * h[j] for j in range(hidden)]
    mem.write('dW2', hidden); mem.write('db2', 1)
    mem.read('W2', hidden); mem.read('dout', 1)
    dh = [W2[0][j] * dout for j in range(hidden)]
    mem.write('dh', hidden)
    mem.read('dW2', hidden); mem.read('W2', hidden)
    for j in range(hidden):
        W2[0][j] -= lr * (dW2_0[j] + wd * W2[0][j])
    mem.write('W2', hidden)
    mem.read('db2', 1); mem.read('b2', 1)
    b2[0] -= lr * (dout + wd * b2[0])
    mem.write('b2', 1)
    mem.read('dh', hidden); mem.read('h_pre', hidden)
    dh_pre = [dh[j] * (1.0 if h_pre[j] > 0 else 0.0) for j in range(hidden)]
    mem.write('dh_pre', hidden)
    mem.read('dh_pre', hidden); mem.read('x', n); mem.read('W1', hidden * n)
    for j in range(hidden):
        for i in range(n):
            W1[j][i] -= lr * (dh_pre[j] * x[i] + wd * W1[j][i])
    mem.write('W1', hidden * n)
    mem.read('dh_pre', hidden); mem.read('b1', hidden)
    for j in range(hidden):
        b1[j] -= lr * (dh_pre[j] + wd * b1[j])
    mem.write('b1', hidden)
    return mem


# =============================================================================
# ALGORITHM 2: GF(2) — reuse distance (1 solve)
# =============================================================================

def run_gf2_solve(xs, ys, n, n_samples):
    mem = MemTracker()
    A = [[int((xs[i][j] + 1) / 2) for j in range(n)] for i in range(n_samples)]
    b = [int((ys[i] + 1) / 2) for i in range(n_samples)]

    mem.write('x_input', n_samples * n)
    mem.read('x_input', n_samples * n)
    mem.write('A', n_samples * n)
    mem.write('y_input', n_samples)
    mem.read('y_input', n_samples)
    mem.write('b', n_samples)

    row_len = n + 1
    mem.read('A', n_samples * n); mem.read('b', n_samples)
    aug = [A[i][:] + [b[i]] for i in range(n_samples)]
    mem.write('aug', n_samples * row_len)

    m = n_samples
    pivot_row = 0
    for col in range(n):
        remaining = m - pivot_row
        if remaining <= 0:
            break
        mem.read('aug_col', remaining)
        found = -1
        for row in range(pivot_row, m):
            if aug[row][col] == 1:
                found = row
                break
        if found == -1:
            continue
        if found != pivot_row:
            mem.read('aug_row', 2 * row_len)
            aug[pivot_row], aug[found] = aug[found], aug[pivot_row]
            mem.write('aug_row', 2 * row_len)
        mem.read('aug_col', m)
        for row in range(m):
            if row != pivot_row and aug[row][col] == 1:
                mem.read('aug_row', 2 * row_len)
                aug[row] = [aug[row][j] ^ aug[pivot_row][j] for j in range(row_len)]
                mem.write('aug_row', row_len)
        pivot_row += 1

    if pivot_row < m:
        mem.read('aug_row', (m - pivot_row) * row_len)
    mem.read('aug_row', pivot_row * row_len)
    mem.write('solution', n)
    return mem


# =============================================================================
# ANALYTICAL FLOP AND MEMORY READ COUNTING
# =============================================================================

def sgd_flops_per_step(n, h):
    """FLOPs for one SGD forward+backward+update step."""
    # Forward: h_pre=W1·x+b1: 2nh, ReLU: h, out=W2·h+b2: 2h
    fwd = 2*n*h + h + 2*h
    # Backward+update: margin(2), dout(1), dW2(h), dh(h),
    #   W2 update(4h), b2 update(4), dh_pre(2h), W1 update(5nh), b1 update(4h)
    bwd = 3 + h + h + 4*h + 4 + 2*h + 5*n*h + 4*h
    return fwd, bwd

def sgd_reads_per_step(n, h):
    """Element reads for one SGD forward+backward+update step."""
    # Forward: x(n)+W1(nh)+b1(h)+h_pre(h)+h(h)+W2(h)+b2(1)
    fwd = n + n*h + h + h + h + h + 1
    # Backward: out(1)+y(1)+dout(2)+h(h)+W2(h)+dout(1)+dW2(h)+W2(h)+db2(1)+b2(1)+
    #           dh(h)+h_pre(h)+dh_pre(h)+x(n)+W1(nh)+dh_pre(h)+b1(h)
    bwd = 1 + 1 + 1 + h + h + 1 + h + h + 1 + 1 + h + h + h + n + n*h + h + h
    return fwd, bwd

def sgd_fwd_reads(n, h):
    """Reads for one forward pass only (test evaluation)."""
    return n + n*h + h + h + h + h + 1

def gf2_flops_and_reads(n, m):
    """FLOPs and reads for GF(2) solve on m×n system (analytical worst-case)."""
    row_len = n + 1
    k = min(n, m)  # number of pivots

    # Conversion: m*n adds + m*n divs + m adds + m divs
    conv_flops = 2*m*n + 2*m
    conv_reads = m*n + m  # read x_input + y_input

    # Build augmented: read A + b
    build_reads = m*n + m

    # Gaussian elimination
    elim_flops = 0
    elim_reads = 0
    for col in range(k):
        rows_left = m - col
        elim_reads += rows_left          # pivot scan (column)
        elim_reads += m                  # elimination scan (column)
        elim_flops += rows_left + m      # comparisons
        # Avg rows eliminated per column ≈ m/2 (rough)
        avg_elim = max(1, m // 2)
        elim_reads += avg_elim * 2 * row_len  # read pivot + target rows
        elim_flops += avg_elim * row_len       # XOR operations
        # Row swap (sometimes)
        elim_reads += 2 * row_len

    # Back-substitution
    backsub_reads = k * row_len
    backsub_flops = k

    # Verification (on test data, counted separately)
    total_flops = conv_flops + elim_flops + backsub_flops
    total_reads = conv_reads + build_reads + elim_reads + backsub_reads
    return total_flops, total_reads


# =============================================================================
# CONVERGENCE FUNCTIONS
# =============================================================================

def sgd_forward_fn(x, W1, b1, W2, b2, n, hidden):
    h_pre = [sum(W1[j][i] * x[i] for i in range(n)) + b1[j] for j in range(hidden)]
    h = [max(0.0, hp) for hp in h_pre]
    out = sum(W2[0][j] * h[j] for j in range(hidden)) + b2[0]
    return out, h_pre, h

def sgd_bwd_fn(x, y, out, h_pre, h, W1, b1, W2, b2, n, hidden, lr, wd):
    margin = out * y
    if margin >= 1.0:
        return False
    dout = -y
    dW2_0 = [dout * h[j] for j in range(hidden)]
    dh = [W2[0][j] * dout for j in range(hidden)]
    for j in range(hidden):
        W2[0][j] -= lr * (dW2_0[j] + wd * W2[0][j])
    b2[0] -= lr * (dout + wd * b2[0])
    dh_pre = [dh[j] * (1.0 if h_pre[j] > 0 else 0.0) for j in range(hidden)]
    for j in range(hidden):
        for i in range(n):
            W1[j][i] -= lr * (dh_pre[j] * x[i] + wd * W1[j][i])
    for j in range(hidden):
        b1[j] -= lr * (dh_pre[j] + wd * b1[j])
    return True

def run_sgd_to_convergence(xs_train, ys_train, xs_test, ys_test, n, hidden, lr, wd, seed, max_epochs=50):
    rng = random.Random(seed)
    std1 = math.sqrt(2.0 / n)
    std2 = math.sqrt(2.0 / hidden)
    W1 = [[rng.gauss(0, std1) for _ in range(n)] for _ in range(hidden)]
    b1 = [0.0] * hidden
    W2 = [[rng.gauss(0, std2) for _ in range(hidden)]]
    b2 = [0.0]

    flops_fwd, flops_bwd = sgd_flops_per_step(n, hidden)
    reads_fwd, reads_bwd = sgd_reads_per_step(n, hidden)
    fwd_reads_only = sgd_fwd_reads(n, hidden)

    t0 = time.time()
    steps = 0
    total_flops = 0
    total_reads = 0

    for epoch in range(1, max_epochs + 1):
        for x, y in zip(xs_train, ys_train):
            out, h_pre, h = sgd_forward_fn(x, W1, b1, W2, b2, n, hidden)
            updated = sgd_bwd_fn(x, y, out, h_pre, h, W1, b1, W2, b2, n, hidden, lr, wd)
            steps += 1
            total_flops += flops_fwd + (flops_bwd if updated else 2)
            total_reads += reads_fwd + (reads_bwd if updated else 2)

        # Test accuracy
        outs = [sgd_forward_fn(x, W1, b1, W2, b2, n, hidden)[0] for x in xs_test]
        correct = sum(1 for o, y in zip(outs, ys_test) if (1.0 if o >= 0 else -1.0) == y)
        acc = correct / len(ys_test)
        total_flops += len(xs_test) * (flops_fwd + 1)
        total_reads += len(xs_test) * fwd_reads_only

        if acc == 1.0:
            return time.time() - t0, steps, acc, epoch, total_flops, total_reads

    return time.time() - t0, steps, acc, max_epochs, total_flops, total_reads


def gf2_gauss_solve(A, b_vec, n):
    m = len(A)
    row_len = n + 1
    aug = [A[i][:] + [b_vec[i]] for i in range(m)]
    pivot_row = 0
    for col in range(n):
        if pivot_row >= m:
            break
        found = -1
        for row in range(pivot_row, m):
            if aug[row][col] == 1:
                found = row
                break
        if found == -1:
            continue
        if found != pivot_row:
            aug[pivot_row], aug[found] = aug[found], aug[pivot_row]
        for row in range(m):
            if row != pivot_row and aug[row][col] == 1:
                aug[row] = [aug[row][j] ^ aug[pivot_row][j] for j in range(row_len)]
        pivot_row += 1
    solution = [0] * n
    for i in range(pivot_row):
        for c in range(n):
            if aug[i][c] == 1:
                solution[c] = aug[i][n]
                break
    return sorted([j for j in range(n) if solution[j] == 1])


def run_gf2_to_accuracy(xs_train, ys_train, xs_test, ys_test, n):
    n_samples = len(xs_train)
    A = [[int((xs_train[i][j] + 1) / 2) for j in range(n)] for i in range(n_samples)]
    b_vec = [int((ys_train[i] + 1) / 2) for i in range(n_samples)]
    b_flip = [1 - b for b in b_vec]

    solve_flops, solve_reads = gf2_flops_and_reads(n, n_samples)
    # Verification on test data
    verify_flops = len(xs_test) * n   # k multiplications per test sample
    verify_reads = len(xs_test) * n   # read k elements per test sample
    total_flops = solve_flops + verify_flops
    total_reads = solve_reads + verify_reads

    t0 = time.time()
    best_acc = 0.0
    for b_try in [b_vec, b_flip]:
        predicted = gf2_gauss_solve(A, b_try, n)
        if not predicted:
            continue
        correct = sum(1 for x, y in zip(xs_test, ys_test)
                      if (1.0 if all(True for _ in []) else
                          math.prod(x[idx] for idx in predicted)) == y)
        # simpler:
        correct = 0
        for x, y in zip(xs_test, ys_test):
            val = 1.0
            for idx in predicted:
                val *= x[idx]
            if val == y:
                correct += 1
        acc = correct / len(ys_test)
        if acc > best_acc:
            best_acc = acc

    elapsed = time.time() - t0
    return elapsed, best_acc, total_flops, total_reads


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("╔" + "═" * 70 + "╗")
    print("║   ENERGY EFFICIENCY COMPARISON: SGD MLP vs GF(2) Gaussian Elim     ║")
    print("╚" + "═" * 70 + "╝")

    # ═══════════════════════════════════════════════════════════════════════
    # PART 1: REUSE DISTANCE (per-step, n=20)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n  REUSE DISTANCE COMPARISON (n={N_BITS}, k={K_SPARSE}, hidden={HIDDEN})")
    print(f"  MLP: {N_BITS} → {HIDDEN} → 1  ({HIDDEN*N_BITS + HIDDEN + HIDDEN + 1:,} params)")

    xs, ys, secret = generate_data(N_BITS, K_SPARSE, N_TRAIN, SEED)
    print(f"  Secret: {secret}")

    sgd_mem = run_sgd_step(xs, ys, N_BITS, HIDDEN, LR, WD, SEED)
    sgd = sgd_mem.report(f"SGD MLP — 1 Step (n={N_BITS}, hidden={HIDDEN})")

    gf2_mem = run_gf2_solve(xs, ys, N_BITS, N_TRAIN)
    gf2 = gf2_mem.report(f"GF(2) Gaussian Elim — 1 Solve (n={N_BITS}, samples={N_TRAIN})")

    # ═══════════════════════════════════════════════════════════════════════
    # PART 2: TIME, FLOPs, READS TO 100% ACCURACY (n=3)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 72}")
    print(f"  TIME / FLOPs / READS TO 100% ACCURACY  (n={CONV_N}, k={CONV_K})")
    print(f"  SGD: {CONV_N} → {CONV_HIDDEN} → 1  |  GF(2): {CONV_TRAIN} samples")
    print(f"{'═' * 72}")

    xs_test, ys_test, _ = generate_data(CONV_N, CONV_K, CONV_TEST, SEED + 1000)

    # GF(2) — 5 seeds
    gf2_results = []
    for s in range(SEED, SEED + 5):
        xs_g, ys_g, _ = generate_data(CONV_N, CONV_K, CONV_TRAIN, s)
        t, acc, flops, reads = run_gf2_to_accuracy(xs_g, ys_g, xs_test, ys_test, CONV_N)
        gf2_results.append((t, acc, flops, reads))
        print(f"  GF(2) seed={s}:  {t*1e6:>8,.0f}µs  {flops:>8,} FLOPs  {reads:>8,} reads  acc={acc:.0%}")

    gf2_ok = [(t, f, r) for t, a, f, r in gf2_results if a == 1.0]
    gf2_avg_t = sum(t for t, _, _ in gf2_ok) / len(gf2_ok) if gf2_ok else float('inf')
    gf2_avg_f = sum(f for _, f, _ in gf2_ok) / len(gf2_ok) if gf2_ok else 0
    gf2_avg_r = sum(r for _, _, r in gf2_ok) / len(gf2_ok) if gf2_ok else 0

    # SGD — 5 seeds
    sgd_results = []
    for s in range(SEED, SEED + 5):
        xs_s, ys_s, _ = generate_data(CONV_N, CONV_K, CONV_TRAIN, s)
        t, steps, acc, epochs, flops, reads = run_sgd_to_convergence(
            xs_s, ys_s, xs_test, ys_test, CONV_N, CONV_HIDDEN, LR, WD, s, max_epochs=50)
        sgd_results.append((t, acc, flops, reads, steps, epochs))
        if acc == 1.0:
            print(f"  SGD  seed={s}:  {t*1e6:>8,.0f}µs  {flops:>8,} FLOPs  {reads:>8,} reads  "
                  f"{epochs}ep/{steps}steps")
        else:
            print(f"  SGD  seed={s}:  FAILED ({acc:.0%}) after {epochs} epochs")

    sgd_ok = [(t, f, r) for t, a, f, r, _, _ in sgd_results if a == 1.0]
    sgd_avg_t = sum(t for t, _, _ in sgd_ok) / len(sgd_ok) if sgd_ok else float('inf')
    sgd_avg_f = sum(f for _, f, _ in sgd_ok) / len(sgd_ok) if sgd_ok else 0
    sgd_avg_r = sum(r for _, _, r in sgd_ok) / len(sgd_ok) if sgd_ok else 0

    # ═══════════════════════════════════════════════════════════════════════
    # SIDE-BY-SIDE SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 72}")
    print(f"  SIDE-BY-SIDE SUMMARY")
    print(f"{'═' * 72}")
    print(f"  {'Metric':<40} {'SGD MLP':>14} {'GF(2)':>14}")
    print(f"  {'─' * 40} {'─' * 14} {'─' * 14}")

    # Reuse distance (n=20)
    for label, key in [('Weighted avg reuse distance', 'weighted_ard'),
                       ('Working set (elements)', 'working_set')]:
        sv = sgd.get(key, 0)
        gv = gf2.get(key, 0)
        print(f"  {label:<40} {sv:>14,.0f} {gv:>14,.0f}")

    # Convergence metrics (n=3)
    print(f"  {'Time to 100% accuracy':<40} {sgd_avg_t:>12.3f}s {gf2_avg_t:>10.6f}s")
    print(f"  {'Total FLOPs to 100%':<40} {sgd_avg_f:>14,.0f} {gf2_avg_f:>14,.0f}")
    print(f"  {'Total memory reads to 100%':<40} {sgd_avg_r:>14,.0f} {gf2_avg_r:>14,.0f}")

    # Ratios
    ratio_ard = sgd['weighted_ard'] / gf2['weighted_ard'] if gf2['weighted_ard'] > 0 else float('inf')
    ratio_time = sgd_avg_t / gf2_avg_t if gf2_avg_t > 0 else float('inf')
    ratio_flops = sgd_avg_f / gf2_avg_f if gf2_avg_f > 0 else float('inf')
    ratio_reads = sgd_avg_r / gf2_avg_r if gf2_avg_r > 0 else float('inf')

    print(f"\n      📊 GF(2) weighted ARD:   {ratio_ard:.0f}× smaller")
    print(f"      📊 GF(2) time:           {ratio_time:,.0f}× faster")
    print(f"      📊 GF(2) FLOPs:          {ratio_flops:,.0f}× fewer")
    print(f"      📊 GF(2) memory reads:   {ratio_reads:,.0f}× fewer")
    print(f"{'═' * 72}")


if __name__ == "__main__":
    main()
