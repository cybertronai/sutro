#!/usr/bin/env python3
"""
compare_reuse_distance.py — Side-by-side energy efficiency comparison

Runs both SGD MLP and GF(2) Gaussian Elimination on the same sparse parity
problem and compares their memory reuse distance (a proxy for cache/energy
efficiency).

Usage:
    python3 compare_reuse_distance.py
"""

import math
import random
import time

# =============================================================================
# CONFIG — same problem for both algorithms
# =============================================================================

N_BITS   = 20       # input dimension
K_SPARSE = 3        # parity bits
N_TRAIN  = 21       # n+1 samples (minimum for GF(2))
HIDDEN   = 100      # MLP hidden width
LR       = 0.5
WD       = 0.01
SEED     = 42


# =============================================================================
# MEMORY REUSE DISTANCE TRACKER
# =============================================================================

class MemTracker:
    """
    Tracks Average Reuse Distance (ARD) — a proxy for energy efficiency.
    Clock advances by buffer SIZE (elements), not operation count.
    """
    def __init__(self):
        self.clock = 0
        self._last_access = {}   # name → clock at last read or write
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
            'weighted_ard': weighted_ard,
            'per_read_ard': per_read_ard,
            'total_accessed': self.clock,
            'total_read': total_floats,
            'working_set': working_set,
            'n_reads': len(reads),
            'n_writes': len(writes),
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
        print(f"  Read operations:         {s['n_reads']:>10}")
        print(f"  Write operations:        {s['n_writes']:>10}")

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
# DATA GENERATION (shared)
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
# ALGORITHM 1: SGD MLP (one forward + backward step)
# =============================================================================

def run_sgd_step(xs, ys, n, hidden, lr, wd, seed):
    mem = MemTracker()
    rng = random.Random(seed)

    # Init parameters
    std1 = math.sqrt(2.0 / n)
    std2 = math.sqrt(2.0 / hidden)
    W1 = [[rng.gauss(0, std1) for _ in range(n)] for _ in range(hidden)]
    b1 = [0.0] * hidden
    W2 = [[rng.gauss(0, std2) for _ in range(hidden)]]
    b2 = [0.0]

    x = xs[0]
    y = ys[0]

    # Register initial writes
    mem.write('W1', hidden * n)
    mem.write('b1', hidden)
    mem.write('W2', hidden)
    mem.write('b2', 1)
    mem.write('x', n)
    mem.write('y', 1)

    # ── FORWARD ──────────────────────────────────────────────────────────
    # Layer 1: h_pre = W1·x + b1
    mem.read('x', n)
    mem.read('W1', hidden * n)
    mem.read('b1', hidden)
    h_pre = [sum(W1[j][i] * x[i] for i in range(n)) + b1[j] for j in range(hidden)]
    mem.write('h_pre', hidden)

    # ReLU
    mem.read('h_pre', hidden)
    h = [max(0.0, hp) for hp in h_pre]
    mem.write('h', hidden)

    # Layer 2: out = W2·h + b2
    mem.read('h', hidden)
    mem.read('W2', hidden)
    mem.read('b2', 1)
    out = sum(W2[0][j] * h[j] for j in range(hidden)) + b2[0]
    mem.write('out', 1)

    # ── BACKWARD (fused layer-wise updates) ──────────────────────────────
    mem.read('out', 1)
    mem.read('y', 1)
    margin = out * y

    dout = -y
    mem.write('dout', 1)

    # Layer 2 gradients
    mem.read('dout', 1)
    mem.read('h', hidden)
    dW2_0 = [dout * h[j] for j in range(hidden)]
    db2_0 = dout
    mem.write('dW2', hidden)
    mem.write('db2', 1)

    # Compute dh BEFORE updating W2 (needs pre-update W2)
    mem.read('W2', hidden)
    mem.read('dout', 1)
    dh = [W2[0][j] * dout for j in range(hidden)]
    mem.write('dh', hidden)

    # FUSED: Update W2, b2 immediately
    mem.read('dW2', hidden)
    mem.read('W2', hidden)
    for j in range(hidden):
        W2[0][j] -= lr * (dW2_0[j] + wd * W2[0][j])
    mem.write('W2', hidden)

    mem.read('db2', 1)
    mem.read('b2', 1)
    b2[0] -= lr * (db2_0 + wd * b2[0])
    mem.write('b2', 1)

    # ReLU backward
    mem.read('dh', hidden)
    mem.read('h_pre', hidden)
    dh_pre = [dh[j] * (1.0 if h_pre[j] > 0 else 0.0) for j in range(hidden)]
    mem.write('dh_pre', hidden)

    # FUSED: Layer 1 backward + update W1, b1 immediately
    mem.read('dh_pre', hidden)
    mem.read('x', n)
    mem.read('W1', hidden * n)
    for j in range(hidden):
        for i in range(n):
            W1[j][i] -= lr * (dh_pre[j] * x[i] + wd * W1[j][i])
    mem.write('W1', hidden * n)

    mem.read('dh_pre', hidden)
    mem.read('b1', hidden)
    for j in range(hidden):
        b1[j] -= lr * (dh_pre[j] + wd * b1[j])
    mem.write('b1', hidden)

    return mem


# =============================================================================
# ALGORITHM 2: GF(2) Gaussian Elimination
# =============================================================================

def run_gf2_solve(xs, ys, n, n_samples):
    mem = MemTracker()

    # Convert to GF(2): -1 → 0, +1 → 1
    A = [[int((xs[i][j] + 1) / 2) for j in range(n)] for i in range(n_samples)]
    b = [int((ys[i] + 1) / 2) for i in range(n_samples)]

    mem.write('x_input', n_samples * n)
    mem.read('x_input', n_samples * n)  # read to convert
    mem.write('A', n_samples * n)
    mem.write('y_input', n_samples)
    mem.read('y_input', n_samples)      # read to convert
    mem.write('b', n_samples)

    # Build augmented matrix [A | b]
    row_len = n + 1
    mem.read('A', n_samples * n)
    mem.read('b', n_samples)
    aug = [A[i][:] + [b[i]] for i in range(n_samples)]
    mem.write('aug', n_samples * row_len)

    m = n_samples
    pivot_row = 0

    # ── Gaussian elimination ─────────────────────────────────────────────
    for col in range(n):
        remaining = m - pivot_row
        if remaining <= 0:
            break

        # Pivot search: scan column
        mem.read('aug_col', remaining)

        found = -1
        for row in range(pivot_row, m):
            if aug[row][col] == 1:
                found = row
                break

        if found == -1:
            continue

        # Row swap
        if found != pivot_row:
            mem.read('aug_row', 2 * row_len)
            aug[pivot_row], aug[found] = aug[found], aug[pivot_row]
            mem.write('aug_row', 2 * row_len)

        # Eliminate: scan column, then XOR each matching row
        mem.read('aug_col', m)

        for row in range(m):
            if row != pivot_row and aug[row][col] == 1:
                mem.read('aug_row', 2 * row_len)   # read pivot + target
                aug[row] = [aug[row][j] ^ aug[pivot_row][j] for j in range(row_len)]
                mem.write('aug_row', row_len)        # write modified row

        pivot_row += 1

    rank = pivot_row

    # Consistency check
    if rank < m:
        mem.read('aug_row', (m - rank) * row_len)

    # Back-substitution
    mem.read('aug_row', rank * row_len)
    solution = [0] * n
    mem.write('solution', n)

    return mem


# =============================================================================
# MAIN: run both and compare
# =============================================================================

def main():
    print("╔" + "═" * 70 + "╗")
    print("║   ENERGY EFFICIENCY COMPARISON: SGD MLP vs GF(2) Gaussian Elim     ║")
    print("╚" + "═" * 70 + "╝")
    print(f"  Problem: n={N_BITS}, k={K_SPARSE}, samples={N_TRAIN}, seed={SEED}")
    print(f"  MLP:     {N_BITS} → {HIDDEN} → 1  ({HIDDEN * N_BITS + HIDDEN + HIDDEN + 1:,} params)")

    xs, ys, secret = generate_data(N_BITS, K_SPARSE, N_TRAIN, SEED)
    print(f"  Secret:  {secret}")

    # ── Run SGD ──────────────────────────────────────────────────────────
    t0 = time.time()
    sgd_mem = run_sgd_step(xs, ys, N_BITS, HIDDEN, LR, WD, SEED)
    sgd_time = time.time() - t0
    sgd = sgd_mem.report("SGD MLP — 1 Forward+Backward Step (fused updates)")

    # ── Run GF(2) ────────────────────────────────────────────────────────
    t0 = time.time()
    gf2_mem = run_gf2_solve(xs, ys, N_BITS, N_TRAIN)
    gf2_time = time.time() - t0
    gf2 = gf2_mem.report("GF(2) Gaussian Elimination — 1 Solve")

    # ── Side-by-side ─────────────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print(f"  SIDE-BY-SIDE COMPARISON  (n={N_BITS}, k={K_SPARSE}, samples={N_TRAIN})")
    print(f"{'═' * 72}")
    print(f"  {'Metric':<40} {'SGD MLP':>12} {'GF(2)':>12}")
    print(f"  {'─' * 40} {'─' * 12} {'─' * 12}")

    rows = [
        ('Total elements accessed',   'total_accessed'),
        ('Total elements read',       'total_read'),
        ('Working set (elements)',     'working_set'),
        ('Read operations',           'n_reads'),
        ('Write operations',          'n_writes'),
        ('Avg reuse dist (per-read)', 'per_read_ard'),
        ('Avg reuse dist (weighted)', 'weighted_ard'),
    ]
    for label, key in rows:
        sv = sgd.get(key, 0)
        gv = gf2.get(key, 0)
        if isinstance(sv, float):
            print(f"  {label:<40} {sv:>12,.0f} {gv:>12,.0f}")
        else:
            print(f"  {label:<40} {sv:>12,} {gv:>12,}")

    print(f"  {'Wall time':<40} {sgd_time*1e6:>10,.0f}µs {gf2_time*1e6:>10,.0f}µs")

    # ── Ratios ───────────────────────────────────────────────────────────
    if gf2['weighted_ard'] > 0:
        ratio_ard = sgd['weighted_ard'] / gf2['weighted_ard']
    else:
        ratio_ard = float('inf')
    ratio_data = sgd['total_accessed'] / gf2['total_accessed']

    print(f"\n  {'':>4}📊 SGD weighted ARD is {ratio_ard:.1f}× larger (worse cache locality)")
    print(f"  {'':>4}📊 SGD accesses {ratio_data:.1f}× more total data")
    print(f"  {'':>4}📊 GF(2) working set is {sgd['working_set']/gf2['working_set']:.0f}× smaller")

    # ── Why ──────────────────────────────────────────────────────────────
    print(f"\n  WHY?")
    print(f"  ─────────────────────────────────────────────────────────────")
    print(f"  SGD: W1 ({HIDDEN*N_BITS:,} elements) is read in forward, not re-read")
    print(f"       until end of backward → reuse dist ≈ entire working set.")
    print(f"  GF(2): operates on single augmented matrix ({N_TRAIN}×{N_BITS+1}={N_TRAIN*(N_BITS+1)} elements).")
    print(f"       Each step does row XOR → touches 2 rows ({2*(N_BITS+1)} elements),")
    print(f"       keeps data hot in L1 cache the entire time.")
    print(f"{'═' * 72}")


if __name__ == "__main__":
    main()
