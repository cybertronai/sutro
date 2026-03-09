#!/usr/bin/env python3
"""
Experiment: Gaussian Elimination over GF(2) for Sparse Parity

Hypothesis: Treating each sample as a linear equation over GF(2) (binary field),
Gaussian elimination recovers the secret parity bits in O(n^2) time with only
n+1 samples. This is the theoretically optimal approach for pure parity.

The key insight: parity over {-1,+1} is equivalent to XOR over {0,1}.
  - Convert inputs: x_bit = (x+1)/2  so -1->0, +1->1
  - Convert labels: y_bit = (y+1)/2  so -1->0, +1->1
  - For odd k:  y_bit = XOR(x_bit[secret]) = sum(x_bit[secret]) mod 2
  - For even k: y_bit = 1 - XOR(x_bit[secret]) (the relationship is inverted)
  - Since we don't know k's parity a priori, we solve BOTH A*s=b and A*s=(1-b)
  - This is a linear system over GF(2): Gaussian elimination finds s
  - The solution vector has 1s at the secret bit positions.

Dependencies: numpy (pip install numpy)

Usage:
    python3 exp_gf2_standalone.py
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from math import comb


# =============================================================================
# MEMORY REUSE DISTANCE TRACKER (inlined from sparse_parity.tracker)
# =============================================================================

class MemTracker:
    """
    Tracks Average Reuse Distance (ARD) — a proxy for energy efficiency.

    Clock advances by buffer SIZE (floats), not operation count.
    Small ARD = data stays in cache = cheap.
    Large ARD = cache miss = expensive external memory access.
    """

    def __init__(self):
        self.clock = 0
        self._write_time = {}
        self._write_size = {}
        self._events = []

    def write(self, name, size):
        """Record writing `size` floats to buffer `name`."""
        self._write_time[name] = self.clock
        self._write_size[name] = size
        self._events.append(('W', name, size, self.clock, None))
        self.clock += size

    def read(self, name, size=None):
        """Record reading from buffer `name`. Returns reuse distance."""
        if size is None:
            size = self._write_size.get(name, 0)
        if name in self._write_time:
            distance = self.clock - self._write_time[name]
        else:
            distance = -1
        self._events.append(('R', name, size, self.clock, distance))
        self.clock += size
        return distance

    def summary(self):
        """Compute summary statistics."""
        reads = [(name, size, dist) for typ, name, size, _, dist in self._events
                 if typ == 'R' and dist >= 0]
        writes = [e for e in self._events if e[0] == 'W']

        if not reads:
            return {'total_floats_accessed': self.clock, 'reads': 0, 'writes': len(writes),
                    'weighted_ard': 0, 'per_buffer': {}}

        total_float_dist = sum(s * d for _, s, d in reads)
        total_floats = sum(s for _, s, _ in reads)
        weighted_ard = total_float_dist / total_floats if total_floats > 0 else 0

        per_buffer = {}
        for name, size, dist in reads:
            if name not in per_buffer:
                per_buffer[name] = {'size': size, 'distances': []}
            per_buffer[name]['distances'].append(dist)

        for name, info in per_buffer.items():
            dists = info['distances']
            info['avg_dist'] = sum(dists) / len(dists)
            info['min_dist'] = min(dists)
            info['max_dist'] = max(dists)
            info['read_count'] = len(dists)

        return {
            'total_floats_accessed': self.clock,
            'reads': len(reads),
            'writes': len(writes),
            'weighted_ard': weighted_ard,
            'total_floats_read': total_floats,
            'per_buffer': per_buffer,
        }

    def to_json(self):
        """Return JSON-serializable dict of all metrics."""
        return self.summary()

    def report(self):
        """Print human-readable report."""
        s = self.summary()
        print(f"\n{'=' * 70}")
        print(f"  MEMORY REUSE DISTANCE REPORT")
        print(f"{'=' * 70}")
        print(f"  Total floats accessed: {s['total_floats_accessed']:,}")
        print(f"  Operations: {s['reads']} reads, {s['writes']} writes")
        print(f"  Weighted ARD: {s['weighted_ard']:,.0f} floats")
        if s['per_buffer']:
            print(f"\n  {'Buffer':<12} {'Size':>8} {'Reads':>5} {'Avg Dist':>10} {'Min':>8} {'Max':>8}")
            print(f"  {'─'*12} {'─'*8} {'─'*5} {'─'*10} {'─'*8} {'─'*8}")
            for name, info in s['per_buffer'].items():
                print(f"  {name:<12} {info['size']:>8,} {info['read_count']:>5} "
                      f"{info['avg_dist']:>10,.0f} {info['min_dist']:>8,} {info['max_dist']:>8,}")
        print(f"{'=' * 70}")


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_data(n_bits, k_sparse, n_samples, seed=42):
    """Generate sparse parity data. Returns x, y, secret."""
    rng = np.random.RandomState(seed)
    secret = sorted(rng.choice(n_bits, k_sparse, replace=False).tolist())
    x = rng.choice([-1.0, 1.0], size=(n_samples, n_bits))
    y = np.prod(x[:, secret], axis=1)
    return x, y, secret


# =============================================================================
# GF(2) GAUSSIAN ELIMINATION
# =============================================================================

def gf2_gauss_elim(A, b, tracker=None):
    """
    Solve A * s = b over GF(2) using Gaussian elimination with partial pivoting.
    Fully instrumented for memory reuse distance tracking.

    A: (m x n) binary matrix (numpy uint8, values 0 or 1)
    b: (m,) binary vector

    Returns:
        solution: (n,) binary vector s such that A*s = b (mod 2), or None if inconsistent
        rank: rank of the augmented matrix
    """
    m, n = A.shape
    row_len = n + 1  # width of augmented matrix

    # Build augmented matrix [A | b]
    if tracker:
        tracker.read('A_gf2', m * n)
        tracker.read('b_gf2', m)

    aug = np.zeros((m, row_len), dtype=np.uint8)
    aug[:, :n] = A
    aug[:, n] = b

    if tracker:
        tracker.write('aug', m * row_len)

    pivot_row = 0
    pivot_cols = []

    for col in range(n):
        # Pivot search: scan column from pivot_row to m
        remaining = m - pivot_row
        if remaining <= 0:
            break

        if tracker:
            tracker.read('aug_col', remaining)  # read 1 element per remaining row

        found = -1
        for row in range(pivot_row, m):
            if aug[row, col] == 1:
                found = row
                break

        if found == -1:
            continue  # no pivot in this column, skip

        # Row swap (if needed)
        if found != pivot_row:
            if tracker:
                tracker.read('aug_row', 2 * row_len)   # read both rows
            aug[[pivot_row, found]] = aug[[found, pivot_row]]
            if tracker:
                tracker.write('aug_row', 2 * row_len)  # write both rows

        pivot_cols.append(col)

        # Eliminate: scan column, then XOR for each row with a 1
        if tracker:
            tracker.read('aug_col', m)  # scan column for rows to eliminate

        for row in range(m):
            if row != pivot_row and aug[row, col] == 1:
                if tracker:
                    tracker.read('aug_row', 2 * row_len)  # read pivot + target row
                aug[row] = aug[row] ^ aug[pivot_row]
                if tracker:
                    tracker.write('aug_row', row_len)      # write modified row

        pivot_row += 1

    rank = pivot_row

    # Consistency check: read remaining rows
    if tracker and rank < m:
        tracker.read('aug_row', (m - rank) * row_len)

    for row in range(rank, m):
        if aug[row, n] == 1:
            return None, rank  # inconsistent

    # Back-substitute: read pivot positions from aug, write solution
    if tracker:
        tracker.read('aug_row', rank * row_len)  # read reduced rows

    solution = np.zeros(n, dtype=np.uint8)
    for i, col in enumerate(pivot_cols):
        solution[col] = aug[i, n]

    if tracker:
        tracker.write('solution', n)

    return solution, rank


def gf2_solve(x, y, n_bits, tracker=None):
    """
    Convert {-1,+1} data to GF(2) and solve with Gaussian elimination.

    For odd k: y_bit = XOR(x_bit[S]), so solve A*s = b
    For even k: y_bit = 1 - XOR(x_bit[S]), so solve A*s = (1-b)
    Since k is unknown, we try both and return whichever yields a valid solution.

    Returns (predicted_secret, solution_vector, rank).
    """
    n_samples = x.shape[0]

    if tracker:
        tracker.write('x_input', x.size)

    # Convert to GF(2): x_bit = (x+1)/2, y_bit = (y+1)/2
    if tracker:
        tracker.read('x_input', x.size)

    A = ((x + 1) / 2).astype(np.uint8)
    b = ((y + 1) / 2).astype(np.uint8)

    if tracker:
        tracker.write('A_gf2', A.size)
        tracker.write('y_input', n_samples)
        tracker.read('y_input', n_samples)
        tracker.write('b_gf2', n_samples)

    # Try both b (odd k) and 1-b (even k)
    # Only the first attempt is tracked (the important one)
    solutions = []
    for i, b_try in enumerate([b, (1 - b).astype(np.uint8)]):
        t = tracker if (i == 0 and tracker) else None
        solution, rank = gf2_gauss_elim(A.copy(), b_try.copy(), tracker=t)
        if solution is not None:
            predicted = sorted(np.where(solution == 1)[0].tolist())
            solutions.append((predicted, solution, rank))

    if not solutions:
        return None, None, rank

    # Verify: which solution produces correct labels?
    for predicted, solution, rank in solutions:
        if len(predicted) > 0:
            if tracker:
                tracker.read('x_input', x.size)
            y_check = np.prod(x[:, predicted], axis=1)
            if np.all(y_check == y):
                return predicted, solution, rank

    predicted, solution, rank = solutions[0]
    return predicted, solution, rank


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_config(n_bits, k_sparse, n_samples_list, seeds, verbose=True):
    """Run GF(2) solver on one (n, k) config with varying sample counts."""
    c_n_k = comb(n_bits, k_sparse)
    if verbose:
        print(f"\n  Config: n={n_bits}, k={k_sparse}, C(n,k)={c_n_k:,}")

    results = []
    first_tracker = None

    for n_samples in n_samples_list:
        seed_results = []
        for seed in seeds:
            x, y, secret = generate_data(n_bits, k_sparse, n_samples, seed=seed)

            use_tracker = (seed == seeds[0] and n_samples == n_samples_list[0])
            tracker = MemTracker() if use_tracker else None

            start = time.time()
            predicted, solution, rank = gf2_solve(x, y, n_bits, tracker=tracker)
            elapsed = time.time() - start

            correct = (predicted == secret) if predicted is not None else False

            # Verify on test data
            if predicted is not None and len(predicted) > 0:
                rng_te = np.random.RandomState(seed + 1000)
                x_te = rng_te.choice([-1.0, 1.0], size=(1000, n_bits))
                y_te = np.prod(x_te[:, secret], axis=1)
                y_pred = np.prod(x_te[:, predicted], axis=1)
                test_acc = float(np.mean(y_pred == y_te))
            else:
                test_acc = 0.0

            seed_result = {
                'seed': seed,
                'n_samples': n_samples,
                'secret': secret,
                'predicted': predicted,
                'correct': correct,
                'test_acc': round(test_acc, 4),
                'rank': int(rank),
                'elapsed_s': round(elapsed, 8),
                'k_found': len(predicted) if predicted is not None else 0,
            }

            if tracker:
                seed_result['tracker'] = tracker.to_json()
                if first_tracker is None:
                    first_tracker = tracker

            seed_results.append(seed_result)

        n_correct = sum(1 for r in seed_results if r['correct'])
        avg_time = np.mean([r['elapsed_s'] for r in seed_results])
        avg_test_acc = np.mean([r['test_acc'] for r in seed_results])

        if verbose:
            status = f"{n_correct}/{len(seeds)} correct" if n_correct > 0 else "ALL FAILED"
            print(f"    n_samples={n_samples:>5}: {status}, "
                  f"avg time={avg_time*1e6:.1f}us, avg test_acc={avg_test_acc:.0%}")

        results.append({
            'n_samples': n_samples,
            'n_correct': n_correct,
            'n_total': len(seeds),
            'avg_time_s': round(float(avg_time), 8),
            'avg_time_us': round(float(avg_time * 1e6), 2),
            'avg_test_acc': round(float(avg_test_acc), 4),
            'per_seed': seed_results,
        })

    # Print tracker report for first config
    if first_tracker:
        first_tracker.report()

    return {
        'n_bits': n_bits,
        'k_sparse': k_sparse,
        'c_n_k': c_n_k,
        'results': results,
    }


def main():
    print("=" * 70)
    print("  EXPERIMENT: Gaussian Elimination over GF(2)")
    print("  Theoretically optimal for pure parity: O(n^2), microseconds")
    print("=" * 70)

    seeds = [42, 43, 44]
    all_results = {}

    # Sample counts to test
    sample_counts_small = [21, 40, 50, 100, 500]   # n+1 = 21 for n=20
    sample_counts_medium = [51, 100, 200, 500]      # n+1 = 51 for n=50
    sample_counts_large = [101, 200, 500]            # n+1 = 101 for n=100

    # -------------------------------------------------------------------
    # Config 1: n=20, k=3
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 1: n=20, k=3")
    print("=" * 70)
    all_results['n20_k3'] = run_config(
        n_bits=20, k_sparse=3,
        n_samples_list=sample_counts_small,
        seeds=seeds,
    )

    # -------------------------------------------------------------------
    # Config 2: n=50, k=3
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 2: n=50, k=3")
    print("=" * 70)
    all_results['n50_k3'] = run_config(
        n_bits=50, k_sparse=3,
        n_samples_list=sample_counts_medium,
        seeds=seeds,
    )

    # -------------------------------------------------------------------
    # Config 3: n=100, k=3
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 3: n=100, k=3")
    print("=" * 70)
    all_results['n100_k3'] = run_config(
        n_bits=100, k_sparse=3,
        n_samples_list=sample_counts_large,
        seeds=seeds,
    )

    # -------------------------------------------------------------------
    # Config 4: n=20, k=5
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 4: n=20, k=5")
    print("=" * 70)
    all_results['n20_k5'] = run_config(
        n_bits=20, k_sparse=5,
        n_samples_list=sample_counts_small,
        seeds=seeds,
    )

    # -------------------------------------------------------------------
    # Config 5: n=20, k=7
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 5: n=20, k=7")
    print("=" * 70)
    all_results['n20_k7'] = run_config(
        n_bits=20, k_sparse=7,
        n_samples_list=sample_counts_small,
        seeds=seeds,
    )

    # -------------------------------------------------------------------
    # Config 6: n=20, k=10
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 6: n=20, k=10")
    print("=" * 70)
    all_results['n20_k10'] = run_config(
        n_bits=20, k_sparse=10,
        n_samples_list=sample_counts_small,
        seeds=seeds,
    )

    # -------------------------------------------------------------------
    # Sample complexity deep dive: n=20, k=3
    # How few samples can we get away with?
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SAMPLE COMPLEXITY: n=20, k=3 -- minimum samples needed?")
    print("=" * 70)
    sample_complexity_results = []
    for n_samp in [5, 10, 15, 18, 19, 20, 21, 22, 25, 30, 40, 50, 100]:
        correct_count = 0
        total_time = 0
        for seed in seeds:
            x, y, secret = generate_data(20, 3, n_samp, seed=seed)
            start = time.time()
            predicted, _, rank = gf2_solve(x, y, 20)
            elapsed = time.time() - start
            if predicted == secret:
                correct_count += 1
            total_time += elapsed
        avg_time = total_time / len(seeds)
        print(f"    n_samples={n_samp:>4}: {correct_count}/{len(seeds)} correct, "
              f"avg {avg_time*1e6:.1f}us")
        sample_complexity_results.append({
            'n_samples': n_samp,
            'correct': correct_count,
            'total': len(seeds),
            'avg_time_us': round(avg_time * 1e6, 2),
        })
    all_results['sample_complexity'] = sample_complexity_results

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("  SUMMARY TABLE")
    print("=" * 90)
    header = (f"  {'Config':<15} | {'C(n,k)':>10} | {'n_samp':>7} | "
              f"{'Correct':>7} | {'Avg Time':>12} | {'Test Acc':>8}")
    print(header)
    print("  " + "-" * 75)

    for key, res in all_results.items():
        if key == 'sample_complexity':
            continue
        n_b = res['n_bits']
        k_s = res['k_sparse']
        c = res['c_n_k']
        # Show best result (highest n_samples)
        best = res['results'][-1]  # last entry = most samples
        print(f"  n={n_b},k={k_s:<8} | {c:>10,} | {best['n_samples']:>7} | "
              f"{best['n_correct']}/{best['n_total']:>5} | "
              f"{best['avg_time_us']:>9.1f} us | {best['avg_test_acc']:>7.0%}")

    # -------------------------------------------------------------------
    # Comparison with other approaches
    # -------------------------------------------------------------------
    print("\n  " + "=" * 85)
    print("  COMPARISON WITH OTHER APPROACHES")
    print("  " + "=" * 85)
    print(f"  {'Config':<20} | {'Method':<20} | {'Time':>12} | {'Samples':>8} | {'Notes'}")
    print("  " + "-" * 80)

    # GF(2) results
    for key in ['n20_k3', 'n50_k3', 'n100_k3', 'n20_k5', 'n20_k7', 'n20_k10']:
        if key not in all_results:
            continue
        res = all_results[key]
        best = res['results'][-1]
        # Also show minimum working sample count
        min_working = None
        for r in res['results']:
            if r['n_correct'] == r['n_total']:
                min_working = r['n_samples']
                break
        min_str = f"min={min_working}" if min_working else "partial"
        print(f"  {key:<20} | {'GF(2) Gauss':<20} | "
              f"{best['avg_time_us']:>9.1f} us | {best['n_samples']:>8} | "
              f"{min_str}")

    print(f"  {'n20_k3':<20} | {'SGD baseline':<20} | {'120,000 us':>12} | {'10000':>8} | ~5 epochs")
    print(f"  {'n50_k3':<20} | {'SGD (curriculum)':<20} | {'---':>12} | {'10000':>8} | 20 epochs")
    print(f"  {'n50_k3':<20} | {'SGD (direct)':<20} | {'---':>12} | {'10000':>8} | FAIL (54%)")
    print(f"  {'n20_k3':<20} | {'Fourier exh.':<20} | {'~3,000 us':>12} | {'500':>8} | C(20,3)=1140 subsets")
    print(f"  {'n20_k3':<20} | {'Random search':<20} | {'~11,000 us':>12} | {'500':>8} | ~881 tries")
    print("=" * 90)

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    results_dir = Path(__file__).resolve().parent / 'results_gf2'
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'experiment': 'exp_gf2',
            'description': 'Gaussian Elimination over GF(2) for sparse parity',
            'hypothesis': 'GF(2) Gaussian elimination solves sparse parity in O(n^2) with n+1 samples',
            'approach': 'blank_slate -- no neural net, no SGD, no gradients',
            'configs': all_results,
        }, f, indent=2, default=str)

    print(f"\n  Results saved to: {results_path}")
    return all_results


if __name__ == '__main__':
    main()
