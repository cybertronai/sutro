import modal
import time
import subprocess

app = modal.App("gpu-toy-problem")

image = modal.Image.debian_slim(python_version="3.11").pip_install("torch")

# Modal L4 pricing (https://modal.com/pricing)
L4_COST_PER_HOUR = 0.84  # USD/GPU-hour
L4_COST_PER_SEC = L4_COST_PER_HOUR / 3600


@app.function(gpu="L4", image=image)
def gpu_toy():
    """Run a toy problem on an NVIDIA L4 GPU."""
    import torch

    container_start = time.time()

    # Show GPU info via nvidia-smi
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(result.stdout)

    # Toy problem: large matrix multiply on GPU
    device = torch.device("cuda")
    print(f"PyTorch sees: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

    N = 4096
    print(f"\nMultiplying two {N}x{N} matrices on GPU...")

    A = torch.randn(N, N, device=device)
    B = torch.randn(N, N, device=device)

    # Warm up
    torch.cuda.synchronize()
    C = A @ B
    torch.cuda.synchronize()

    # Timed run
    start = time.time()
    for _ in range(100):
        C = A @ B
    torch.cuda.synchronize()
    compute_elapsed = time.time() - start

    container_elapsed = time.time() - container_start
    tflops = (2 * N**3 * 100) / compute_elapsed / 1e12
    print(f"100 matmuls in {compute_elapsed:.3f}s  ({tflops:.2f} TFLOPS)")
    print(f"Result shape: {C.shape}, sum: {C.sum().item():.2f}")

    return {
        "compute_elapsed": compute_elapsed,
        "container_elapsed": container_elapsed,
        "tflops": tflops,
    }


@app.local_entrypoint()
def main():
    wall_start = time.time()
    result = gpu_toy.remote()
    wall_elapsed = time.time() - wall_start

    # Modal bills per-second of container uptime (wall time includes startup)
    gpu_cost = wall_elapsed * L4_COST_PER_SEC

    print(f"\n{'=' * 55}")
    print(f"  GPU TFLOPS:          {result['tflops']:.2f}")
    print(f"  GPU compute time:    {result['compute_elapsed']:.3f}s")
    print(f"  Container time:      {result['container_elapsed']:.1f}s")
    print(f"  Total wall time:     {wall_elapsed:.1f}s  (incl. startup)")
    print(f"  Estimated cost:      ${gpu_cost:.4f}  "
          f"(L4 @ ${L4_COST_PER_HOUR}/hr)")
    print(f"{'=' * 55}")
