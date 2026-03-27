#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///
""" Example Transformer code with shape suffixes.

Based on Noam Shazeer's naming convention illustration, made runnable
with random data and concrete dimensions.

Dimension key:

B: batch size
L: sequence length
M: memory length (length of sequence being attended to)
D: model dimension (sometimes called d_model or embedding_dim)
V: vocabulary size
F: feed-forward subnetwork hidden size
H: number of attention heads in a layer
K: size of each attention key or value (sometimes called d_kv)
"""

import torch
import torch.nn.functional as F
import time

# ---------- hyper-parameters ----------
B = 64       # batch size
L = 1024     # sequence length
D = 512      # model dimension
V = 1024     # vocabulary size
F_DIM = 2048 # feed-forward hidden size
H = 8        # number of attention heads
K = 64       # key/value head dim  (D == H * K)
NUM_LAYERS = 1

# ---------- helpers ----------
def layer_norm(x, params):
    """Simple layer norm with learned gain and bias."""
    return F.layer_norm(x, (x.shape[-1],), weight=params["gain"], bias=params["bias"])


def make_layernorm_params(dim):
    return {"gain": torch.ones(dim), "bias": torch.zeros(dim)}


# ---------- core functions ----------
def transformer(input_token_id_BL, params):
    hidden_BLD = params["embedding_VD"][input_token_id_BL]
    for i in range(params["num_layers"]):
        hidden_BLD = hidden_BLD + attention(hidden_BLD, params["attention_params"][i])
        hidden_BLD = hidden_BLD + ffn(hidden_BLD, params["ffn_params"][i])
    hidden_BLD = layer_norm(hidden_BLD, params["final_layernorm_params"])
    logits_BLV = torch.matmul(hidden_BLD, params["embedding_VD"].T)
    return logits_BLV


def ffn(input_BLD, params):
    input_BLD = layer_norm(input_BLD, params["layernorm_params"])
    hidden_BLF = torch.nn.functional.gelu(torch.matmul(input_BLD, params["w_in_DF"]))
    output_BLD = torch.matmul(hidden_BLF, params["w_out_FD"])
    return output_BLD


def attention(input_BLD, params):
    input_BLD = layer_norm(input_BLD, params["layernorm_params"])
    query_BLHK = torch.einsum('BLD,DHK->BLHK', input_BLD, params["w_q_DHK"])
    key_BMHK   = torch.einsum('BMD,DHK->BMHK', input_BLD, params["w_k_DHK"])
    value_BMHK = torch.einsum('BMD,DHK->BMHK', input_BLD, params["w_v_DHK"])
    logits_BHLM = torch.einsum('BLHK,BMHK->BHLM', query_BLHK, key_BMHK)
    _B, _L, _H, _K = query_BLHK.shape
    logits_BHLM = logits_BHLM / (_K ** 0.5)
    # causal mask: prevent attending to future positions
    masked_out_LM = torch.arange(_L).unsqueeze(1) < torch.arange(_L).unsqueeze(0)
    logits_BHLM = logits_BHLM + torch.where(masked_out_LM, torch.tensor(-float('inf')), torch.tensor(0.0))
    weights_BHLM = torch.softmax(logits_BHLM, dim=-1)
    wtd_values_BLHK = torch.einsum('BHLM,BMHK->BLHK', weights_BHLM, value_BMHK)
    out_BLD = torch.einsum('BLHK,HKD->BLD', wtd_values_BLHK, params["w_o_HKD"])
    return out_BLD


# ---------- initialise random parameters ----------
def make_params():
    attention_params = []
    ffn_params = []
    for _ in range(NUM_LAYERS):
        attention_params.append({
            "layernorm_params": make_layernorm_params(D),
            "w_q_DHK": torch.randn(D, H, K) * 0.02,
            "w_k_DHK": torch.randn(D, H, K) * 0.02,
            "w_v_DHK": torch.randn(D, H, K) * 0.02,
            "w_o_HKD": torch.randn(H, K, D) * 0.02,
        })
        ffn_params.append({
            "layernorm_params": make_layernorm_params(D),
            "w_in_DF":  torch.randn(D, F_DIM) * 0.02,
            "w_out_FD": torch.randn(F_DIM, D) * 0.02,
        })
    return {
        "num_layers": NUM_LAYERS,
        "embedding_VD": torch.randn(V, D) * 0.02,
        "attention_params": attention_params,
        "ffn_params": ffn_params,
        "final_layernorm_params": make_layernorm_params(D),
    }


# ---------- main ----------
if __name__ == "__main__":
    torch.manual_seed(42)
    params = make_params()
    input_token_id_BL = torch.randint(0, V, (B, L))

    # Warmup run (first call has overhead from lazy op compilation)
    _ = transformer(input_token_id_BL, params)

    # Profiled run
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        logits_BLV = transformer(input_token_id_BL, params)

    # Export timeline for chrome://tracing or https://ui.perfetto.dev
    trace_path = "trace.json"
    prof.export_chrome_trace(trace_path)

    # Print summary table
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    print(f"\nInput shape:  {tuple(input_token_id_BL.shape)}  (B, L)")
    print(f"Output shape: {tuple(logits_BLV.shape)}  (B, L, V)")
    print(f"Output mean:  {logits_BLV.mean().item():.6f}")
    print(f"Output std:   {logits_BLV.std().item():.6f}")
    print(f"\nTimeline saved to {trace_path}")
    print("Open in chrome://tracing or https://ui.perfetto.dev")