# MLIP Inference Bench

Throughput benchmarks for top machine-learning interatomic potentials (MLIPs) from [matbench-discovery](https://matbench-discovery.materialsproject.org/), run on NVIDIA A100 GPUs via [Modal](https://modal.com) and powered by [TorchSim](https://github.com/TorchSim/torch-sim).

## Models

| Model | Origin | matbench-discovery Rank | TorchSim Wrapper |
|-------|--------|:-----------------------:|------------------|
| **EquiformerV2-UMA** | FAIR (Meta) | 1 | `fairchem` |
| **PET-MAD** | COSMO Lab | 2 | `metatomic` |
| **eSEN-SM-OC25** | FAIR Chemistry | 4 | `fairchem` |
| **Nequip-OAM-S** | MIR Group | 6 | `nequip` |
| **ORB-v3** | Orbital Materials | ~15 | `orb` |

## Results (NVIDIA A100 80GB PCIe)

Real benchmark data — 100 timed forward passes after 10 warmup steps, `torch.cuda.synchronize()` for accurate GPU timing. PyTorch 2.8.0, CUDA 12.4.

### 64-atom FCC Cu system, batch size 16

| Model | Single (ms/step) | Batched (ms/step) | Batched atoms/s | Peak Memory |
|-------|------------------:|------------------:|----------------:|------------:|
| **Nequip-OAM-S** | 15.1 | **15.5** | **66,065** | 397 MB |
| ORB-v3 | 21.2 | 74.7 | 13,708 | 5.2 GB |
| PET-MAD | 30.3 | 109.2 | 9,378 | 1.6 GB |
| EquiformerV2-UMA | 80.4 | 701.9 | 1,459 | 6.7 GB |

### Key findings

- **Nequip is the fastest by a wide margin** — 4.8x faster than ORB-v3 and 45x faster than UMA in batched throughput, while using 17x less memory than UMA
- **Nequip achieves near-perfect batch parallelism** — batched latency (15.5 ms) barely exceeds single-system latency (15.1 ms), meaning the GPU is efficiently utilized
- **ORB-v3 is the second fastest** — strong single-system performance (21.2 ms) with good batched scaling
- **PET-MAD offers a middle ground** — 2.7x faster than UMA with moderate memory usage (1.6 GB)
- **UMA (EquiformerV2-based) is the most expensive** — highest latency and memory footprint, reflecting the cost of the full attention mechanism

### eSEN status

eSEN-SM-OC25 is blocked by a gated HuggingFace repo (`facebook/OC25`). Request access at [huggingface.co/facebook/OC25](https://huggingface.co/facebook/OC25) to benchmark it.

## Methodology

Each model runs inference on FCC copper supercells using TorchSim's batched forward pass API on A100 GPUs via Modal.

```
10 warmup steps  →  100 timed forward passes
torch.cuda.synchronize() before/after timing
```

**Metrics per model per system size:**
- Atoms/second (raw throughput)
- ms/step (wall-clock latency per forward pass)
- Peak GPU memory (MB)
- Batched speedup (ratio of batched to single-system throughput)

## Quickstart

```bash
# Run benchmarks on Modal A100
uv run modal run modal_app.py

# Quick test (small systems only)
uv run modal run modal_app.py --quick

# Deploy API endpoint
uv run modal deploy modal_app.py
```

## Frontend

An interactive Chart.js dashboard in `frontend/index.html` visualizes the results.

```bash
python3 -m http.server 8765
# open http://localhost:8765/frontend/
```

## Project Structure

```
├── benchmark/
│   ├── config.py            # Model specs, system sizes, timing params
│   └── run_benchmark.py     # TorchSim benchmark loop
├── frontend/
│   └── index.html           # Interactive results dashboard
├── results/
│   └── sample_results.json  # Real A100 benchmark data
├── modal_app.py             # Modal A100 deployment + JSON API
└── pyproject.toml
```

## License

MIT
