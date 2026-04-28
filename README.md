# MLIP Inference Bench

Throughput benchmarks for top machine-learning interatomic potentials (MLIPs) from [matbench-discovery](https://matbench-discovery.materialsproject.org/), run on NVIDIA A100 GPUs via [Modal](https://modal.com) and powered by [TorchSim](https://github.com/TorchSim/torch-sim).

## Models

| Model | Origin | matbench-discovery Rank | TorchSim Wrapper |
|-------|--------|:-----------------------:|------------------|
| **EquiformerV2-UMA** | FAIR (Meta) | 1 | `fairchem` |
| **PET-MAD** | COSMO Lab | 2 | `metatomic` |
| **eSEN-SM-OC25** | FAIR Chemistry | 4 | `fairchem` |
| **Nequip-OAM-S** | MIR Group | 6 | `nequip` |
| **ORB-v3-Conservative** | Orbital Materials | ~15 | `orb` |
| **ORB-v3-Direct** | Orbital Materials | ~15 | `orb` |

## Results (NVIDIA A100-SXM4-40GB)

Real benchmark data — 100 timed forward passes after 10 warmup steps, `torch.cuda.synchronize()` for accurate GPU timing. PyTorch 2.8.0, CUDA 12.4.

### 64-atom FCC Cu system, batch size 16

| Model | Single (ms/step) | Batched (ms/step) | Batched atoms/s | Peak Memory |
|-------|------------------:|------------------:|----------------:|------------:|
| **Nequip-OAM-S** | 11.7 | **11.5** | **89,043** | 397 MB |
| ORB-v3-Direct | 10.7 | 36.4 | 28,132 | 1.2 GB |
| ORB-v3-Conservative | 18.2 | 72.5 | 14,124 | 5.2 GB |
| PET-MAD | 26.9 | 105.0 | 9,752 | 1.6 GB |
| EquiformerV2-UMA | 71.8 | 751.6 | 1,362 | 6.7 GB |

### Key findings

- **Nequip is the fastest in batched throughput** — 89k atoms/s, 3.2x faster than ORB-v3-Direct and 65x faster than UMA, while using only 397 MB
- **Nequip achieves near-perfect batch parallelism** — batched latency (11.5 ms) actually *lower* than single-system (11.7 ms), meaning the GPU is maximally utilized
- **ORB-v3-Direct is fastest in single-system latency** (10.7 ms) and 2x faster than ORB-v3-Conservative in batched mode, with 4.3x less memory (1.2 GB vs 5.2 GB)
- **ORB-v3-Conservative trades speed for energy conservation** — useful for long MD trajectories where energy drift matters
- **PET-MAD offers a middle ground** — 7x faster than UMA with moderate memory usage (1.6 GB)
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
