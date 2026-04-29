# MLIP Inference Bench

Throughput benchmarks for top machine-learning interatomic potentials (MLIPs) from [matbench-discovery](https://matbench-discovery.materialsproject.org/), run on NVIDIA A100 GPUs via [Modal](https://modal.com) and powered by [TorchSim](https://github.com/TorchSim/torch-sim).

**Interactive dashboard: [adeeshkolluru.com/benchmarks](https://adeeshkolluru.com/benchmarks/)**

## Results (NVIDIA A100-SXM4-40GB)

64-atom FCC Cu cell, batch size 16, 100 timed steps after 10 warmup, `torch.cuda.synchronize()` for accurate GPU timing. PyTorch 2.8.0, CUDA 12.4.

### Leaderboard models

| Model (checkpoint) | CPS | Single (ms) | Batched (ms) | Atoms/s | Memory |
|---------------------|----:|------------:|-------------:|--------:|-------:|
| **ORB-v3-Direct** (inf, OMAT) | — | 16.8 | **42.7** | **23,974** | 1.2 GB |
| **ORB-v3-Conservative** (inf, OMAT) | 0.860 | 29.6 | 84.5 | 12,118 | 5.2 GB |
| **PET-OAM-XL** | 0.898 | 143.6 | 1,333.2 | 768 | 29.7 GB |
| **NequIP-OAM-XL** | 0.886 | 181.0 | 1,911.5 | 536 | 34.5 GB |

### Smaller / other variants

| Model (checkpoint) | Single (ms) | Batched (ms) | Atoms/s | Memory |
|---------------------|------------:|-------------:|--------:|-------:|
| **NequIP-OAM-S** | 29.3 | **29.9** | **34,247** | 531 MB |
| **PET-MAD-S** | 49.7 | 142.6 | 7,180 | 1.8 GB |
| **UMA-S-1p1** | 135.0 | 607.5 | 1,686 | 6.7 GB |

CPS = Combined Performance Score from [matbench-discovery](https://matbench-discovery.materialsproject.org/). ORB-v3 CPS 0.860 is for the conservative-inf-mpa checkpoint. ORB-v3-Direct is not separately scored. UMA-S-1p1 (Meta's Universal Model for Atoms) is not on the leaderboard. EquiformerV3+DeNS-OAM (CPS 0.902) and eSEN are not yet benchmarked (no TorchSim wrapper / gated HF repo).

### Key findings

- **ORB-v3-Direct is the fastest leaderboard model** — 24k atoms/s batched, 2x faster than Conservative and 31x faster than PET-OAM-XL
- **NequIP-OAM-S is the fastest small model** — 34k atoms/s with near-perfect batch parallelism (29.9 ms batched vs 29.3 ms single)
- **XL models are memory-bound** — NequIP-OAM-XL and PET-OAM-XL use 30-35 GB, nearly saturating the A100 40GB. Batching barely helps
- **Accuracy vs speed tradeoff** — PET-OAM-XL has the highest benchmarked CPS (0.898) but is 31x slower than ORB-v3-Direct

## Models

| Checkpoint | Architecture | Origin | TorchSim wrapper |
|------------|-------------|--------|-----------------|
| **NequIP-OAM-XL / S** | E(3)-equivariant message passing | MIR Group (Harvard) | `nequip_framework` |
| **PET-OAM-XL / MAD-S** | Point Edge Transformer | COSMO Lab (EPFL) | `metatomic` |
| **ORB-v3-Conservative** | Graph network (conservative) | Orbital Materials | `orb` |
| **ORB-v3-Direct** | Graph network (direct) | Orbital Materials | `orb` |
| **UMA-S-1p1** | Universal Model for Atoms | FAIR (Meta) | `fairchem` |

## Methodology

Each model runs forward passes on a 64-atom FCC copper supercell using [TorchSim](https://github.com/TorchSim/torch-sim)'s batched API on A100 GPUs via [Modal](https://modal.com). 10 warmup steps are excluded, then 100 steps are timed with `torch.cuda.synchronize()` before and after. Batch size is 16 independent copies of the same system.

## Quickstart

```bash
# Run benchmarks on Modal A100
uv run modal run modal_app.py

# Quick test (small systems only)
uv run modal run modal_app.py --quick

# Deploy API endpoint
uv run modal deploy modal_app.py
```

## Project Structure

```
├── benchmark/
│   ├── config.py            # Model specs, system sizes, timing params
│   └── run_benchmark.py     # TorchSim benchmark loop
├── frontend/
│   └── index.html           # Local results dashboard
├── results/
│   └── sample_results.json  # Real A100 benchmark data
├── modal_app.py             # Modal A100 deployment + JSON API
└── pyproject.toml
```

## License

MIT
