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

## Results (NVIDIA A100-SXM4-40GB)

Real benchmark data — 100 timed forward passes after 10 warmup steps, `torch.cuda.synchronize()` before/after timing.

### 64-atom FCC Cu system, batch size 16

| Model | Single (ms/step) | Batched (ms/step) | Batched atoms/s | Peak Memory |
|-------|------------------:|------------------:|----------------:|------------:|
| **Nequip-OAM-S** | 16.6 | **17.3** | **59,190** | 394 MB |
| PET-MAD | 31.8 | 134.0 | 7,642 | 3.6 GB |
| EquiformerV2-UMA | 128.6 | 782.8 | 1,308 | 6.7 GB |

### Key findings

- **Nequip is the fastest by a wide margin** — 7.7x faster than PET and 45x faster than UMA in batched throughput, while using 17x less memory
- **Nequip's batched latency matches single-system latency** (17.3 vs 16.6 ms), indicating near-perfect batch parallelism
- **UMA (EquiformerV2-based) is the most expensive** — highest latency and memory footprint, reflecting the cost of its attention mechanism
- **PET-MAD offers a middle ground** — 3.8x faster than UMA with moderate memory usage

### Models not yet benchmarked

- **eSEN-SM-OC25**: Blocked by gated HuggingFace repo (`facebook/OMol25`). Requires requesting access.
- **ORB-v3**: API mismatch between `orb-models` and `torch-sim` wrappers. Needs package update.

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

Features:
- Toggle between single and batched modes
- Toggle between atoms/s and ms/step metrics
- Ranking cards, throughput/latency charts, memory bars, speedup comparison
- Load from Modal API, local JSON, or bundled sample data

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
