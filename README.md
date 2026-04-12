# MLIP Inference Bench

Throughput benchmarks for the top machine-learning interatomic potentials (MLIPs) from [matbench-discovery](https://matbench-discovery.materialsproject.org/), run on NVIDIA A100 GPUs via [Modal](https://modal.com) and powered by [TorchSim](https://github.com/TorchSim/torch-sim).

## Models

| Model | Origin | matbench-discovery Rank | TorchSim Wrapper |
|-------|--------|:-----------------------:|------------------|
| **EquiformerV3-OAM** | Atomic Architects | 1 | `fairchem` |
| **PET-OAM-XL** | COSMO Lab | 2 | `metatomic` |
| **eSEN-30M-OAM** | FAIR Chemistry | 4 | `fairchem` |
| **Nequip-OAM-XL** | MIR Group | 6 | `nequip` |
| **ORB-v3** | Orbital Materials | ~15 | `orb` |

## Methodology

Each model is benchmarked on FCC copper supercells across five system sizes (32 — 2,916 atoms), in both **single-system** and **batched** (16x) modes.

```
┌─────────────────────────────────────────────┐
│  10 warmup steps  →  100 timed forward passes  │
│  torch.cuda.synchronize() before/after timing   │
└─────────────────────────────────────────────┘
```

**Metrics collected per model per system size:**
- **Atoms/second** — raw throughput
- **ms/step** — wall-clock latency per forward pass
- **Peak GPU memory** (MB)
- **Batched speedup** — ratio of batched to single-system throughput

## Sample Results (A100 80GB)

Largest system size (2,916 atoms), batched mode:

| Model | Throughput | Latency | Peak Memory |
|-------|----------:|--------:|------------:|
| ORB-v3 | **443k atoms/s** | 105 ms | 22.8 GB |
| eSEN-30M-OAM | 203k atoms/s | 231 ms | 38.2 GB |
| Nequip-OAM-XL | 193k atoms/s | 242 ms | 32.1 GB |
| PET-OAM-XL | 164k atoms/s | 285 ms | 37.2 GB |
| EquiformerV3-OAM | 121k atoms/s | 385 ms | 45.6 GB |

> These are sample/placeholder numbers. Run the benchmarks yourself to get real measurements.

## Quickstart

```bash
# Install dependencies
uv sync

# Run benchmarks on Modal A100
uv run modal run modal_app.py

# Quick test (small systems only)
uv run modal run modal_app.py --quick

# Deploy API endpoint
uv run modal deploy modal_app.py
```

## Frontend

A self-contained dashboard in `frontend/index.html` visualizes the results with interactive charts.

```bash
python3 -m http.server 8765
# open http://localhost:8765/frontend/
```

Features:
- Toggle between **single** and **batched** modes
- Toggle between **atoms/s** and **ms/step** metrics
- Ranking cards, throughput/latency line charts, memory bars, speedup comparison
- Load results from the Modal API, a local JSON file, or the bundled sample data

## Project Structure

```
├── benchmark/
│   ├── config.py            # Model specs, system sizes, timing params
│   └── run_benchmark.py     # TorchSim benchmark loop
├── frontend/
│   └── index.html           # Interactive results dashboard
├── results/
│   └── sample_results.json  # Placeholder data for frontend preview
├── modal_app.py             # Modal A100 deployment + JSON API
└── pyproject.toml
```

## License

MIT
