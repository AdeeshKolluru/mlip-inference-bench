# MLIP Inference Bench

Throughput benchmarks for top machine-learning interatomic potentials (MLIPs) from [matbench-discovery](https://matbench-discovery.materialsproject.org/), run on NVIDIA A100 GPUs via [Modal](https://modal.com) and powered by [TorchSim](https://github.com/TorchSim/torch-sim).

**Interactive dashboard: [adeeshkolluru.com/benchmarks/mlip-inference](https://adeeshkolluru.com/benchmarks/mlip-inference/)**

## Results (NVIDIA A100-SXM4-40GB)

64-atom FCC Cu cell, batch size 16, 100 timed steps after 10 warmup, `torch.cuda.synchronize()` for accurate GPU timing. PyTorch 2.8.0, CUDA 12.4.

<table>
  <thead>
    <tr>
      <th align="left">Model</th>
      <th align="right">CPS</th>
      <th align="right">Single (ms)</th>
      <th align="right">Batched (ms)</th>
      <th align="right">Atoms/s</th>
      <th align="left">Throughput</th>
      <th align="right">Memory</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>🟢 <strong>NequIP-OAM-S</strong></td>
      <td align="right">—</td>
      <td align="right">29.3</td>
      <td align="right"><strong>29.9</strong></td>
      <td align="right"><strong>34,247</strong></td>
      <td><img src="https://img.shields.io/badge/-%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88-16a34a" alt="100%"></td>
      <td align="right">531 MB</td>
    </tr>
    <tr>
      <td>🔵 <strong>ORB-v3-Direct</strong></td>
      <td align="right">—</td>
      <td align="right">16.8</td>
      <td align="right">42.7</td>
      <td align="right">23,974</td>
      <td><img src="https://img.shields.io/badge/-%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88-0ea5e9" alt="70%"></td>
      <td align="right">1.2 GB</td>
    </tr>
    <tr>
      <td>🟣 <strong>ORB-v3-Conservative</strong></td>
      <td align="right">0.860</td>
      <td align="right">29.6</td>
      <td align="right">84.5</td>
      <td align="right">12,118</td>
      <td><img src="https://img.shields.io/badge/-%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88%E2%96%88-8b5cf6" alt="35%"></td>
      <td align="right">5.2 GB</td>
    </tr>
    <tr>
      <td>🔷 <strong>PET-MAD-S</strong></td>
      <td align="right">—</td>
      <td align="right">49.7</td>
      <td align="right">142.6</td>
      <td align="right">7,180</td>
      <td><img src="https://img.shields.io/badge/-%E2%96%88%E2%96%88%E2%96%88%E2%96%88-6366f1" alt="21%"></td>
      <td align="right">1.8 GB</td>
    </tr>
    <tr>
      <td>🟡 <strong>UMA-S-1p1</strong></td>
      <td align="right">—</td>
      <td align="right">135.0</td>
      <td align="right">607.5</td>
      <td align="right">1,686</td>
      <td><img src="https://img.shields.io/badge/-%E2%96%88-f59e0b" alt="5%"></td>
      <td align="right">6.7 GB</td>
    </tr>
    <tr>
      <td>🟩 <strong>PET-OAM-XL</strong></td>
      <td align="right">0.898</td>
      <td align="right">143.6</td>
      <td align="right">1,333</td>
      <td align="right">768</td>
      <td><img src="https://img.shields.io/badge/-%E2%96%8F-059669" alt="2%"></td>
      <td align="right">29.7 GB</td>
    </tr>
    <tr>
      <td>🔴 <strong>NequIP-OAM-XL</strong></td>
      <td align="right">0.886</td>
      <td align="right">181.0</td>
      <td align="right">1,912</td>
      <td align="right">536</td>
      <td><img src="https://img.shields.io/badge/-%E2%96%8F-ef4444" alt="2%"></td>
      <td align="right">34.5 GB</td>
    </tr>
  </tbody>
</table>

Sorted by batched throughput (highest first). CPS from [matbench-discovery](https://matbench-discovery.materialsproject.org/) where the exact checkpoint is on the leaderboard. ORB-v3 CPS is for conservative-inf-mpa.

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

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure Modal

Create a [Modal](https://modal.com) account, then authenticate:

```bash
uv run modal setup
```

This opens a browser to link your Modal account and writes credentials to `~/.modal.toml`.

### 3. Create a HuggingFace secret

Some models download checkpoints from HuggingFace. Create a Modal secret with your HF token:

```bash
modal secret create hf-token HF_TOKEN=hf_your_token_here
```

You can get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### 4. Run benchmarks

```bash
# Full benchmark (all models, all system sizes)
uv run modal run modal_app.py

# Quick test (small systems only, ~15 min)
uv run modal run modal_app.py --quick

# Run detached (survives terminal disconnect)
uv run modal run --detach modal_app.py

# Deploy as a persistent API endpoint
uv run modal deploy modal_app.py
```

Results are saved to a Modal volume (`mlip-bench-results`) and printed to stdout. The `--quick` flag benchmarks only 27-atom and 64-atom systems.

### GPU selection

The default GPU is `A100`. To change it, edit the `gpu="A100"` parameter in `modal_app.py`. Modal supports `T4`, `A10G`, `A100`, `H100`, and others.

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
