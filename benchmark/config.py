"""Model configurations for MLIP inference benchmarks.

Top models from matbench-discovery benchmarked with TorchSim:
1. EquiformerV3-OAM  — matbench-discovery rank 1
2. PET-OAM-XL        — matbench-discovery rank 2 (metatomic wrapper)
3. eSEN-30M-OAM      — matbench-discovery rank 4 (fairchem wrapper)
4. Nequip-OAM-XL     — matbench-discovery rank 6 (nequip wrapper)
5. ORB-v3            — Orbital Materials (orb wrapper)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelSpec:
    name: str
    torchsim_key: str
    description: str
    matbench_rank: int | None


MODELS: list[ModelSpec] = [
    ModelSpec(
        name="EquiformerV3-OAM",
        torchsim_key="equiformer_v3",
        description="EquiformerV3 from Atomic Architects, top matbench-discovery model",
        matbench_rank=1,
    ),
    ModelSpec(
        name="PET-OAM-XL",
        torchsim_key="pet",
        description="Point Edge Transformer, trained on OMat + Alex + MPtrj",
        matbench_rank=2,
    ),
    ModelSpec(
        name="eSEN-30M-OAM",
        torchsim_key="fairchem",
        description="FAIR Chemistry eSEN 30M parameter model",
        matbench_rank=4,
    ),
    ModelSpec(
        name="Nequip-OAM-XL",
        torchsim_key="nequip",
        description="NequIP equivariant neural network potential",
        matbench_rank=6,
    ),
    ModelSpec(
        name="ORB-v3",
        torchsim_key="orb",
        description="Orbital Materials ORB v3 universal potential",
        matbench_rank=15,
    ),
]

# System sizes (number of atoms) to benchmark
SYSTEM_SIZES = [64, 216, 512, 1000, 2744]

# Number of MD steps for throughput measurement
N_STEPS = 100

# Number of warmup steps (excluded from timing)
N_WARMUP = 10

# Batch size for batched benchmarks
BATCH_SIZE = 16
