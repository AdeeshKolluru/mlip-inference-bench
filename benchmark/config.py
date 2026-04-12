"""Model configurations for MLIP inference benchmarks.

Top models from matbench-discovery benchmarked with TorchSim:
1. EquiformerV2/UMA  — FairChem UMA (EquiformerV2-based), closest to EquiformerV3
2. PET-MAD           — Point Edge Transformer via metatomic/upet
3. eSEN-30M-OAM      — FairChem eSEN 30M parameter model
4. Nequip-OAM-S      — NequIP equivariant neural network potential
5. ORB-v3            — Orbital Materials ORB v3 universal potential
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
        name="EquiformerV2-UMA",
        torchsim_key="fairchem_eqv2",
        description="FairChem UMA (EquiformerV2-based), top matbench-discovery architecture",
        matbench_rank=1,
    ),
    ModelSpec(
        name="PET-MAD",
        torchsim_key="pet",
        description="Point Edge Transformer via upet/metatomic",
        matbench_rank=2,
    ),
    ModelSpec(
        name="eSEN-SM-OC25",
        torchsim_key="fairchem_esen",
        description="FAIR Chemistry eSEN small conserving model (OC25)",
        matbench_rank=4,
    ),
    ModelSpec(
        name="Nequip-OAM-S",
        torchsim_key="nequip",
        description="NequIP equivariant neural network potential (compiled AOT)",
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
SYSTEM_SIZES = [64, 216, 1000]

# Number of MD steps for throughput measurement
N_STEPS = 50

# Number of warmup steps (excluded from timing)
N_WARMUP = 5

# Batch size for batched benchmarks
BATCH_SIZE = 8
