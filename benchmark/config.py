"""Model configurations for MLIP inference benchmarks.

Models from matbench-discovery and related MLIP ecosystems, benchmarked with TorchSim.
XL variants match the exact leaderboard entries where possible.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelSpec:
    name: str
    torchsim_key: str
    description: str
    matbench_cps: float | None  # CPS score from matbench-discovery leaderboard


MODELS: list[ModelSpec] = [
    # --- Leaderboard models (XL variants) ---
    ModelSpec(
        name="NequIP-OAM-XL",
        torchsim_key="nequip_xl",
        description="NequIP OAM-XL, matbench-discovery leaderboard entry",
        matbench_cps=0.886,
    ),
    ModelSpec(
        name="PET-OAM-XL",
        torchsim_key="pet_xl",
        description="PET OAM-XL via upet/metatomic, matbench-discovery leaderboard entry",
        matbench_cps=0.898,
    ),
    # --- Smaller / other variants ---
    ModelSpec(
        name="NequIP-OAM-S",
        torchsim_key="nequip",
        description="NequIP OAM-S, smaller variant",
        matbench_cps=None,
    ),
    ModelSpec(
        name="PET-MAD-S",
        torchsim_key="pet",
        description="PET-MAD small via upet/metatomic",
        matbench_cps=None,
    ),
    ModelSpec(
        name="ORB-v3-Conservative",
        torchsim_key="orb_conservative",
        description="Orbital Materials ORB v3 conservative (inf neighbors, OMAT)",
        matbench_cps=0.860,
    ),
    ModelSpec(
        name="ORB-v3-Direct",
        torchsim_key="orb_direct",
        description="Orbital Materials ORB v3 direct (inf neighbors, OMAT)",
        matbench_cps=0.860,
    ),
    ModelSpec(
        name="UMA-S-1p1",
        torchsim_key="fairchem_eqv2",
        description="Meta's Universal Model for Atoms, small variant v1.1",
        matbench_cps=None,
    ),
    ModelSpec(
        name="eSEN-SM-OC25",
        torchsim_key="fairchem_esen",
        description="FAIR Chemistry eSEN small conserving model (OC25)",
        matbench_cps=None,
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
