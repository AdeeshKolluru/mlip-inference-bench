"""Standalone EquiformerV3 benchmark using OCPCalculator (no TorchSim dependency)."""

from __future__ import annotations

import json
import logging
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

CKPT_URL = "https://huggingface.co/mirror-physics/equiformer_v3/resolve/main/checkpoint/omat24-mptrj-salex_gradient.pt"


def _build_fcc_system(n_atoms: int, element: str = "Cu", a: float = 3.615):
    from ase.build import bulk
    n_repeat = max(1, round((n_atoms / 4) ** (1 / 3)))
    return bulk(element, "fcc", a=a) * (n_repeat, n_repeat, n_repeat)


def run_eqv3_benchmark(
    system_sizes: list[int] | None = None,
    n_steps: int = 100,
    n_warmup: int = 10,
    output_path: str = "/results/eqv3_results.json",
):
    """Benchmark EquiformerV3+DeNS-OAM on GPU via OCPCalculator."""
    from fairchem.core.common.relaxation.ase_utils import OCPCalculator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Download checkpoint
    tmp = tempfile.mkdtemp()
    ckpt_path = f"{tmp}/eqv3-oam.pt"
    logger.info("Downloading EquiformerV3+DeNS-OAM checkpoint...")
    urllib.request.urlretrieve(CKPT_URL, ckpt_path)
    logger.info("Checkpoint downloaded.")

    # Load model
    calc = OCPCalculator(checkpoint_path=ckpt_path, cpu=device.type == "cpu")

    sizes = system_sizes or [64, 216]
    results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "n_steps": n_steps,
            "n_warmup": n_warmup,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda or "N/A",
        },
        "model_name": "EquiformerV3+DeNS-OAM",
        "model_key": "eqv3",
        "matbench_cps": 0.902,
        "sizes": {},
    }

    for n_atoms_target in sizes:
        atoms = _build_fcc_system(n_atoms_target)
        actual_n_atoms = len(atoms)
        atoms.calc = calc

        logger.info(f"Benchmarking EquiformerV3 with {actual_n_atoms} atoms...")

        # Warmup — perturb positions to prevent ASE caching
        base_positions = atoms.get_positions().copy()
        for i in range(n_warmup):
            atoms.set_positions(base_positions + np.random.randn(*base_positions.shape) * 1e-4)
            atoms.get_potential_energy()
            atoms.get_forces()
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Timed single-system — perturb each step
        t0 = time.perf_counter()
        for i in range(n_steps):
            atoms.set_positions(base_positions + np.random.randn(*base_positions.shape) * 1e-4)
            atoms.get_potential_energy()
            atoms.get_forces()
        if device.type == "cuda":
            torch.cuda.synchronize()
        single_elapsed = time.perf_counter() - t0

        single_ms = (single_elapsed / n_steps) * 1000
        single_aps = (actual_n_atoms * n_steps) / single_elapsed

        peak_mem = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        results["sizes"][str(actual_n_atoms)] = {
            "actual_atoms": actual_n_atoms,
            "single": {
                "time_per_step_ms": round(single_ms, 3),
                "atoms_per_second": round(single_aps),
            },
            "peak_memory_mb": round(peak_mem, 1),
        }

        logger.info(
            f"  {actual_n_atoms} atoms: single={single_ms:.1f}ms/step, "
            f"peak_mem={peak_mem:.0f}MB"
        )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, indent=2))
    logger.info(f"Results saved to {output_path}")

    return results
