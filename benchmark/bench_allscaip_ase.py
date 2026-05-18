"""Standalone AllScAIP ASE benchmark using FAIRChemCalculator."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _build_fcc_system(n_atoms: int, element: str = "Cu", a: float = 3.615):
    from ase.build import bulk
    n_repeat = max(1, round((n_atoms / 4) ** (1 / 3)))
    return bulk(element, "fcc", a=a) * (n_repeat, n_repeat, n_repeat)


def run_allscaip_ase_benchmark(
    system_sizes: list[int] | None = None,
    n_steps: int = 100,
    n_warmup: int = 10,
    output_path: str = "/results/allscaip_ase_results.json",
):
    """Benchmark AllScAIP on GPU via FAIRChemCalculator."""
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    logger.info("Loading AllScAIP-MD-Conserving model...")
    predictor = pretrained_mlip.get_predict_unit(
        "allscaip-md-conserving-all-omol", device=str(device)
    )
    calc = FAIRChemCalculator(predictor, task_name="omol")
    logger.info("Model loaded.")

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
        "model_name": "AllScAIP-MD-Conserving",
        "model_key": "allscaip_ase",
        "sizes": {},
    }

    for n_atoms_target in sizes:
        atoms = _build_fcc_system(n_atoms_target)
        actual_n_atoms = len(atoms)
        atoms.calc = calc

        logger.info(f"Benchmarking AllScAIP with {actual_n_atoms} atoms...")

        base_positions = atoms.get_positions().copy()
        for i in range(n_warmup):
            atoms.set_positions(base_positions + np.random.randn(*base_positions.shape) * 1e-4)
            atoms.get_potential_energy()
            atoms.get_forces()
        if device.type == "cuda":
            torch.cuda.synchronize()

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
