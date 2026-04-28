"""Core benchmark logic using TorchSim for MLIP inference throughput."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _build_fcc_system(n_atoms: int, element: str = "Cu", a: float = 3.615):
    """Build an FCC copper supercell with approximately n_atoms atoms."""
    from ase.build import bulk

    n_repeat = max(1, round((n_atoms / 4) ** (1 / 3)))
    atoms = bulk(element, "fcc", a=a) * (n_repeat, n_repeat, n_repeat)
    return atoms


def _load_model(model_key: str, device: torch.device):
    """Load a TorchSim model wrapper by key."""
    if model_key == "fairchem_esen":
        from torch_sim.models.fairchem import FairChemModel

        # Use OC25-trained eSEN (not gated, unlike OMol25 models)
        return FairChemModel(
            model="esen-sm-conserving-all-oc25",
            device=device,
            dtype=torch.float32,
        )

    elif model_key == "fairchem_eqv2":
        from torch_sim.models.fairchem import FairChemModel

        # UMA is EquiformerV2-based, closest to EquiformerV3 in FairChem
        return FairChemModel(
            model="uma-s-1p1",
            task_name="omat",
            device=device,
            dtype=torch.float32,
        )

    elif model_key == "pet":
        from torch_sim.models.metatomic import MetatomicModel
        from upet import get_upet

        pet_model = get_upet(model="pet-mad")
        return MetatomicModel(model=pet_model, device=device)

    elif model_key == "nequip":
        import tempfile

        import urllib.request
        from nequip.scripts.compile import main as nequip_compile
        from torch_sim.models.nequip_framework import NequIPFrameworkModel

        # Download NequIP-OAM-S checkpoint
        zip_url = "https://zenodo.org/records/18775904/files/NequIP-OAM-S-0.1.nequip.zip?download=1"
        tmp = tempfile.mkdtemp()
        zip_path = f"{tmp}/nequip-oam-s.nequip.zip"
        compiled_path = f"{tmp}/nequip-oam-s.nequip.pt2"

        logger.info("  Downloading NequIP-OAM-S checkpoint...")
        urllib.request.urlretrieve(zip_url, zip_path)

        logger.info("  Compiling NequIP model for batch mode...")
        device_str = "cuda" if device.type == "cuda" else "cpu"
        compiled_path = f"{tmp}/nequip-oam-s.nequip.pth"
        nequip_compile(args=[
            zip_path, compiled_path,
            "--mode", "torchscript",
            "--device", device_str,
            "--target", "batch",
        ])

        return NequIPFrameworkModel.from_compiled_model(
            compiled_path,
            device=device,
            chemical_species_to_atom_type_map=True,
        )

    elif model_key == "orb_conservative":
        from orb_models.forcefield import pretrained
        from torch_sim.models.orb import OrbModel

        orb_ff, atoms_adapter = pretrained.orb_v3_conservative_inf_omat(
            device=device,
            precision="float32-high",
        )
        return OrbModel(orb_ff, atoms_adapter, device=device)

    elif model_key == "orb_direct":
        from orb_models.forcefield import pretrained
        from torch_sim.models.orb import OrbModel

        orb_ff, atoms_adapter = pretrained.orb_v3_direct_inf_omat(
            device=device,
            precision="float32-high",
        )
        return OrbModel(orb_ff, atoms_adapter, device=device)

    else:
        raise ValueError(f"Unknown model key: {model_key}")


def benchmark_single_model(
    model_key: str,
    model_name: str,
    system_sizes: list[int],
    n_steps: int = 100,
    n_warmup: int = 10,
    batch_size: int = 16,
    device: torch.device = torch.device("cuda"),
) -> dict:
    """Benchmark a single model across system sizes."""
    import torch_sim as ts

    logger.info(f"Loading model: {model_name} ({model_key})")
    model = _load_model(model_key, device)

    results = {
        "model_name": model_name,
        "model_key": model_key,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "sizes": {},
    }

    for n_atoms_target in system_sizes:
        logger.info(f"  Benchmarking {model_name} with ~{n_atoms_target} atoms...")
        atoms = _build_fcc_system(n_atoms_target)
        actual_n_atoms = len(atoms)

        # --- Single system benchmark ---
        state = ts.io.atoms_to_state(atoms, device=device, dtype=torch.float32)

        # Warmup
        for _ in range(n_warmup):
            model(state)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_steps):
            model(state)
        if device.type == "cuda":
            torch.cuda.synchronize()
        single_elapsed = time.perf_counter() - t0

        single_time_per_step_ms = (single_elapsed / n_steps) * 1000
        single_atoms_per_sec = (actual_n_atoms * n_steps) / single_elapsed

        # --- Batched benchmark ---
        batch_atoms = [atoms] * batch_size
        batch_state = ts.io.atoms_to_state(batch_atoms, device=device, dtype=torch.float32)

        # Warmup
        for _ in range(n_warmup):
            model(batch_state)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_steps):
            model(batch_state)
        if device.type == "cuda":
            torch.cuda.synchronize()
        batch_elapsed = time.perf_counter() - t0

        batch_time_per_step_ms = (batch_elapsed / n_steps) * 1000
        batch_atoms_per_sec = (actual_n_atoms * batch_size * n_steps) / batch_elapsed

        peak_memory_mb = (
            torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        results["sizes"][str(actual_n_atoms)] = {
            "target_atoms": n_atoms_target,
            "actual_atoms": actual_n_atoms,
            "single": {
                "time_per_step_ms": round(single_time_per_step_ms, 3),
                "atoms_per_second": round(single_atoms_per_sec),
            },
            "batched": {
                "batch_size": batch_size,
                "time_per_step_ms": round(batch_time_per_step_ms, 3),
                "atoms_per_second": round(batch_atoms_per_sec),
            },
            "peak_memory_mb": round(peak_memory_mb, 1),
        }

        logger.info(
            f"    {actual_n_atoms} atoms: "
            f"single={single_time_per_step_ms:.1f}ms/step, "
            f"batched={batch_time_per_step_ms:.1f}ms/step ({batch_size}x), "
            f"peak_mem={peak_memory_mb:.0f}MB"
        )

    return results


def run_all_benchmarks(
    models: list[tuple[str, str]],
    system_sizes: list[int],
    n_steps: int = 100,
    n_warmup: int = 10,
    batch_size: int = 16,
    output_path: str = "/results/benchmark_results.json",
) -> dict:
    """Run benchmarks for all models and save results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running benchmarks on device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    all_results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "n_steps": n_steps,
            "n_warmup": n_warmup,
            "batch_size": batch_size,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda or "N/A",
        },
        "models": [],
    }

    for model_key, model_name in models:
        try:
            result = benchmark_single_model(
                model_key=model_key,
                model_name=model_name,
                system_sizes=system_sizes,
                n_steps=n_steps,
                n_warmup=n_warmup,
                batch_size=batch_size,
                device=device,
            )
            all_results["models"].append(result)
        except Exception as e:
            logger.error(f"Failed to benchmark {model_name}: {e}", exc_info=True)
            all_results["models"].append({
                "model_name": model_name,
                "model_key": model_key,
                "error": str(e),
            })

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(all_results, indent=2))
    logger.info(f"Results saved to {output_path}")

    return all_results
