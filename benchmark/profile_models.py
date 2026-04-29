"""Profile inference bottlenecks for each MLIP model class using torch.profiler."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _build_fcc_system(n_atoms: int, element: str = "Cu", a: float = 3.615):
    from ase.build import bulk

    n_repeat = max(1, round((n_atoms / 4) ** (1 / 3)))
    return bulk(element, "fcc", a=a) * (n_repeat, n_repeat, n_repeat)


def _load_model(model_key: str, device: torch.device):
    """Load model — same as run_benchmark.py but only the models we want to profile."""
    if model_key == "nequip":
        import tempfile
        import urllib.request

        from nequip.scripts.compile import main as nequip_compile
        from torch_sim.models.nequip_framework import NequIPFrameworkModel

        zip_url = "https://zenodo.org/records/18775904/files/NequIP-OAM-S-0.1.nequip.zip?download=1"
        tmp = tempfile.mkdtemp()
        zip_path = f"{tmp}/nequip-oam-s.nequip.zip"
        compiled_path = f"{tmp}/nequip-oam-s.nequip.pth"
        logger.info("  Downloading NequIP-OAM-S...")
        urllib.request.urlretrieve(zip_url, zip_path)
        logger.info("  Compiling NequIP model...")
        device_str = "cuda" if device.type == "cuda" else "cpu"
        nequip_compile(args=[
            zip_path, compiled_path,
            "--mode", "torchscript",
            "--device", device_str,
            "--target", "batch",
        ])
        return NequIPFrameworkModel.from_compiled_model(
            compiled_path, device=device, chemical_species_to_atom_type_map=True,
        )

    elif model_key == "orb":
        from orb_models.forcefield import pretrained
        from torch_sim.models.orb import OrbModel

        orb_ff, atoms_adapter = pretrained.orb_v3_conservative_inf_omat(
            device=device, precision="float32-high",
        )
        return OrbModel(orb_ff, atoms_adapter, device=device)

    elif model_key == "uma":
        from torch_sim.models.fairchem import FairChemModel

        return FairChemModel(
            model="uma-s-1p1", task_name="omat",
            device=device, dtype=torch.float32,
        )

    elif model_key == "pet":
        from torch_sim.models.metatomic import MetatomicModel
        from upet import get_upet

        pet_model = get_upet(model="pet-mad")
        return MetatomicModel(model=pet_model, device=device)

    else:
        raise ValueError(f"Unknown model key: {model_key}")


def profile_model(model_key: str, model_name: str, device: torch.device, n_atoms: int = 216):
    """Profile a single model and return top operations by GPU time."""
    import torch_sim as ts

    logger.info(f"Profiling {model_name}...")
    model = _load_model(model_key, device)

    atoms = _build_fcc_system(n_atoms)
    state = ts.io.atoms_to_state(atoms, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(5):
        model(state)
    torch.cuda.synchronize()

    # Profile
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        for _ in range(20):
            model(state)
        torch.cuda.synchronize()

    # Extract top CUDA operations
    events = prof.key_averages()

    # Sort by total CUDA time (attribute name varies by PyTorch version)
    cuda_events = []
    for evt in events:
        # PyTorch 2.8+ uses device_time_total, older uses cuda_time_total
        cuda_time_us = getattr(evt, 'device_time_total', None) or getattr(evt, 'cuda_time_total', 0)
        if cuda_time_us > 0:
            mem = getattr(evt, 'cuda_memory_usage', 0) or getattr(evt, 'device_memory_usage', 0) or 0
            cuda_events.append({
                "name": evt.key,
                "cuda_time_ms": round(cuda_time_us / 1000, 3),
                "cpu_time_ms": round(evt.cpu_time_total / 1000, 3),
                "calls": evt.count,
                "cuda_mem_mb": round(mem / 1e6, 1),
            })

    cuda_events.sort(key=lambda x: x["cuda_time_ms"], reverse=True)

    # Total CUDA time
    total_cuda_ms = sum(e["cuda_time_ms"] for e in cuda_events)

    # Top 15 ops with percentage
    top_ops = []
    for e in cuda_events[:15]:
        pct = (e["cuda_time_ms"] / total_cuda_ms * 100) if total_cuda_ms > 0 else 0
        top_ops.append({
            "name": e["name"],
            "cuda_time_ms": e["cuda_time_ms"],
            "pct": round(pct, 1),
            "calls": e["calls"],
        })

    # Also get the profiler table as text
    try:
        table_str = prof.key_averages().table(sort_by="device_time_total", row_limit=20)
    except Exception:
        try:
            table_str = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
        except Exception:
            table_str = prof.key_averages().table(sort_by="cpu_time_total", row_limit=20)

    return {
        "model_name": model_name,
        "model_key": model_key,
        "n_atoms": len(atoms),
        "n_steps_profiled": 20,
        "total_cuda_time_ms": round(total_cuda_ms, 1),
        "top_operations": top_ops,
        "profiler_table": table_str,
    }


def run_all_profiles(output_path: str = "/results/profile_results.json"):
    """Profile all model classes."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Profiling on device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    models_to_profile = [
        ("nequip", "NequIP-OAM-S"),
        ("orb", "ORB-v3-Conservative"),
        ("pet", "PET-MAD-S"),
        ("uma", "UMA-S-1p1"),
    ]

    results = {
        "metadata": {
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch_version": torch.__version__,
        },
        "profiles": [],
    }

    for model_key, model_name in models_to_profile:
        try:
            profile = profile_model(model_key, model_name, device)
            results["profiles"].append(profile)

            # Print table for logs
            logger.info(f"\n{'='*60}")
            logger.info(f"  {model_name} — Top CUDA operations")
            logger.info(f"{'='*60}")
            logger.info(profile["profiler_table"])
        except Exception as e:
            logger.error(f"Failed to profile {model_name}: {e}", exc_info=True)
            results["profiles"].append({
                "model_name": model_name,
                "model_key": model_key,
                "error": str(e),
            })

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"\nProfile results saved to {output_path}")

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    run_all_profiles()
