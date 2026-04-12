"""Modal app for running MLIP inference benchmarks on A100 GPUs.

Usage:
    # Run benchmarks
    uv run modal run modal_app.py

    # Run detached (survives disconnect)
    uv run modal run --detach modal_app.py

    # Quick test (small systems only)
    uv run modal run modal_app.py --quick
"""

from __future__ import annotations

import modal

app = modal.App("mlip-inference-bench")

results_vol = modal.Volume.from_name("mlip-bench-results", create_if_missing=True)

gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .pip_install(
        "torch>=2.2",
        "torch-sim-atomistic>=0.5",
        "ase>=3.22",
        "numpy>=1.24",
        # FairChem (eSEN + UMA/EquiformerV2)
        "fairchem-core>=2.2",
        # NequIP
        "nequip>=0.17",
        # ORB-v3
        "orb-models>=0.4",
        # PET via metatomic
        "metatomic",
        "metatomic-torchsim",
        "metatomic-ase",
        "upet",
        "matplotlib>=3.7",
    )
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .add_local_dir("benchmark", "/root/benchmark")
)

serve_image = modal.Image.debian_slim(python_version="3.12").pip_install("fastapi[standard]")


@app.function(
    image=gpu_image,
    gpu="A100",
    volumes={"/results": results_vol},
    secrets=[modal.Secret.from_name("hf-token")],
    timeout=7200,
    memory=32768,
)
def run_benchmarks(
    system_sizes: list[int] | None = None,
    n_steps: int = 100,
    batch_size: int = 16,
) -> str:
    """Run all MLIP benchmarks on A100 GPU. Returns JSON string."""
    import json
    import logging
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    sys.path.insert(0, "/root")
    from benchmark.config import BATCH_SIZE, MODELS, N_STEPS, N_WARMUP, SYSTEM_SIZES
    from benchmark.run_benchmark import run_all_benchmarks

    sizes = system_sizes or SYSTEM_SIZES
    models = [(m.torchsim_key, m.name) for m in MODELS]

    results = run_all_benchmarks(
        models=models,
        system_sizes=sizes,
        n_steps=n_steps or N_STEPS,
        n_warmup=N_WARMUP,
        batch_size=batch_size or BATCH_SIZE,
        output_path="/results/benchmark_results.json",
    )

    results_vol.commit()
    return json.dumps(results)


@app.function(
    image=serve_image,
    volumes={"/results": results_vol},
)
@modal.fastapi_endpoint(method="GET")
def get_results():
    """Serve benchmark results as JSON API."""
    import json
    from pathlib import Path

    results_file = Path("/results/benchmark_results.json")
    results_vol.reload()

    if not results_file.exists():
        return {"error": "No benchmark results yet. Run benchmarks first."}

    return json.loads(results_file.read_text())


@app.function(
    image=serve_image,
    volumes={"/results": results_vol},
)
@modal.fastapi_endpoint(method="GET")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.local_entrypoint()
def main(
    n_steps: int = 100,
    batch_size: int = 16,
    quick: bool = False,
):
    """Run MLIP inference benchmarks on Modal A100."""
    import json

    sizes = [64, 216] if quick else None
    results_json = run_benchmarks.remote(
        system_sizes=sizes,
        n_steps=n_steps,
        batch_size=batch_size,
    )
    results = json.loads(results_json)

    n_models = len(results.get("models", []))
    gpu = results.get("metadata", {}).get("gpu_name", "unknown")
    print(f"\nBenchmark complete: {n_models} models on {gpu}")

    for model in results.get("models", []):
        if "error" in model:
            print(f"  {model['model_name']}: FAILED - {model['error']}")
        else:
            sizes_data = model.get("sizes", {})
            if sizes_data:
                largest = max(sizes_data.keys(), key=lambda k: int(k))
                info = sizes_data[largest]
                print(
                    f"  {model['model_name']}: "
                    f"{info['single']['atoms_per_second']:,} atoms/s (single), "
                    f"{info['batched']['atoms_per_second']:,} atoms/s (batched x{info['batched']['batch_size']})"
                )
