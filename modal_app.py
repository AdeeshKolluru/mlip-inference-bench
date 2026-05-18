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
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        # PyTorch 2.7 + CUDA 12.8 (matching EquiformerV3 setup)
        "torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        # PyG extensions (prebuilt wheels for torch 2.7 + cu128)
        "pyg_lib", "torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv",
        find_links="https://data.pyg.org/whl/torch-2.7.0+cu128.html",
    )
    .pip_install(
        "torch_geometric",
        "fairchem-core @ git+https://github.com/atomicarchitects/equiformer_v3.git#subdirectory=packages/fairchem-core",
        "ase>=3.22",
        "numpy>=1.24",
        "scipy==1.10.1",
        "matplotlib>=3.7",
    )
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .add_local_dir("benchmark", "/root/benchmark")
)

eqv3_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "pyg_lib", "torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv",
        find_links="https://data.pyg.org/whl/torch-2.7.0+cu128.html",
    )
    .pip_install(
        "torch_geometric",
        "fairchem-core @ git+https://github.com/atomicarchitects/equiformer_v3.git#subdirectory=packages/fairchem-core",
        "ase>=3.22",
        "numpy>=1.24",
    )
    .pip_install("scipy==1.10.1")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .add_local_dir("benchmark", "/root/benchmark")
)

ase_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "pyg_lib", "torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv",
        find_links="https://data.pyg.org/whl/torch-2.7.0+cu128.html",
    )
    .pip_install(
        "torch_geometric",
        "fairchem-core",
        "orb-models>=0.6.2",
        "ase>=3.22",
        "numpy>=1.24",
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
    model_filter: str | None = None,
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
    if model_filter:
        models = [(m.torchsim_key, m.name) for m in MODELS if m.torchsim_key == model_filter]
    else:
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


@app.function(
    image=eqv3_image,
    gpu="A100",
    volumes={"/results": results_vol},
    secrets=[modal.Secret.from_name("hf-token")],
    timeout=7200,
    memory=32768,
)
def run_eqv3_benchmark() -> str:
    """Benchmark EquiformerV3+DeNS-OAM on A100."""
    import json
    import logging
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    sys.path.insert(0, "/root")
    from benchmark.bench_eqv3 import run_eqv3_benchmark

    results = run_eqv3_benchmark(output_path="/results/eqv3_results.json")
    results_vol.commit()
    return json.dumps(results)


@app.function(
    image=gpu_image,
    gpu="A100",
    volumes={"/results": results_vol},
    secrets=[modal.Secret.from_name("hf-token")],
    timeout=7200,
    memory=32768,
)
def run_profiles() -> str:
    """Profile inference bottlenecks for each model class."""
    import logging
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    sys.path.insert(0, "/root")
    from benchmark.profile_models import run_all_profiles

    results = run_all_profiles(output_path="/results/profile_results.json")
    results_vol.commit()

    import json
    return json.dumps(results)


@app.function(
    image=gpu_image,
    gpu="A100",
    volumes={"/results": results_vol},
    secrets=[modal.Secret.from_name("hf-token")],
    timeout=7200,
    memory=32768,
)
def run_cuda_graph_test() -> str:
    """Test CUDA graph acceleration for each model class."""
    import logging
    import shutil
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    # Copy our CUDA graph wrapper into the installed torch_sim package
    override_src = "/root/torch_sim_models_override/cuda_graph.py"
    override_dst = "/usr/local/lib/python3.12/site-packages/torch_sim/models/cuda_graph.py"
    shutil.copy2(override_src, override_dst)
    logging.getLogger(__name__).info(f"Installed cuda_graph.py into torch_sim")

    sys.path.insert(0, "/root")
    from benchmark.test_cuda_graph import run_cuda_graph_test

    results = run_cuda_graph_test(output_path="/results/cuda_graph_results.json")
    results_vol.commit()

    import json
    return json.dumps(results)


@app.function(
    image=ase_image,
    gpu="A100",
    volumes={"/results": results_vol},
    secrets=[modal.Secret.from_name("hf-token")],
    timeout=7200,
    memory=32768,
)
def run_orb_ase_benchmark() -> str:
    """Benchmark ORB-v3-Direct via ASE calculator on A100."""
    import json
    import logging
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    sys.path.insert(0, "/root")
    from benchmark.bench_orb_ase import run_orb_ase_benchmark

    results = run_orb_ase_benchmark(output_path="/results/orb_ase_results.json")
    results_vol.commit()
    return json.dumps(results)


@app.function(
    image=ase_image,
    gpu="A100",
    volumes={"/results": results_vol},
    secrets=[modal.Secret.from_name("hf-token")],
    timeout=7200,
    memory=32768,
)
def run_allscaip_ase_benchmark() -> str:
    """Benchmark AllScAIP via FAIRChemCalculator on A100."""
    import json
    import logging
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    sys.path.insert(0, "/root")
    from benchmark.bench_allscaip_ase import run_allscaip_ase_benchmark

    results = run_allscaip_ase_benchmark(output_path="/results/allscaip_ase_results.json")
    results_vol.commit()
    return json.dumps(results)


@app.local_entrypoint()
def main(
    n_steps: int = 100,
    batch_size: int = 16,
    quick: bool = False,
    profile: bool = False,
    cuda_graph: bool = False,
    model: str = "",
):
    """Run MLIP inference benchmarks on Modal A100."""
    import json

    if model == "eqv3":
        results_json = run_eqv3_benchmark.remote()
        results = json.loads(results_json)
        print(f"\nEquiformerV3+DeNS-OAM — {results.get('metadata', {}).get('gpu_name', 'unknown')}")
        for sz, info in results.get("sizes", {}).items():
            print(f"  {sz} atoms: {info['single']['time_per_step_ms']:.1f}ms/step, {info['single']['atoms_per_second']} atoms/s, {info['peak_memory_mb']:.0f}MB")
        return

    if model == "orb_ase":
        results_json = run_orb_ase_benchmark.remote()
        results = json.loads(results_json)
        print(f"\nORB-v3-Direct (ASE) — {results.get('metadata', {}).get('gpu_name', 'unknown')}")
        for sz, info in results.get("sizes", {}).items():
            print(f"  {sz} atoms: {info['single']['time_per_step_ms']:.1f}ms/step, {info['single']['atoms_per_second']} atoms/s, {info['peak_memory_mb']:.0f}MB")
        return

    if model == "allscaip_ase":
        results_json = run_allscaip_ase_benchmark.remote()
        results = json.loads(results_json)
        print(f"\nAllScAIP (ASE) — {results.get('metadata', {}).get('gpu_name', 'unknown')}")
        for sz, info in results.get("sizes", {}).items():
            print(f"  {sz} atoms: {info['single']['time_per_step_ms']:.1f}ms/step, {info['single']['atoms_per_second']} atoms/s, {info['peak_memory_mb']:.0f}MB")
        return

    if cuda_graph:
        results_json = run_cuda_graph_test.remote()
        results = json.loads(results_json)
        print(f"\nCUDA Graph Test — {results['gpu_name']}")
        print(f"{'='*60}")
        for m in results["models"]:
            if m.get("speedup"):
                print(f"  {m['model_name']:25s} {m['baseline_ms']:7.1f}ms -> {m['cuda_graph_ms']:7.1f}ms  ({m['speedup']:.2f}x)")
            else:
                print(f"  {m['model_name']:25s} {m['baseline_ms']:7.1f}ms -> FAILED ({m.get('error', 'unknown')})")
        return

    if profile:
        results_json = run_profiles.remote()
        results = json.loads(results_json)
        for p in results.get("profiles", []):
            if "error" in p:
                print(f"\n{p['model_name']}: FAILED — {p['error']}")
            else:
                print(f"\n{'='*60}")
                print(f"  {p['model_name']} ({p['n_atoms']} atoms, {p['n_steps_profiled']} steps)")
                print(f"  Total CUDA time: {p['total_cuda_time_ms']:.1f} ms")
                print(f"{'='*60}")
                for op in p["top_operations"]:
                    bar = "#" * int(op["pct"] / 2)
                    print(f"  {op['pct']:5.1f}% {bar:<25s} {op['name']} ({op['calls']} calls, {op['cuda_time_ms']:.1f} ms)")
        return

    sizes = [64, 216] if quick else None
    results_json = run_benchmarks.remote(
        system_sizes=sizes,
        n_steps=n_steps,
        batch_size=batch_size,
        model_filter=model or None,
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
