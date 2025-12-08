"""Small helper for printing CUDA information.

This file provides functions that attempt to report useful CUDA device
information including device name, compute capability, SM count and an
estimated total CUDA core count (SMs * cores_per_SM). It prefers `torch`
for device properties (already used in this workspace) and falls back
to other methods when necessary.

Run with: `python3 Main.py`
"""

import subprocess
import json
import argparse
from typing import Tuple, Optional, List, Dict, Any

import torch


def cores_per_sm_from_cc(major: int, minor: int) -> int:
    """Return typical CUDA cores per SM for a given compute capability.

    This mapping is based on commonly used values from NVIDIA documentation
    and community-maintained tables. It is not exhaustive, but covers a
    broad set of architectures. If unknown, returns a reasonable default.
    """
    table = {
        (1, 0): 8, (1, 1): 8, (1, 2): 8, (1, 3): 8,
        (2, 0): 32, (2, 1): 48,
        (3, 0): 192, (3, 2): 192, (3, 5): 192, (3, 7): 192,
        (5, 0): 128, (5, 2): 128, (5, 3): 128,
        (6, 0): 64, (6, 1): 128, (6, 2): 128,
        (7, 0): 64, (7, 2): 64, (7, 5): 64,
        (8, 0): 64, (8, 6): 64, (8, 7): 64, (8, 9): 64,
        (9, 0): 128,
    }
    return table.get((major, minor), 64)


def get_device_props_torch(device_index: int = 0):
    """Get properties for `device_index` using PyTorch.

    Returns a dict with keys: name, total_memory (bytes), major, minor,
    multi_processor_count.
    """
    props = torch.cuda.get_device_properties(device_index)
    return {
        "name": props.name,
        "total_memory": int(props.total_memory),
        "major": int(props.major),
        "minor": int(props.minor),
        "multi_processor_count": int(props.multi_processor_count),
    }

def get_torch_version() -> str:
    """Return the installed torch version as a string."""
    return torch.__version__

def estimate_total_cuda_cores_from_torch(device_index: int = 0) -> Tuple[int, Optional[str]]:
    """Estimate total CUDA cores for a device using torch properties.

    Returns a tuple `(estimated_cores, message)` where `message` is None
    when estimation succeeded cleanly or contains a short note otherwise.
    """
    if not torch.cuda.is_available():
        return 0, "CUDA not available (torch reports no CUDA)."

    props = get_device_props_torch(device_index)
    major, minor = props["major"], props["minor"]
    sm_count = props["multi_processor_count"]
    cpsm = cores_per_sm_from_cc(major, minor)
    estimated = sm_count * cpsm
    msg = None
    return estimated, f"Computed as {sm_count} SMs * {cpsm} cores/SM (CC {major}.{minor})"


def PrintCudaInfo() -> None:
    """Print a friendly CUDA device summary.

    Uses torch for detailed device properties. If `torch` is present but
    CUDA is not available, prints that status. The goal is concise, human-
    readable output.
    """
    print("=== CUDA Device Summary ===")
    if not torch.cuda.is_available():
        print("CUDA is not available (torch reports no CUDA).")
        # try to show `nvidia-smi` output as fallback
        try:
            out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                           "--format=csv,noheader,nounits"], text=True)
            print("nvidia-smi output (name, driver, memory MB):")
            print(out.strip())
        except Exception:
            pass
        return

    info = get_cuda_info()
    device_count = info.get("device_count", 0)
    print(f"CUDA available. Device count: {device_count}")
    print(f"Torch CUDA version: {info.get('torch_cuda_version')}")
    for d in info.get("devices", []):
        print("------------------------------")
        print(f"Device {d['index']}: {d['name']}")
        print(f"  Compute Capability: {d['compute_capability']}")
        print(f"  Multiprocessors (SMs): {d['multi_processor_count']}")
        print(f"  Estimated CUDA cores: {d['estimated_cores']}")
        if d.get('estimate_note'):
            print(f"  Note: {d['estimate_note']}")
        print(f"  Total memory: {d['total_memory_bytes']} bytes ({d['total_memory_human']})")

    print(f"Torch version: {get_torch_version()}")


def format_bytes(n: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def get_cuda_info() -> Dict[str, Any]:
    """Return structured CUDA info using torch as the source of truth.

    The returned dict contains `device_count`, `torch_cuda_version`, and
    `devices` (a list of per-device dicts).
    """
    out: Dict[str, Any] = {}
    out['torch_cuda_version'] = torch.version.cuda
    if not torch.cuda.is_available():
        out['device_count'] = 0
        out['devices'] = []
        # try to include nvidia-smi info if available
        try:
            smi = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                           "--format=csv,noheader,nounits"], text=True)
            out['nvidia_smi'] = smi.strip()
        except Exception:
            out['nvidia_smi'] = None
        return out

    device_count = torch.cuda.device_count()
    out['device_count'] = device_count
    devices: List[Dict[str, Any]] = []
    for i in range(device_count):
        props = get_device_props_torch(i)
        est_cores, note = estimate_total_cuda_cores_from_torch(i)
        dev = {
            'index': i,
            'name': props['name'],
            'compute_capability': f"{props['major']}.{props['minor']}",
            'major': props['major'],
            'minor': props['minor'],
            'multi_processor_count': props['multi_processor_count'],
            'estimated_cores': est_cores,
            'estimate_note': note,
            'total_memory_bytes': props['total_memory'],
            'total_memory_human': format_bytes(props['total_memory']),
        }
        devices.append(dev)
    out['devices'] = devices
    return out


def _main_cli():
    parser = argparse.ArgumentParser(description="Print CUDA device information.")
    parser.add_argument('--json', action='store_true', help='Print machine-readable JSON output')
    args = parser.parse_args()
    if args.json:
        info = get_cuda_info()
        print(json.dumps(info, indent=2))
    else:
        PrintCudaInfo()


if __name__ == "__main__":
    _main_cli()