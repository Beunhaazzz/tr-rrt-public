import platform
import argparse

import torch
import visdom

import mrrt
import mrrt.sdf

MACOS_USE_MPS = False

def resolve_device(requested: str):
    if requested == 'mps' and platform.system() == "Darwin" and MACOS_USE_MPS:
        print("Cuda version:", torch.version.cuda)
        return torch.device('mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cpu')
    if requested == 'cuda':
        print("Cuda version:", torch.version.cuda)
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("No CUDA, using CPU")
    return torch.device('cpu')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, help='Puzzle name key, e.g., 09301')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu', help='Training device')
    parser.add_argument('--samples', type=int, default=500_000, help='Number of SDF samples')
    parser.add_argument('--epochs0', type=int, default=50, help='Epochs for part 0')
    parser.add_argument('--epochs1', type=int, default=10, help='Epochs for part 1')
    parser.add_argument('--visdom', action='store_true', help='Enable Visdom visualization')
    args = parser.parse_args()

    device = resolve_device(args.device)
    print("Using device:", device)
    vis = visdom.Visdom(env='sdf') if args.visdom else None

    base = "./resources/models/joint_assembly_rotation/general/{}/".format(args.name)

    mesh = mrrt.sdf.SDFMesh(base + "0.obj", device, vis)
    mesh.fit(num_samples=args.samples, num_epochs=args.epochs0)
    mesh.load()
    mesh.to_mesh(base + "0_sdf.obj", n=500)

    mesh = mrrt.sdf.SDFMesh(base + "1.obj", device, vis)
    mesh.fit(num_samples=args.samples, num_epochs=args.epochs1)
    mesh.load()
    mesh.to_mesh(base + "1_sdf.obj", n=500)
