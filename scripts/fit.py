import platform
import argparse
import os

import torch
import visdom

import mrrt
import mrrt.sdf
from matplotlib import pyplot as plt

MACOS_USE_MPS = False

def resolve_device(requested: str):
    if requested == 'mps' and platform.system() == "Darwin" and MACOS_USE_MPS:
        print("Cuda version:", torch.version.cuda)
        return torch.device('mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cpu')
    if requested == 'cuda':
        print("Cuda version:", torch.version.cuda)
        print("Found {} CUDA device(s)".format(torch.cuda.device_count()))
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("No CUDA, using CPU")
    return torch.device('cpu')

def remove_old_files(base_path, remove_type):
    if remove_type in ['weights', 'all']:
        for part_id in [0, 1]:
            weight_path = base_path + "{}.obj.pth".format(part_id)
            try:
                os.remove(weight_path)
                print(f"Removed old weights: {weight_path}")
            except FileNotFoundError:
                pass
    if remove_type in ['meshes', 'all']:
        for part_id in [0, 1]:
            mesh_path = base_path + "{}_sdf.obj".format(part_id)
            try:
                os.remove(mesh_path)
                print(f"Removed old mesh: {mesh_path}")
            except FileNotFoundError:
                pass
    if remove_type in ['pickle', 'all']:
        for part_id in [0, 1]:
            pkl_path = base_path + "{}.obj.pkl".format(part_id)
            try:
                os.remove(pkl_path)
                print(f"Removed old pickle: {pkl_path}")
            except FileNotFoundError:
                pass
    if remove_type in ['plots', 'all']:
        for part_id in [0, 1]:
            plot_path = base_path + "loss_object_{}.png".format(part_id)
            try:
                os.remove(plot_path)
                print(f"Removed old plot: {plot_path}")
            except FileNotFoundError:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, help='Puzzle name key, e.g., 09301')
    parser.add_argument('--category', choices=['general', 'puzzle', 'screw'], default='general', help='Puzzle category')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu', help='Training device')
    parser.add_argument('--samples', type=int, default=1000000, help='Number of SDF samples')
    parser.add_argument('--epochs0', type=int, default=100, help='Epochs for part 0')
    parser.add_argument('--epochs1', type=int, default=2, help='Epochs for part 1')
    parser.add_argument('--visdom', action='store_true', help='Enable Visdom visualization')
    parser.add_argument('--remove-old', choices=['none', 'weights', 'meshes', 'pickle', 'plots', 'all'], default='none', help='Remove old files before training')
    args = parser.parse_args()
    if args.remove_old != 'none':
        base_path = "./resources/models/joint_assembly_rotation/{}/{}/".format(args.category, args.name)
        remove_old_files(base_path, args.remove_old)
    device = resolve_device(args.device)
    vis = visdom.Visdom(env='sdf') if args.visdom else None
    print("Number of samples:", args.samples)
    print("Epochs for part 0:", args.epochs0)
    print("Epochs for part 1:", args.epochs1)
    base = "./resources/models/joint_assembly_rotation/{}/{}/".format(args.category, args.name)

    mesh = mrrt.sdf.SDFMesh(base + "0.obj", device, vis)
    losses_object_0 = mesh.fit(num_samples=args.samples, num_epochs=args.epochs0)
    mesh.load()
    mesh.to_mesh(base + "0_sdf.obj", n=500)

    # Plot and save losses for object 0
    if losses_object_0:
        plt.figure(figsize=(8, 4))
        plt.plot(range(len(losses_object_0)), losses_object_0, label='Object 0 loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss curve - object 0 ({args.name})')
        plt.legend()
        plt.tight_layout()
        out_path_0 = base + "loss_object_0.png"
        plt.savefig(out_path_0)
        plt.close()
        print(f"Saved loss plot for object 0 to {out_path_0}")

    mesh = mrrt.sdf.SDFMesh(base + "1.obj", device, vis)
    losses_object_1 = mesh.fit(num_samples=args.samples, num_epochs=args.epochs1)
    mesh.load()
    mesh.to_mesh(base + "1_sdf.obj", n=500)

    # Plot and save losses for object 1
    if losses_object_1:
        plt.figure(figsize=(8, 4))
        plt.plot(range(len(losses_object_1)), losses_object_1, label='Object 1 loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss curve - object 1 ({args.name})')
        plt.legend()
        plt.tight_layout()
        out_path_1 = base + "loss_object_1.png"
        plt.savefig(out_path_1)
        plt.close()
        print(f"Saved loss plot for object 1 to {out_path_1}")