from mrrt.sdf import *
from mrrt.rrt import *
from mrrt.visualize import *

import os
import time
import json
import platform
import argparse

# The path for the JSON which converts puzzle_name to a directory 
PUZZLE_NAME_DICT_PATH = './resources/puzzle_dict.json'
with open(PUZZLE_NAME_DICT_PATH, 'r') as fp:
    puzzle_name_dict = json.load(fp)

BASE_RESULTS_DIR = './results'
if not os.path.isdir(BASE_RESULTS_DIR):
    os.mkdir(BASE_RESULTS_DIR)


###############
# Setup torch
###############
MACOS_USE_MPS = False
# device is set after parsing args to allow manual override
device = torch.device('cpu')

###############
# Setup args
###############
parser = argparse.ArgumentParser()
parser.add_argument('--name', default="09301")
parser.add_argument('--eta', type=float, default=0.002, required=False)
parser.add_argument('--threshold', type=float, default=-0.002, required=False)
parser.add_argument('--sampling', type=int, default=2500, required=False)
parser.add_argument('--num_test', type=int, default=1, required=False)
# parser.add_argument('--start_test', type=int, default=0, required=False)
parser.add_argument('--gui', type=bool, default=False, required=False)
parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu', required=False,
                    help='Force device: cpu (default), cuda, or mps (Apple).')
args = parser.parse_args()


if __name__ == "__main__":
    # Resolve device after args
    if args.device == 'mps' and platform.system() == "Darwin" and MACOS_USE_MPS:
        device = torch.device('mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cpu')
    elif args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    print("Cuda version:", torch.version.cuda)
    print("Using device:", device)
    print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

    puzzle_path = puzzle_name_dict[args.name]

    mesh_1_file = os.path.join(puzzle_path, '0.obj')
    mesh_2_file = os.path.join(puzzle_path, '1.obj')

    # Load puzzle properties
    with open(os.path.join(puzzle_path, 'properties.json'), 'r') as fp:
        puzzle_properties = json.load(fp)
    
    q1 = puzzle_properties['q1']
    q2_start = puzzle_properties['q2_start']
    q2_end = puzzle_properties['q2_end']

    print(q1, q2_start, q2_end)
    print("Loading meshes...")
    
    m1 = SDFMesh(mesh_1_file, device, None); m1.load(); m1.generate_sampling(args.sampling)
    m2 = SDFMesh(mesh_2_file, device, None); m2.load(); m2.generate_sampling(args.sampling)
    print(args.sampling)

    bv = None
    # if False:
    if True:
    # if args.gui and False:
        m1_urdf = os.path.join(puzzle_path, '0.urdf')
        m2_urdf = os.path.join(puzzle_path, '1.urdf')
        bv = BulletVisualization(gui=True)
        bv.add_urdf(m1_urdf, "m1")
        bv.set_object_color("m1", [0, 0, 1, 1])
        bv.set_object_configuration("m1", xyzrpy_2_SE3(q1))

        bv.add_urdf(m2_urdf, "m2")
        bv.set_object_color("m2", [1, 0, 0, 1])
        bv.set_object_configuration("m2", xyzrpy_2_SE3(q2_start))

    # Setup result dirs
    results_dir = os.path.join(BASE_RESULTS_DIR, args.name)
    paths_dir = os.path.join(results_dir, 'paths')
    trees_dir = os.path.join(results_dir, 'trees')
    logs_dir = os.path.join(results_dir, 'logs')

    for d in [results_dir, paths_dir, trees_dir, logs_dir]:
        if not os.path.isdir(d):
            os.mkdir(d)
    
    for i in range(args.num_test - 1, args.num_test):
        print("############################################")
        print(f"# Test - {i+1}, {args.name}")
        print("############################################")

        rrt = RRT(m1, q1, m2, q2_start, q2_end, device, bv)
        rrt.eps_contact = 0.25 * args.eta
        rrt.extend = lambda q_near, direction: extend_with_slide(
                rrt.m1, rrt.q1, rrt.m2, q_near, direction,
                rrt.extend_eta, rrt.extend_threshold, rrt.eps_contact, rrt.device, slide_duration=50, bv=rrt.bv, verbose=False)
        rrt.extend_eta = args.eta
        rrt.extend_threshold = args.threshold
        rrt.verbose = False
        rrt.tree_path = os.path.join(trees_dir, f'{i}.pkl')
        rrt.log_path = os.path.join(logs_dir, f'{i}.txt')

        start = time.time()
        # path = rrt.plan(q2_start, q2_end)
        path = rrt.plan(num_iterations=100000, timeout=1e6)
        end = time.time()

        # Save path only if planner returned a valid path. If planning failed
        # (path is None) create a failure marker file so users can see it quickly.
        pkl_out = os.path.join(paths_dir, f'{i}.pkl')
        if path is None:
            print(f'Planner returned None for test {i}; not saving path pickle ({pkl_out}).')
            fail_marker = os.path.join(results_dir, f'failed_{i}.txt')
            with open(fail_marker, 'w') as ff:
                ff.write('Planner returned None\n')
        else:
            with open(pkl_out, 'wb') as fp:
                pickle.dump(path, fp)
        
        time_took = end - start
        num_iterations = rrt.iteration + 1
        with open(os.path.join(results_dir, 'stats.txt'), 'a') as fp:
            fp.write(f'Test #{i+1}: {num_iterations} iterations, {time_took}[sec]\n')

