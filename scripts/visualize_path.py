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
if platform.system() == "Darwin" and MACOS_USE_MPS:
    # Test for M1 GPUs
    device = torch.device('mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cpu')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

###############
# Setup args
###############
parser = argparse.ArgumentParser()
parser.add_argument('--name', default="az")
parser.add_argument('--pathid', type=int, default=0, required=False)
args = parser.parse_args()


if __name__ == "__main__":
    puzzle_path = puzzle_name_dict[args.name]

    mesh_1_file = os.path.join(puzzle_path, '0.obj')
    mesh_2_file = os.path.join(puzzle_path, '1.obj')

    # Load puzzle properties
    with open(os.path.join(puzzle_path, 'properties.json'), 'r') as fp:
        puzzle_properties = json.load(fp)
    
    q1 = puzzle_properties['q1']
    q2_start = puzzle_properties['q2_start']
    q2_end = puzzle_properties['q2_end']
    
    m1 = SDFMesh(mesh_1_file, device, None); m1.load(); m1.generate_sampling(2000)
    m2 = SDFMesh(mesh_2_file, device, None); m2.load(); m2.generate_sampling(2000)

    m1_urdf = os.path.join(puzzle_path, '0.urdf')
    m2_urdf = os.path.join(puzzle_path, '1.urdf')
    bv = BulletVisualization(gui=True)
    bv.add_urdf(m1_urdf, "m1")
    bv.set_object_color("m1", [215/256.0, 215/256.0, 215/256.0, 1])
    bv.set_object_configuration("m1", xyzrpy_2_SE3(q1))

    bv.add_urdf(m2_urdf, "m2")
    bv.set_object_color("m2", [201/256.0, 176/256.0, 55/256.0, 1])
    bv.set_object_configuration("m2", xyzrpy_2_SE3(q2_start))


    # Setup result dirs
    results_dir = os.path.join(BASE_RESULTS_DIR, args.name)
    paths_dir = os.path.join(results_dir, 'paths')

    with open(os.path.join(paths_dir, f'{args.pathid}.pkl'), 'rb') as fp:
        pkl_path = os.path.join(paths_dir, f"{args.pathid}.pkl")
        print(f'Loading path from {pkl_path}')
        try:
            path = pickle.load(fp)
        except Exception as e:
            print(f'Failed to load pickle file: {pkl_path}\n  Error: {e}')
            path = None

    # Trivial smoothing of the path
    if path is None:
        print(f'No path found in {pkl_path}. Check {os.path.join(results_dir, "logs")} and run the planner to generate paths.')
    else:
        curr = 0
        while curr < len(path) - 2:
            q_a = path[curr]
            q_b = path[curr+2]
            dist = distance_between_configurations(q_a, q_b)
            if is_edge_valid(m1, q1, m2, q_a, q_b, int(dist * 20), -0.02, device):
                del path[curr+1]
            else:
                curr += 1
            print(f'{curr}/{len(path)}')
        print("DONE smoothing the path")

    # Draw the path lines
    # for i in range(len(path) - 1):
    #     bv.add_debug_line(
    #         xyzrpy_2_SE3(path[i]),
    #         xyzrpy_2_SE3(path[i+1]), 
    #         color=[1,0,0]
    #     )
    
    # Uncomment to skip path playback
    # while True:
    #     bv.step()

    if path is None:
        print('rrt returned an empty path')
    else:
        frame = 0
        length = len(path)
        while True:
            bv.set_object_configuration("m2", xyzrpy_2_SE3(path[frame]))
            bv.step()
            next_frame = (frame + 1) % length
            dist = distance_between_configurations(path[frame], path[next_frame])
            speedup = 0.2
            # time.sleep(dist * speedup * 0.5)
            time.sleep(0.3)
            
            frame = next_frame
            if frame % length ==0:
                time.sleep(0.2)

