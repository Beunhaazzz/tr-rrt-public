import platform
import argparse
import torch
import os
import json
import numpy as np
import time
import tracemalloc
import trimesh
import mrrt
import mrrt.sdf
from mesh_to_sdf import mesh_to_sdf
from matplotlib import pyplot as plt
from spatialmath import SE3
from scipy.spatial import cKDTree
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

MACOS_USE_MPS = False
TEST_RESULTS_DIR = './test_results'
CREATE_PLOTS = True

#samplecount uit de file scripts/run_tests.py, in de args parameters is het 2500
neuralSdfSamplecount: int = 2500
kdtreeCache: dict = {}

def ResolveDevice(requested: str) -> torch.device:
    r"""
    Resolve the appropriate torch device based on the requested type and system capabilities.
    """
    if requested == 'mps' and platform.system() == "Darwin" and MACOS_USE_MPS:
        if not torch.backends.mps.is_available():
            print("MPS not available, falling back to CPU.")
            return torch.device('cpu')
        return torch.device('mps')
    if requested == 'cuda':
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU.")
            return torch.device('cpu')
        return torch.device('cuda')
    print("Using CPU device.")
    return torch.device('cpu')

def GenerateRandomConfigurations(numSamples: int) -> list:
    r"""
    Generate a list of random configurations within specified bounds.
    Each configuration consists of 6 values: [x, y, z, roll, pitch, yaw].
    """
    configurations = []
    for _ in range(numSamples):
        config = [
            np.random.uniform(-0.5, 0.5),  # x
            np.random.uniform(-0.5, 0.5),  # y
            np.random.uniform(-0.5, 0.5),  # z
            np.random.uniform(0, 2 * np.pi),  # roll
            np.random.uniform(0, 2 * np.pi),  # pitch
            np.random.uniform(0, 2 * np.pi)   # yaw
        ]
        configurations.append(config)
    return configurations

def MapXYZRPYToSE3(q: list) -> SE3:
    r"""
    Convert a configuration in XYZRPY format to an SE3 transformation.
    """
    return SE3.Rx(q[3]) * SE3.Ry(q[4]) * SE3.Rz(q[5]) * SE3.Tx(q[0]).Ty(q[1]).Tz(q[2])

def TrimeshCollisionCheck(meshA: trimesh.Trimesh, q1: list, meshB: trimesh.Trimesh, q2: list) -> bool:
    r"""
    Perform collision checking between two meshes using trimesh with given transforms.
    """
    se3Q1: SE3 = MapXYZRPYToSE3(q1)
    se3Q2: SE3 = MapXYZRPYToSE3(q2)
    t1: np.ndarray = se3Q1.A
    t2: np.ndarray = se3Q2.A
    try:
        from trimesh.collision import CollisionManager
        cm = CollisionManager()
        cm.add_object('meshA', meshA, transform=t1)
        cm.add_object('meshB', meshB, transform=t2)
        collision, names, data = cm.in_collision_internal(return_names=True, return_data=True)
        return collision
    except Exception as e:
        print("Error during collision check, falling back to AABB:", e)
        return AABBCollisionCheck(meshA, q1, meshB, q2)

def NeuralSDFCollisionCheck(sdfMeshA: mrrt.sdf.SDFMesh, q1: list, sdfMeshB: mrrt.sdf.SDFMesh, q2: list, device: torch.device) -> bool:
    r"""
    Perform collision checking between two SDF meshes using neural SDF method with given transforms for each mesh.
    """
    try:
        se3Q1: SE3 = MapXYZRPYToSE3(q1)
        se3Q2: SE3 = MapXYZRPYToSE3(q2)
        distance = mrrt.sdf.signed_distance(sdfMeshA, se3Q1, sdfMeshB, se3Q2, device)
        return bool(distance < 0.002)  # Collision threshold uit de paper Tight Motion Planning 
    except Exception as e:
        print("Error during SDF collision check:", e)
        return False

def AABBCollisionCheck(meshA: trimesh.Trimesh, q1: list, meshB: trimesh.Trimesh, q2: list) -> bool:
    r"""
    Perform collision checking between two meshes using their Axis-Aligned Bounding Boxes (AABB) with given transforms for each mesh.
    """
    se3Q1: SE3 = MapXYZRPYToSE3(q1)
    se3Q2: SE3 = MapXYZRPYToSE3(q2)
    meshATransformed: trimesh.Trimesh = meshA.copy()
    meshBTransformed: trimesh.Trimesh = meshB.copy()
    t1: np.ndarray = se3Q1.A
    t2: np.ndarray = se3Q2.A
    meshATransformed.apply_transform(t1)
    meshBTransformed.apply_transform(t2)
    try:
        # Use robust bounds overlap check provided by trimesh
        return bool(trimesh.bounds_overlap(meshATransformed.bounds, meshBTransformed.bounds))
    except Exception:
        # Manual overlap check as fallback
        a_min, a_max = meshATransformed.bounds
        b_min, b_max = meshBTransformed.bounds
        return bool(
            (a_min[0] <= b_max[0] and a_max[0] >= b_min[0]) and
            (a_min[1] <= b_max[1] and a_max[1] >= b_min[1]) and
            (a_min[2] <= b_max[2] and a_max[2] >= b_min[2])
        )

def PointCloudCollisionCheck(meshA: trimesh.Trimesh, q1: list, meshB: trimesh.Trimesh, q2: list) -> bool:
    r"""
    Perform collision checking between two meshes using point cloud distance with given transforms for each mesh.
    """
    def ApplyTransform(mesh: trimesh.Trimesh, q: list):
        T: SE3 = SE3.Rx(q[3]) * SE3.Ry(q[4]) * SE3.Rz(q[5]) * SE3.Tx(q[0]).Ty(q[1]).Tz(q[2])
        newMesh: trimesh.Trimesh = mesh.copy()
        newMesh.apply_transform(T.A)
        return newMesh
    meshATransformed: trimesh.Trimesh = ApplyTransform(meshA, q1)
    meshBTransformed: trimesh.Trimesh = ApplyTransform(meshB, q2)
    points: np.ndarray = meshATransformed.sample(5000)
    sdfValues: np.ndarray = mesh_to_sdf(meshBTransformed, points)
    # If any point has SDF <= threshold, treat as collision
    # Negative (inside) or zero exactly on surface is typically collision
    return bool(np.any(sdfValues <= 0.002))  # Collision threshold uit de paper Tight Motion Planning

def GetMeshKDTree(mesh: trimesh.Trimesh, num_points=10000):
    r"""
    Build or reuse a KDTree over surface samples of a mesh.
    """
    key: int = id(mesh)
    if key in kdtreeCache:
        return kdtreeCache[key]
    points = mesh.sample(num_points)
    tree = cKDTree(points)
    kdtreeCache[key] = tree
    return tree

def KDTreeCollisionCheck(meshA: trimesh.Trimesh, q1: list, meshB: trimesh.Trimesh, q2: list, threshold=0.002) -> bool:
    r"""
    Collision check using KDTree nearest surface distance.
    """
    se3Q1: SE3 = MapXYZRPYToSE3(q1)
    se3Q2: SE3 = MapXYZRPYToSE3(q2)
    meshAT: trimesh.Trimesh = meshA.copy()
    meshBT: trimesh.Trimesh = meshB.copy()
    meshAT.apply_transform(se3Q1.A)
    meshBT.apply_transform(se3Q2.A)
    pointsA = meshAT.sample(5000)
    treeB = GetMeshKDTree(meshBT)
    dists, _ = treeB.query(pointsA, k=1)
    # If any surface is closer than threshold → collision
    return bool(np.any(dists < threshold))

def RunCollisionTestIsolated(sdfMeshA: mrrt.sdf.SDFMesh, sdfMeshB: mrrt.sdf.SDFMesh, trimeshA: trimesh.Trimesh, trimeshB: trimesh.Trimesh, q: list, device: torch.device, function: str) -> dict:
    r"""
    Run a single collision detection test using the specified algorithm and collect performance metrics.
    """
    result: dict = {'time': 0.0, 'memory': 0, 'collision': False}
    startTime: float = time.time()
    if function == 'trimesh':
        col: bool = TrimeshCollisionCheck(trimeshA, q, trimeshB, q)
    elif function == 'neural_sdf':
        col: bool = NeuralSDFCollisionCheck(sdfMeshA, q, sdfMeshB, q, device)
    elif function == 'aabb':
        col: bool = AABBCollisionCheck(trimeshA, q, trimeshB, q)
    elif function == 'point_cloud':
        col: bool = PointCloudCollisionCheck(trimeshA, q, trimeshB, q)
    elif function == 'kdtree':
        col = KDTreeCollisionCheck(trimeshA, q, trimeshB, q)
    else:
        raise ValueError(f"Unknown collision detection function: {function}")
    endTime: float = time.time()
    result['time'] = endTime - startTime
    result['collision'] = col
    return result

def RunCollisionTest(sdfMeshA: mrrt.sdf.SDFMesh, sdfMeshB: mrrt.sdf.SDFMesh, trimeshA: trimesh.Trimesh, trimeshB: trimesh.Trimesh, numTests: int, device: torch.device) -> dict:
    r"""
    Run collision detection tests using different algorithms and collect performance metrics.
    """
    results: dict = {
        'trimesh': {'times': [], 'memories': [], 'collisions': []},
        'neural_sdf': {'times': [], 'memories': [], 'collisions': []},
        'aabb': {'times': [], 'memories': [], 'collisions': []},
        'point_cloud': {'times': [], 'memories': [], 'collisions': []},
        'kdtree': {'times': [], 'memories': [], 'collisions': []}
    }
    configurations: list = GenerateRandomConfigurations(numTests)
    for q in configurations:
        print("Testing configuration:", q)
        for algo in results.keys():
            tracemalloc.start()
            testResult: dict = RunCollisionTestIsolated(sdfMeshA, sdfMeshB, trimeshA, trimeshB, q, device, algo)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            results[algo]['times'].append(testResult['time'])
            results[algo]['memories'].append(peak)
            results[algo]['collisions'].append(testResult['collision'])
    return results

def CreatePlotsFromResults(results: dict, puzzleName: str) -> None:
    r"""
    Create and save plots from the collision test results.
    """
    algorithms: list = ['trimesh', 'neural_sdf', 'aabb', 'point_cloud', 'kdtree']
    for algo in algorithms:
        times: list = results[algo]['times']
        plt.figure()
        plt.plot(times, marker='o')
        plt.title(f'Collision Check Times - {algo} - Puzzle {puzzleName}')
        plt.xlabel('Test Index')
        plt.ylabel('Time (s)')
        plt.grid()
        plt.savefig(os.path.join(TEST_RESULTS_DIR, f'{puzzleName}_{algo}_times.png'))
        plt.close()

        memories: list = results[algo]['memories']
        plt.figure()
        plt.plot(memories, marker='o', color='orange')
        plt.title(f'Collision Check Memory Usage - {algo} - Puzzle {puzzleName}')
        plt.xlabel('Test Index')
        plt.ylabel('Memory (bytes)')
        plt.grid()
        plt.savefig(os.path.join(TEST_RESULTS_DIR, f'{puzzleName}_{algo}_memory.png'))
        plt.close()

        collisions: list = results[algo]['collisions']
        plt.figure()
        plt.plot(collisions, marker='o', color='green')
        plt.title(f'Collision Check Results - {algo} - Puzzle {puzzleName}')
        plt.xlabel('Test Index')
        plt.ylabel('Collision (1=True, 0=False)')
        plt.grid()
        plt.savefig(os.path.join(TEST_RESULTS_DIR, f'{puzzleName}_{algo}_collisions.png'))
        plt.close()

def Main() -> None:
    r"""
    Main function to execute the collision benchmark tests.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="A benchmark for collision checking algorithms against the neural SDF method.")
    parser.add_argument('--name', required=True, help='Puzzle name key, e.g., 09301')
    parser.add_argument('--category', choices=['general', 'puzzle', 'screw'], default='general', help='Puzzle category')
    parser.add_argument('--device', type=str, default='cuda', help="Device to use: 'cpu', 'cuda', or 'mps'")
    parser.add_argument('--num-tests', type=int, default=1, help='Number of collision detection tests to run')
    args: argparse.Namespace = parser.parse_args()

    device: torch.device = ResolveDevice(args.device)

    if not os.path.isdir(TEST_RESULTS_DIR):
        print("Creating test results directory at", TEST_RESULTS_DIR)
        os.mkdir(TEST_RESULTS_DIR)

    # Paths and files
    path: str = "./resources/models/joint_assembly_rotation/{}/{}/".format(args.category, args.name)
    meshFile0: str = os.path.join(path, '0.obj')
    meshFile1: str = os.path.join(path, '1.obj')
    if not os.path.isfile(meshFile0) or not os.path.isfile(meshFile1):
        print("Error: Mesh files not found at specified path:", path)
        return
    
    # Load meshes and SDFs
    trimesh0: trimesh.Trimesh = trimesh.load(meshFile0)
    trimesh1: trimesh.Trimesh = trimesh.load(meshFile1)
    sdf0: mrrt.sdf.SDFMesh = mrrt.sdf.SDFMesh(meshFile0, device)
    sdf1: mrrt.sdf.SDFMesh = mrrt.sdf.SDFMesh(meshFile1, device)
    sdf0.load()
    sdf1.load()
    sdf0.generate_sampling(neuralSdfSamplecount)
    sdf1.generate_sampling(neuralSdfSamplecount)
    print("Meshes and SDFs loaded successfully.")

    # Dit moet de grote piek voorkomen bij het meten van geheugen gebruik
    tracemalloc.start()
    tracemalloc.stop()

    # Run collision tests
    results: dict = RunCollisionTest(sdf0, sdf1, trimesh0, trimesh1, args.num_tests, device)

    # Save results
    resultFile: str = os.path.join(TEST_RESULTS_DIR, f'results_{args.name}.json')
    with open(resultFile, 'w') as f:
        json.dump(results, f, indent=2)
    print("Test results saved to", resultFile)

    # Create plots
    # if CREATE_PLOTS:
    #     CreatePlotsFromResults(results, args.name)

if __name__ == "__main__":
    Main()