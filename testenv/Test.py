import platform
import argparse
import torch
import os
import json
import numpy as np
import time
import trimesh
import psutil
import mrrt
import mrrt.sdf
from matplotlib import pyplot as plt
from spatialmath import SE3

MACOS_USE_MPS = False
TEST_RESULTS_DIR = './test_results'

def ResolveDevice(requested: str) -> torch.device:
    if requested == 'mps' and platform.system() == "Darwin" and MACOS_USE_MPS:
        print("MacOS detected, attempting to use MPS backend")
        device: torch.device = torch.device('mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cpu')
        print("Backend selected:", device)
        return device
    if requested == 'cuda':
        print("Cuda version:", torch.version.cuda)
        print("Found {} CUDA device(s)".format(torch.cuda.device_count()))
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("No CUDA, using CPU")
    return torch.device('cpu')

def generate_random_configurations(num_samples: int) -> list:
    """Generate random SE3 configurations for testing"""
    configs = []
    for _ in range(num_samples):
        # Random translation [-0.5, 0.5]
        x = np.random.uniform(-0.5, 0.5)
        y = np.random.uniform(-0.5, 0.5)
        z = np.random.uniform(-0.5, 0.5)
        # Random rotation
        rx = np.random.uniform(0, 2 * np.pi)
        ry = np.random.uniform(0, 2 * np.pi)
        rz = np.random.uniform(0, 2 * np.pi)
        configs.append([x, y, z, rx, ry, rz])
    return configs

def xyzrpy_to_SE3(config: list) -> SE3:
    """Convert [x, y, z, rx, ry, rz] to SE3"""
    return SE3.Rx(config[3]) * SE3.Ry(config[4]) * SE3.Rz(config[5]) * SE3.Tx(config[0]).Ty(config[1]).Tz(config[2])

def trimesh_distance_between_objects(mesh1, q1: list, mesh2, q2: list) -> float:
    """
    Calculate minimum distance between two meshes using trimesh (deterministic method).
    """
    se3_q1 = xyzrpy_to_SE3(q1)
    se3_q2 = xyzrpy_to_SE3(q2)
    
    # Transform mesh vertices using SE3 matrix
    mesh1_transformed = mesh1.copy()
    mesh2_transformed = mesh2.copy()
    
    # Get the 4x4 transformation matrix and apply it
    T1 = se3_q1.data  # 4x4 matrix
    T2 = se3_q2.data  # 4x4 matrix
    
    # Apply transformation: add homogeneous coordinate, transform, then remove it
    vertices1_h = np.hstack([mesh1_transformed.vertices, np.ones((mesh1_transformed.vertices.shape[0], 1))])
    vertices2_h = np.hstack([mesh2_transformed.vertices, np.ones((mesh2_transformed.vertices.shape[0], 1))])
    
    mesh1_transformed.vertices = (T1 @ vertices1_h.T).T[:, :3]
    mesh2_transformed.vertices = (T2 @ vertices2_h.T).T[:, :3]
    
    try:
        min_dist = trimesh.proximity.distance.distance(mesh1_transformed, mesh2_transformed)
        return min_dist
    except:
        # Fallback: use bounding box collision
        return float(np.linalg.norm(mesh1_transformed.centroid - mesh2_transformed.centroid))

def neural_sdf_distance(sdf_mesh1, q1: list, sdf_mesh2, q2: list, device) -> float:
    """
    Calculate minimum distance using neural network SDF.
    """
    try:
        se3_q1 = xyzrpy_to_SE3(q1)
        se3_q2 = xyzrpy_to_SE3(q2)
        
        dist = mrrt.sdf.signed_distance(sdf_mesh1, se3_q1, sdf_mesh2, se3_q2, device)
        return dist
    except Exception as e:
        print(f"Error in neural SDF distance calculation: {e}")
        return float('inf')

def test_collision_detection(sdf_mesh1, sdf_mesh2, trimesh1, trimesh2, num_tests: int, device) -> dict:
    """
    Test collision detection with random configurations.
    Returns comparison metrics including memory usage.
    """
    print(f"\n{'='*60}")
    print(f"Running collision detection tests ({num_tests} configurations)")
    print(f"{'='*60}")
    
    # Get process for memory monitoring
    process = psutil.Process(os.getpid())
    
    # Generate random test configurations
    configs1 = generate_random_configurations(num_tests)
    configs2 = generate_random_configurations(num_tests)
    
    neural_distances = []
    trimesh_distances = []
    neural_times = []
    trimesh_times = []
    neural_memory = []
    trimesh_memory = []
    distance_errors = []
    
    for i, (config1, config2) in enumerate(zip(configs1, configs2)):
        if (i + 1) % max(1, num_tests // 10) == 0:
            print(f"Progress: {i + 1}/{num_tests}")
        
        # Test neural SDF distance with memory tracking
        start_time = time.time()
        mem_before = process.memory_info().rss / 1024 / 1024  # Convert to MB
        try:
            neural_dist = neural_sdf_distance(sdf_mesh1, config1, sdf_mesh2, config2, device)
            neural_time = time.time() - start_time
            mem_after = process.memory_info().rss / 1024 / 1024
            neural_distances.append(neural_dist)
            neural_times.append(neural_time)
            neural_memory.append(mem_after - mem_before)
        except Exception as e:
            print(f"  Error in neural SDF test {i}: {e}")
            neural_distances.append(float('inf'))
            neural_times.append(0)
            neural_memory.append(0)
        
        # Test trimesh distance (deterministic ground truth) with memory tracking
        start_time = time.time()
        mem_before = process.memory_info().rss / 1024 / 1024
        try:
            trimesh_dist = trimesh_distance_between_objects(trimesh1, config1, trimesh2, config2)
            trimesh_time = time.time() - start_time
            mem_after = process.memory_info().rss / 1024 / 1024
            trimesh_distances.append(trimesh_dist)
            trimesh_times.append(trimesh_time)
            trimesh_memory.append(mem_after - mem_before)
        except Exception as e:
            print(f"  Error in trimesh test {i}: {e}")
            trimesh_distances.append(float('inf'))
            trimesh_times.append(0)
            trimesh_memory.append(0)
        
        # Calculate error
        if neural_distances[-1] != float('inf') and trimesh_distances[-1] != float('inf'):
            error = abs(neural_distances[-1] - trimesh_distances[-1])
            distance_errors.append(error)
        else:
            distance_errors.append(float('inf'))
    
    results = {
        'neural_distances': neural_distances,
        'trimesh_distances': trimesh_distances,
        'neural_times': neural_times,
        'trimesh_times': trimesh_times,
        'neural_memory': neural_memory,
        'trimesh_memory': trimesh_memory,
        'distance_errors': distance_errors,
        'num_tests': num_tests,
    }
    
    return results

def plot_and_save_results(results: dict, puzzle_name: str) -> None:
    """
    Create comprehensive plots comparing neural SDF vs trimesh collision detection,
    including memory usage metrics.
    """
    neural_distances = np.array(results['neural_distances'])
    trimesh_distances = np.array(results['trimesh_distances'])
    neural_times = np.array(results['neural_times'])
    trimesh_times = np.array(results['trimesh_times'])
    neural_memory = np.array(results['neural_memory'])
    trimesh_memory = np.array(results['trimesh_memory'])
    distance_errors = np.array(results['distance_errors'])
    num_tests = results['num_tests']
    
    # Filter out infinite values for plotting
    valid_mask = (neural_distances != np.inf) & (trimesh_distances != np.inf)
    valid_errors = distance_errors[valid_mask]
    
    # Check if we have any valid data
    if not np.any(valid_mask):
        print("\n⚠ Warning: No valid distance comparisons. All trimesh tests failed.")
        print("This may indicate an issue with mesh transformation or compatibility.")
        return
    
    # Create a figure with multiple subplots (3x3 grid)
    fig = plt.figure(figsize=(18, 14))
    
    # 1. Distance Comparison Scatter Plot
    ax1 = plt.subplot(3, 3, 1)
    ax1.scatter(trimesh_distances[valid_mask], neural_distances[valid_mask], alpha=0.6, s=30)
    min_val = min(trimesh_distances[valid_mask].min(), neural_distances[valid_mask].min())
    max_val = max(trimesh_distances[valid_mask].max(), neural_distances[valid_mask].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect match', linewidth=2)
    ax1.set_xlabel('Trimesh Distance (Ground Truth)', fontsize=11)
    ax1.set_ylabel('Neural SDF Distance', fontsize=11)
    ax1.set_title('Distance Comparison: Neural SDF vs Trimesh', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Execution Time Comparison
    ax2 = plt.subplot(3, 3, 2)
    methods = ['Neural SDF', 'Trimesh']
    avg_times = [neural_times.mean(), trimesh_times.mean()]
    std_times = [neural_times.std(), trimesh_times.std()]
    bars = ax2.bar(methods, avg_times, yerr=std_times, capsize=10, color=['#1f77b4', '#ff7f0e'], alpha=0.7)
    ax2.set_ylabel('Time (seconds)', fontsize=11)
    ax2.set_title('Average Execution Time per Test', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, avg in zip(bars, avg_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{avg:.4f}s', ha='center', va='bottom', fontsize=10)
    
    # 3. Memory Usage Comparison
    ax3 = plt.subplot(3, 3, 3)
    avg_memory = [neural_memory.mean(), trimesh_memory.mean()]
    std_memory = [neural_memory.std(), trimesh_memory.std()]
    bars = ax3.bar(methods, avg_memory, yerr=std_memory, capsize=10, color=['#2ca02c', '#d62728'], alpha=0.7)
    ax3.set_ylabel('Memory Usage (MB)', fontsize=11)
    ax3.set_title('Average Memory Usage per Test', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, avg in zip(bars, avg_memory):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{avg:.2f}MB', ha='center', va='bottom', fontsize=10)
    
    # 4. Error Distribution Histogram
    ax4 = plt.subplot(3, 3, 4)
    valid_errors_filtered = valid_errors[valid_errors != np.inf]
    if len(valid_errors_filtered) > 0:
        ax4.hist(valid_errors_filtered, bins=30, color='#9467bd', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Absolute Error (units)', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.set_title('Distance Error Distribution', fontsize=12, fontweight='bold')
        ax4.axvline(valid_errors_filtered.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {valid_errors_filtered.mean():.6f}')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Time Comparison Over Tests
    ax5 = plt.subplot(3, 3, 5)
    test_indices = np.arange(num_tests)
    ax5.plot(test_indices, neural_times, label='Neural SDF', alpha=0.7, linewidth=1.5)
    ax5.plot(test_indices, trimesh_times, label='Trimesh', alpha=0.7, linewidth=1.5)
    ax5.set_xlabel('Test Number', fontsize=11)
    ax5.set_ylabel('Time (seconds)', fontsize=11)
    ax5.set_title('Execution Time per Test', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Memory Usage Over Tests
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(test_indices, neural_memory, label='Neural SDF', alpha=0.7, linewidth=1.5, marker='o', markersize=4)
    ax6.plot(test_indices, trimesh_memory, label='Trimesh', alpha=0.7, linewidth=1.5, marker='s', markersize=4)
    ax6.set_xlabel('Test Number', fontsize=11)
    ax6.set_ylabel('Memory Usage (MB)', fontsize=11)
    ax6.set_title('Memory Usage per Test', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Distance Values Over Tests
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(test_indices, trimesh_distances, label='Trimesh (Ground Truth)', alpha=0.7, linewidth=1.5)
    ax7.plot(test_indices, neural_distances, label='Neural SDF', alpha=0.7, linewidth=1.5)
    ax7.set_xlabel('Test Number', fontsize=11)
    ax7.set_ylabel('Distance (units)', fontsize=11)
    ax7.set_title('Distance Values Across Tests', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Performance Metrics Comparison
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')
    
    # Normalize metrics for visualization (0-1 scale where 1 is best)
    speedup = trimesh_times.mean() / neural_times.mean() if neural_times.mean() > 0 else 1
    memory_ratio = trimesh_memory.mean() / neural_memory.mean() if neural_memory.mean() > 0 else 1
    accuracy_neural = 1 - (valid_errors_filtered.mean() / max(trimesh_distances[valid_mask].max(), 0.001))
    accuracy_neural = max(0, min(1, accuracy_neural))  # Clamp to [0, 1]
    
    metrics = ['Execution\nSpeed', 'Memory\nEfficiency', 'Accuracy']
    neural_scores = [1 / speedup, 1, accuracy_neural]  # Lower speed is worse, so invert
    trimesh_scores = [1, memory_ratio, 1]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax8.bar(x - width/2, neural_scores, width, label='Neural SDF', color='#1f77b4', alpha=0.7)
    bars2 = ax8.bar(x + width/2, trimesh_scores, width, label='Trimesh', color='#ff7f0e', alpha=0.7)
    
    ax8.set_ylabel('Score (normalized)', fontsize=11)
    ax8.set_title('Relative Performance Metrics', fontsize=12, fontweight='bold')
    ax8.set_xticks(x)
    ax8.set_xticklabels(metrics)
    ax8.legend()
    ax8.set_ylim(0, max(max(neural_scores), max(trimesh_scores)) * 1.1)
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Statistics Summary (as text)
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    stats_text = f"""
COLLISION DETECTION COMPARISON - {puzzle_name}

Neural SDF:
  Distance: {neural_distances[valid_mask].mean():.6f}±{neural_distances[valid_mask].std():.6f}
  Time: {neural_times.mean():.6f}±{neural_times.std():.6f}s
  Memory: {neural_memory.mean():.2f}±{neural_memory.std():.2f}MB

Trimesh (Ground Truth):
  Distance: {trimesh_distances[valid_mask].mean():.6f}±{trimesh_distances[valid_mask].std():.6f}
  Time: {trimesh_times.mean():.6f}±{trimesh_times.std():.6f}s
  Memory: {trimesh_memory.mean():.2f}±{trimesh_memory.std():.2f}MB

Error Analysis:
  Mean Error: {valid_errors_filtered.mean():.6f}
  Std Dev: {valid_errors_filtered.std():.6f}
  Max Error: {valid_errors_filtered.max():.6f}

Performance:
  Time Speedup: {speedup:.2f}x
  Memory Ratio: {memory_ratio:.2f}x
  Tests: {num_tests}
    """
    
    ax9.text(0.05, 0.5, stats_text, fontsize=9, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(TEST_RESULTS_DIR, f'collision_detection_comparison_{puzzle_name}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to: {output_path}")
    plt.close()
    
    # Save detailed results as JSON
    json_path = os.path.join(TEST_RESULTS_DIR, f'collision_detection_results_{puzzle_name}.json')
    json_results = {
        'puzzle_name': puzzle_name,
        'num_tests': num_tests,
        'neural_sdf': {
            'mean_distance': float(neural_distances[valid_mask].mean()),
            'std_distance': float(neural_distances[valid_mask].std()),
            'mean_time': float(neural_times.mean()),
            'std_time': float(neural_times.std()),
            'mean_memory_mb': float(neural_memory.mean()),
            'std_memory_mb': float(neural_memory.std()),
        },
        'trimesh': {
            'mean_distance': float(trimesh_distances[valid_mask].mean()),
            'std_distance': float(trimesh_distances[valid_mask].std()),
            'mean_time': float(trimesh_times.mean()),
            'std_time': float(trimesh_times.std()),
            'mean_memory_mb': float(trimesh_memory.mean()),
            'std_memory_mb': float(trimesh_memory.std()),
        },
        'error_metrics': {
            'mean_error': float(valid_errors_filtered.mean()),
            'std_error': float(valid_errors_filtered.std()),
            'max_error': float(valid_errors_filtered.max()),
            'min_error': float(valid_errors_filtered.min()),
        },
        'performance_ratios': {
            'time_speedup': float(speedup),
            'memory_ratio': float(memory_ratio),
        }
    }
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"✓ Saved detailed results to: {json_path}")

def Main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, help='Puzzle name key, e.g., 09301')
    parser.add_argument('--category', choices=['general', 'puzzle', 'screw'], default='general', help='Puzzle category')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu', help='Device to use for the SDF neural network usage')
    parser.add_argument('--num-tests', type=int, default=50, help='Number of collision detection tests to run')
    args: argparse.Namespace = parser.parse_args()
    
    if not os.path.isdir(TEST_RESULTS_DIR):
        print("Creating test results directory at", TEST_RESULTS_DIR)
        os.mkdir(TEST_RESULTS_DIR)
    
    device: torch.device = ResolveDevice(args.device)
    print("Using device:", device)
    
    puzzlePath: str = "./resources/models/joint_assembly_rotation/{}/{}/".format(args.category, args.name)
    print("Puzzle path:", puzzlePath)
    meshFile0: str = os.path.join(puzzlePath, '0.obj')
    meshFile1: str = os.path.join(puzzlePath, '1.obj')
    
    # Load trimesh versions (for ground truth)
    print("Loading trimesh models...")
    if not os.path.isfile(meshFile0) or not os.path.isfile(meshFile1):
        print("Error: Mesh files not found.")
        return
    print("Mesh file 0:", meshFile0)
    print("Mesh file 1:", meshFile1)
    trimesh0 = trimesh.load(meshFile0)
    trimesh1 = trimesh.load(meshFile1)
    
    # Load neural SDF models
    if not os.path.isfile(os.path.join(puzzlePath, '0_sdf.obj')) or not os.path.isfile(os.path.join(puzzlePath, '1_sdf.obj')):
        print("Error: Neural SDF mesh files not found. Please run the fitting script first.")
        return
    print("Loading neural SDF models...")
    sdf_mesh0 = mrrt.sdf.SDFMesh(meshFile0, device)
    sdf_mesh0.load()
    sdf_mesh0.generate_sampling(10000)
    
    sdf_mesh1 = mrrt.sdf.SDFMesh(meshFile1, device)
    sdf_mesh1.load()
    sdf_mesh1.generate_sampling(10000)
    
    # Run collision detection tests
    print("Starting collision detection tests...")
    results = test_collision_detection(sdf_mesh0, sdf_mesh1, trimesh0, trimesh1, args.num_tests, device)
    
    # Plot and save results
    plot_and_save_results(results, args.name)
    
    print(f"\n{'='*60}")
    print("Testing complete!")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    Main()
