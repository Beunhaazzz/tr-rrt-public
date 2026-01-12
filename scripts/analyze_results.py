import os
import json
import glob
import math
import csv
from statistics import mean
from typing import Dict, List, Tuple

# Criteria configuration
# Minimize: time, memory; Maximize: accuracy
WEIGHTS = {
    "time": 0.5,
    "memory": 0.2,
    "accuracy": 0.3,
}

COST_CRITERIA = {"time", "memory"}
BENEFIT_CRITERIA = {"accuracy"}


def safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def load_result_files(results_dir: str) -> List[str]:
    pattern = os.path.join(results_dir, "*.json")
    files = sorted(glob.glob(pattern))
    return files


def parse_dataset(file_path: str) -> Dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics_per_dataset(data: Dict) -> Dict[str, Dict[str, float]]:
    """Compute time, memory, and accuracy per algorithm for a single dataset.
    Accuracy is computed against trimesh collisions as baseline.
    """
    if "trimesh" not in data:
        raise ValueError("Baseline 'trimesh' not found in dataset: keys=" + ",".join(data.keys()))

    baseline_collisions = data["trimesh"]["collisions"]
    n_trials = len(baseline_collisions)

    results = {}
    for algo, metrics in data.items():
        # Time & memory means
        t = safe_mean(metrics.get("times", []))
        m = safe_mean(metrics.get("memories", []))

        # Accuracy vs baseline
        collisions = metrics.get("collisions", [])
        if len(collisions) != n_trials:
            # pad or truncate to match baseline length
            collisions = collisions[:n_trials] + [collisions[-1]] * max(0, n_trials - len(collisions))
        correct = sum(1 for i in range(n_trials) if collisions[i] == baseline_collisions[i])
        acc = correct / n_trials if n_trials > 0 else float("nan")

        results[algo] = {"time": t, "memory": m, "accuracy": acc}
    return results


def aggregate_across_datasets(per_dataset: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics (mean across datasets) per algorithm."""
    agg: Dict[str, Dict[str, List[float]]] = {}
    for ds in per_dataset:
        for algo, metrics in ds.items():
            if algo not in agg:
                agg[algo] = {"time": [], "memory": [], "accuracy": []}
            agg[algo]["time"].append(metrics["time"]) 
            agg[algo]["memory"].append(metrics["memory"]) 
            agg[algo]["accuracy"].append(metrics["accuracy"]) 

    # Compute means
    means: Dict[str, Dict[str, float]] = {}
    for algo, lists in agg.items():
        means[algo] = {
            "time": safe_mean([v for v in lists["time"] if not math.isnan(v)]),
            "memory": safe_mean([v for v in lists["memory"] if not math.isnan(v)]),
            "accuracy": safe_mean([v for v in lists["accuracy"] if not math.isnan(v)]),
        }
    return means


def min_max_normalize(values: Dict[str, float], is_benefit: bool) -> Dict[str, float]:
    vals = list(values.values())
    vmin = min(vals)
    vmax = max(vals)
    if math.isclose(vmin, vmax):
        # All equal → neutral scores
        return {k: 1.0 for k in values.keys()}
    norm = {}
    for k, v in values.items():
        if is_benefit:
            norm[k] = (v - vmin) / (vmax - vmin)
        else:  # cost
            norm[k] = (vmax - v) / (vmax - vmin)
    return norm


def weighted_sum_ranking(matrix: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    # Normalize each criterion
    time_norm = min_max_normalize({a: m["time"] for a, m in matrix.items()}, is_benefit=False)
    mem_norm = min_max_normalize({a: m["memory"] for a, m in matrix.items()}, is_benefit=False)
    acc_norm = min_max_normalize({a: m["accuracy"] for a, m in matrix.items()}, is_benefit=True)

    scores = {}
    for algo in matrix.keys():
        scores[algo] = (
            WEIGHTS["time"] * time_norm[algo]
            + WEIGHTS["memory"] * mem_norm[algo]
            + WEIGHTS["accuracy"] * acc_norm[algo]
        )
    return scores


def topsis_ranking(matrix: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    # Vector normalization per criterion
    def vec_norm(values: Dict[str, float]) -> Dict[str, float]:
        denom = math.sqrt(sum(v * v for v in values.values()))
        if math.isclose(denom, 0.0):
            return {k: 1.0 for k in values.keys()}
        return {k: v / denom for k, v in values.items()}

    time_vals = {a: m["time"] for a, m in matrix.items()}
    mem_vals = {a: m["memory"] for a, m in matrix.items()}
    acc_vals = {a: m["accuracy"] for a, m in matrix.items()}

    time_n = vec_norm(time_vals)
    mem_n = vec_norm(mem_vals)
    acc_n = vec_norm(acc_vals)

    # Apply weights
    time_w = {a: time_n[a] * WEIGHTS["time"] for a in time_n}
    mem_w = {a: mem_n[a] * WEIGHTS["memory"] for a in mem_n}
    acc_w = {a: acc_n[a] * WEIGHTS["accuracy"] for a in acc_n}

    # Ideal best/worst per criterion
    # Costs: best=min, worst=max; Benefits: best=max, worst=min
    time_best = min(time_w.values()); time_worst = max(time_w.values())
    mem_best = min(mem_w.values()); mem_worst = max(mem_w.values())
    acc_best = max(acc_w.values()); acc_worst = min(acc_w.values())

    scores = {}
    for algo in matrix.keys():
        d_best = math.sqrt(
            (time_w[algo] - time_best) ** 2 +
            (mem_w[algo] - mem_best) ** 2 +
            (acc_w[algo] - acc_best) ** 2
        )
        d_worst = math.sqrt(
            (time_w[algo] - time_worst) ** 2 +
            (mem_w[algo] - mem_worst) ** 2 +
            (acc_w[algo] - acc_worst) ** 2
        )
        # Relative closeness
        closeness = d_worst / (d_best + d_worst) if (d_best + d_worst) > 0 else 0.0
        scores[algo] = closeness
    return scores


def write_csv(output_path: str, matrix: Dict[str, Dict[str, float]], weighted_scores: Dict[str, float], topsis_scores: Dict[str, float]):
    fieldnames = ["algorithm", "time_mean", "memory_mean", "accuracy_mean", "weighted_score", "topsis_score"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for algo, metrics in matrix.items():
            writer.writerow({
                "algorithm": algo,
                "time_mean": metrics["time"],
                "memory_mean": metrics["memory"],
                "accuracy_mean": metrics["accuracy"],
                "weighted_score": weighted_scores.get(algo, float("nan")),
                "topsis_score": topsis_scores.get(algo, float("nan")),
            })


def write_markdown(output_path: str, matrix: Dict[str, Dict[str, float]], weighted_scores: Dict[str, float], topsis_scores: Dict[str, float]):
    lines = []
    lines.append("# MCDM Summary\n")
    lines.append("Criteria: minimize time & memory, maximize accuracy.\n")
    lines.append(f"Weights: time={WEIGHTS['time']}, memory={WEIGHTS['memory']}, accuracy={WEIGHTS['accuracy']}\n\n")

    # Rankings
    w_rank = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
    t_rank = sorted(topsis_scores.items(), key=lambda x: x[1], reverse=True)

    lines.append("## Weighted Sum Ranking\n")
    for i, (algo, score) in enumerate(w_rank, 1):
        m = matrix[algo]
        lines.append(f"{i}. {algo} — score={score:.4f} (time={m['time']:.4f}, memory={m['memory']:.0f}, accuracy={m['accuracy']:.3f})\n")
    lines.append("\n## TOPSIS Ranking\n")
    for i, (algo, score) in enumerate(t_rank, 1):
        m = matrix[algo]
        lines.append(f"{i}. {algo} — closeness={score:.4f} (time={m['time']:.4f}, memory={m['memory']:.0f}, accuracy={m['accuracy']:.3f})\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    # Infer repo root from this script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    results_dir = os.path.join(repo_root, "test_results")

    files = load_result_files(results_dir)
    if not files:
        print(f"No JSON files found in {results_dir}")
        return

    per_dataset = []
    for fp in files:
        try:
            data = parse_dataset(fp)
            per_dataset.append(compute_metrics_per_dataset(data))
        except Exception as e:
            print(f"Failed to parse {fp}: {e}")

    matrix = aggregate_across_datasets(per_dataset)

    # Remove baseline from ranking if undesired; keep all for visibility
    weighted_scores = weighted_sum_ranking(matrix)
    topsis_scores = topsis_ranking(matrix)

    # Print concise summary
    print("\nAggregated means across datasets (time↓, memory↓, accuracy↑):")
    for algo, m in matrix.items():
        print(f"- {algo:10s} time={m['time']:.4f}s, memory={m['memory']:.0f} bytes, accuracy={m['accuracy']:.3f}")

    print("\nWeighted Sum ranking:")
    for i, (algo, score) in enumerate(sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True), 1):
        print(f"{i}. {algo} -> {score:.4f}")

    print("\nTOPSIS ranking:")
    for i, (algo, score) in enumerate(sorted(topsis_scores.items(), key=lambda x: x[1], reverse=True), 1):
        print(f"{i}. {algo} -> {score:.4f}")

    # Write outputs
    out_csv = os.path.join(results_dir, "mcdm_summary.csv")
    out_md = os.path.join(results_dir, "mcdm_summary.md")
    write_csv(out_csv, matrix, weighted_scores, topsis_scores)
    write_markdown(out_md, matrix, weighted_scores, topsis_scores)

    print(f"\nWrote {out_csv} and {out_md}")


if __name__ == "__main__":
    main()
