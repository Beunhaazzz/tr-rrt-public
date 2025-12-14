#this file is identical to Test.py, but evenry test it will train the sdf meshes from scratch with increasing epochs to see how that affects the test results
#for each iteration, we call fit.py to retrain the sdf meshes with increased epochs
#when the sdf meshes are trained, we run the same tests as in Test.py
#afther that, we increase the epochs and repeat

import os
import subprocess
import argparse
from pathlib import Path

# Resolve project root (parent of this testenv folder)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def pipenv_python():
    """Return full path to Pipenv's Python interpreter, or None if unavailable."""
    try:
        # pipenv --py prints the interpreter path for the Pipenv environment
        py_path = subprocess.check_output(["pipenv", "--py"], cwd=PROJECT_ROOT, text=True).strip()
        return py_path if py_path else None
    except Exception:
        return None

def run_fit_script(puzzle_name, category, device, samples, epochs0, epochs1, remove_old):
    py_exec = pipenv_python()
    if py_exec:
        cmd = [
            py_exec, 'scripts/fit.py',
            '--name', puzzle_name,
            '--category', category,
            '--device', device,
            '--samples', str(samples),
            '--epochs0', str(epochs0),
            '--epochs1', str(epochs1)
        ]
    else:
        # Fallback: run via pipenv run (no interactive shell needed)
        cmd = [
            'pipenv', 'run', 'python', 'scripts/fit.py',
            '--name', puzzle_name,
            '--category', category,
            '--device', device,
            '--samples', str(samples),
            '--epochs0', str(epochs0),
            '--epochs1', str(epochs1)
        ]
    if remove_old:
        cmd.extend(['--remove-old', remove_old])
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

def run_test_script(puzzle_name, epochs0, epochs1):
    py_exec = pipenv_python()
    if py_exec:
        cmd = [
            py_exec, 'testenv/Test.py',
            '--name', puzzle_name,
            '--epochs0', str(epochs0),
            '--epochs1', str(epochs1),
            '--num_tests', '100'
        ]
    else:
        cmd = [
            'pipenv', 'run', 'python', 'testenv/Test.py',
            '--name', puzzle_name,
            '--epochs0', str(epochs0),
            '--epochs1', str(epochs1),
            '--num_tests', '100'
        ]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, help='Puzzle name key, e.g., 09301')
    parser.add_argument('--category', choices=['general', 'puzzle', 'screw'], default='general', help='Puzzle category')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu', help='Training device')
    parser.add_argument('--samples', type=int, default=1000000, help='Number of SDF samples')
    parser.add_argument('--start-epochs0', type=int, default=10, help='Starting epochs for part 0')
    parser.add_argument('--start-epochs1', type=int, default=2, help='Starting epochs for part 1')
    parser.add_argument('--epoch-step0', type=int, default=10, help='Epoch increment for part 0')
    parser.add_argument('--epoch-step1', type=int, default=2, help='Epoch increment for part 1')
    parser.add_argument('--max-epochs0', type=int, default=1000, help='Maximum epochs for part 0')
    parser.add_argument('--max-epochs1', type=int, default=1000, help='Maximum epochs for part 1')
    parser.add_argument('--remove-old', choices=['none', 'weights', 'meshes', 'pickle', 'plots', 'all'], default='none', help='Remove old files before training')
    args = parser.parse_args()

    epochs0 = args.start_epochs0
    epochs1 = args.start_epochs1

    while epochs0 <= args.max_epochs0: #we only check epochs0 because part 0 is the bigger number of epochs
        print(f"Training SDF meshes with epochs0={epochs0}, epochs1={epochs1}...")
        run_fit_script(args.name, args.category, args.device, args.samples, epochs0, epochs1, args.remove_old if (epochs0 == args.start_epochs0 and epochs1 == args.start_epochs1) else 'none')
        
        print(f"Running tests with trained meshes (epochs0={epochs0}, epochs1={epochs1})...")
        run_test_script(args.name, epochs0, epochs1)

        epochs0 += args.epoch_step0
        epochs1 += args.epoch_step1