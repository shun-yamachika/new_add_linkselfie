# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements **LinkSelFiE** (Link Selection and Fidelity Estimation), a quantum network benchmarking system for the INFOCOM'24 paper. It simulates quantum entanglement distribution across multiple paths and evaluates different link selection algorithms under various noise models.

## Prerequisites

- **NetSquid**: Quantum network simulator (https://netsquid.org/)
- scipy, matplotlib, numpy

## Running Experiments

Execute all experiments and generate figures:
```bash
python main.py
```

The `main.py` file controls which experiments run via flags at the top:
- `RUN_BUDGET`: Budget sweep experiments
- `RUN_GAP_RANDOM`: Gap sweep with randomized fidelities
- `RUN_GAP_FIX`: Gap sweep with fixed arithmetic sequences
- `RUN_PAIRS`: Number of destination pairs sweep

Results are cached in `outputs/` as pickle files with file locking for parallel execution safety.

## Architecture

### Core Components

**Network Simulation (`network.py`, `nb_protocol.py`)**
- `QuantumNetwork`: Builds quantum network topology with multiple paths between Alice and Bob
- `NBProtocolAlice` / `NBProtocolBob`: NetSquid protocols implementing quantum teleportation and network benchmarking
- Network benchmarking uses randomized Clifford operations across multiple "bounces" to estimate link fidelity
- The `benchmark_path()` method runs the benchmarking protocol and returns estimated parameter `p` where fidelity = `p + (1-p)/2`

**Schedulers (`schedulers/`)**
- Four main algorithms: `LNaive`, `Greedy`, `Groups`, `WNaive`
- `lonline_nb.py`: Implements the core LinkSelFiE algorithm with two phases:
  - `lonline_init()`: Exploration phase (s=1) - uniform sampling across all links
  - `lonline_continue()`: Exploitation phase (s≥2) - successive elimination with doubling rounds
- All schedulers use **1-origin indexing** for path IDs (paths are numbered 1..L, not 0..L-1)
- `run_scheduler()` in `schedulers/__init__.py` is the unified entry point

**Evaluation Pipeline (`evaluation.py`, `evaluationgap.py`, `evaluationpair.py`)**
- Shared sweep caching: `_run_or_load_shared_sweep()` generates or loads cached experimental data
- Cache files use MD5 signatures of experimental parameters for reproducibility
- File locking (`*.pickle.lock`) prevents race conditions during parallel execution
- Each "repeat" fixes a single topology and sweeps all budget/gap/pair values across it
- Metrics inject both `est_fid_by_path` (estimated) and `true_fid_by_path` (ground truth) into `per_pair_details`
- All fidelity dictionaries use **1-origin keys** (path_id: 1..L)

**Utilities (`utils/`)**
- `ids.py`: Index conversion helpers for 1-origin ↔ 0-origin translation
  - `to_idx0(path_id)`: Convert 1-origin ID to 0-origin index
  - `to_id1(idx)`: Convert 0-origin index to 1-origin ID
  - `normalize_to_1origin(dict, L)`: Normalize dictionary keys to 1-origin
  - `is_keys_1origin(keys, L)`: Validate keys are exactly {1,2,...,L}
- `fidelity.py`: Fidelity list generators (random, gap-based, fixed sequences)
  - Uses `np.random.default_rng(seed)` for reproducibility
- `netsquid_helpers.py`: NetSquid quantum processor and connection setup

**Metrics and Visualization (`metrics/`, `viz/`)**
- `metrics/widths.py`: Confidence interval calculations (Hoeffding bounds)
- `viz/plots.py`: Plotting utilities with 95% CI bands
- All plots use consistent styling (TeX Gyre Termes font, color cycler)

## Key Design Patterns

### 1-Origin Indexing Convention
- **Public API** (scheduler inputs/outputs, dictionaries): Paths are 1..L
- **Internal lists** (Python arrays, fidelity banks): Indexed 0..L-1
- Use `to_idx0()` / `to_id1()` from `utils/ids.py` for conversion
- `normalize_to_1origin()` auto-detects and converts 0-origin dictionaries to 1-origin

### Caching and Reproducibility
- Experimental results are cached with hash signatures based on all parameters
- `seed` parameter controls global randomness via `np.random.default_rng(seed)`
- Importance mode: `"fixed"` uses `importance_list` directly; `"uniform"` samples from `importance_uniform` range per repeat
- Cache invalidation: Change `version` field in `_sweep_signature()` when schema changes

### Scheduler Return Format
When `return_details=True`, schedulers return:
```python
(per_pair_results, total_cost, per_pair_details)
```
- `per_pair_results`: List of `(correctness, cost, best_fid)` tuples per destination pair
- `total_cost`: Total measurement budget consumed
- `per_pair_details`: List of dicts with keys:
  - `alloc_by_path`: Dict `{path_id: num_bounces_allocated}` (1-origin)
  - `est_fid_by_path`: Dict `{path_id: estimated_fidelity}` (1-origin)
  - `true_fid_by_path`: Dict `{path_id: true_fidelity}` (injected by evaluation, 1-origin)

### Network Benchmarking Protocol
- `bounces`: List of bounce counts (e.g., `[1,2,3,4]`) - each bounce = one teleportation round
- `sample_times`: Dict `{bounce: num_repetitions}` - how many times to repeat each bounce count
- Cost accounting: `cost = sum(bounce * sample_times[bounce] for bounce in bounces)`
- The protocol validates `p` is in valid range `[0, 1.5)` before accepting (retry loop in `network.py:94-111`)

## Common Modifications

**Adding a new scheduler:**
1. Create `schedulers/my_scheduler.py` implementing `my_scheduler_budget_scheduler(node_path_list, importance_list, bounces, C_total, network_generator, return_details=False)`
2. Import in `schedulers/__init__.py`
3. Add case to `run_scheduler()` function
4. Use 1-origin path IDs in all input/output dictionaries

**Adding a new noise model:**
1. Add to `NOISE_MODELS` list in `main.py`
2. NetSquid supports: `"Depolar"`, `"Dephase"`, `"AmplitudeDamping"`, `"BitFlip"`
3. Noise model is passed to `EntanglingConnectionOnDemand` in `utils/netsquid_helpers.py`

**Changing experiment parameters:**
Edit configuration sections in `main.py`:
- Budget sweep: `BUDGET_LIST`, `BUDGET_NODE_PATHS`, `BUDGET_IMPORTANCES`
- Gap sweep: `GAP_LIST_RANDOM`, `GAP_RANDOM_NODE_PATHS`, etc.
- Pairs sweep: `PAIRS_LIST`, `PATHS_PER_PAIR`, `C_PAIRS_TOTAL`
- Global: `BOUNCES`, `REPEAT`, `SEED`

## Output Files

All outputs are in `outputs/`:
- `shared_sweep_<NoiseModel>_<hash>.pickle`: Cached experimental data
- `pair_sweep_<NoiseModel>_<hash>.pickle`: Pairs experiment cache
- `plot_accuracy_vs_budget_<NoiseModel>.pdf`: Correctness vs budget plots
- `plot_value_vs_budget_<NoiseModel>.pdf`: Value (weighted fidelity) vs budget
- `plot_value_vs_used_<NoiseModel>.pdf`: Value vs actual consumed budget
- PDFs are auto-cropped if `pdfcrop` is available
- `.pickle.lock` files are temporary lock files created during parallel execution

**Note**: Output files are gitignored to avoid repository bloat. Cache files are reproducible via the `seed` parameter.

## Important Notes

- **Index conversion**: Always use `utils.ids` helpers when converting between path IDs and array indices
- **State management**: `lonline_init()` and `lonline_continue()` maintain state dicts with `s`, `candidate_set`, `estimated_fidelities`, `alloc_by_path`, etc.
- **Budget constraints**: Schedulers enforce "all-or-nothing" round constraints - a measurement round only executes if full budget for all candidate links is available
- **NetSquid timing**: Protocol uses `yield self.await_timer(100)` for qubit propagation delays
- **Random seed**: Set via `SEED` in `main.py`; propagates to NumPy, Python random, and NetSquid

## File Organization

**Active Files**:
- All `.py` files in root, `schedulers/`, `utils/`, `metrics/`, `viz/`
- `evaluation.org`: Research notes (not executed)
- `groups.org`: Group configuration notes

**Gitignored**:
- Backup files (`*~`, `*.bak`)
- Python cache (`__pycache__/`, `*.pyc`)
- Output files (`outputs/*.pickle`, `outputs/*.pdf`, `outputs/*.json`)
- IDE settings (`.vscode/`, `.idea/`)

**Utility Scripts**:
- `convert.py`: Convert pickle cache files to JSON format for inspection
  ```bash
  python convert.py outputs/shared_sweep_Depolar_<hash>.pickle
  ```
