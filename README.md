# Quantum Network Link Selection (Modified LinkSelFiE)

This repository is a modified version of the LinkSelFiE codebase, adapted for research purposes. The original LinkSelFiE paper (*Link Selection and Fidelity Estimation in Quantum Networks*, INFOCOM'24) can be found at:

[![DOI](https://zenodo.org/badge/737247284.svg)](https://zenodo.org/doi/10.5281/zenodo.10444443)

## Prerequisites

To get started, ensure you have the following packages installed:

* [NetSquid](https://netsquid.org/) - Quantum network simulator
* scipy
* matplotlib
* numpy

## Repository Structure

* **schedulers/**: Implementation of various link selection algorithms
    * `lnaive_scheduler.py`: Naive uniform allocation (L-Naive)
    * `greedy_scheduler.py`: Two-phase greedy scheduler based on LinkSelFiE
    * `groups_scheduler.py`: Group-based allocation
    * `w_naive_scheduler.py`: Weighted naive scheduler
    * `lonline_nb.py`: Core LinkSelFiE algorithm implementation
* **evaluation.py**: Main evaluation script with caching for budget sweep experiments
* **evaluationgap.py**: Gap-based evaluation experiments
* **evaluationpair.py**: Number of destination pairs sweep experiments
* **nb_protocol.py**: Network benchmarking protocol implementation using NetSquid
* **network.py**: Quantum network topology and simulation
* **utils/**: Helper functions
    * `ids.py`: Index conversion utilities (1-origin ↔ 0-origin)
    * `fidelity.py`: Fidelity generation functions
    * `netsquid_helpers.py`: NetSquid quantum processor and connection helpers
* **metrics/**: Confidence interval and width calculations
* **viz/**: Plotting utilities
* **convert.py**: Utility to convert pickle outputs to JSON format

## How to Run

Edit `main.py` to configure which experiments to run, then execute:

```sh
python main.py
```

Configuration flags in `main.py`:
* `RUN_BUDGET`: Budget sweep experiments
* `RUN_GAP_RANDOM`: Gap sweep with randomized fidelities
* `RUN_GAP_FIX`: Gap sweep with fixed arithmetic sequences
* `RUN_PAIRS`: Number of destination pairs sweep

Results are saved in the `outputs/` directory as PDFs and pickle cache files.

## Key Modifications from Original LinkSelFiE

* Refactored scheduler architecture with unified interface
* Added support for multiple experimental configurations (budget, gap, pairs)
* Implemented file-based caching with locking for parallel execution
* Added 1-origin indexing convention throughout codebase
* Extended evaluation metrics and visualization

## Documentation

See [CLAUDE.md](./CLAUDE.md) for detailed architectural documentation and development guide.

## License

See [LICENSE](LICENSE)
