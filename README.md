# Effects of innovation length and forgetting factor on multi-innovation stochastic gradient identification for delayed state space systems

Code and curated numerical inputs for the manuscript by Dingqi Chen, Ruiqi Li, Ruiyang Zhang and Feiyan Chen.

This repository is the consolidated code release for the study. It contains the MISG estimator, delayed state-space system generator, simulation and validation programs, manuscript figure scripts, numerical stability checks, tests, and the compact data products needed to inspect the reported results.
## Repository structure

| Path | Contents |
|---|---|
| [`src/core/`](src/core) | MISG and comparison estimators, system generation, metrics, random-number streams, search, TOST, and weighting utilities |
| [`src/experiments/`](src/experiments) | Main simulation, innovation length/forgetting factor studies, confirmation experiments, robustness checks, computational-cost analysis, and the DaISy heating experiment |
| [`src/analysis/`](src/analysis) | Branch validation, sensitivity analysis, record length analysis, correction diagnostics, RLS follow up, and revision checks |
| [`src/figures/`](src/figures) | Programs that regenerate the retained manuscript figures and their source tables |
| [`src/theory/`](src/theory) | Numerical scripts supporting the stability and escape-certificate analysis |
| [`data/`](data) | Frozen system families, estimator golden traces, factorial results, and compact figure/source tables |
| [`tests/`](tests) | Unit, regression, reproducibility, performance, and repository integrity tests |
| [`tools/`](tools) | Environment snapshots, golden-trace checks, timing, source audits, and figure-alignment checks |

## Included data

| File | Purpose |
|---|---|
| [`data/systems.npz`](data/systems.npz) | Frozen pilot, training, validation, and test system families used by the simulation workflows |
| [`data/new_systems.npz`](data/new_systems.npz) | Independently sampled systems retained for holdout checks |
| [`data/balanced_full_factorial_results.json`](data/balanced_full_factorial_results.json) | Balanced 30-cell factorial result for five innovation lengths and six forgetting factors |
| [`data/estimators_golden.npz`](data/estimators_golden.npz) | Golden estimator trajectories used by regression tests |
| [`data/GOLDEN_PROVENANCE.md`](data/GOLDEN_PROVENANCE.md) | Software version and configuration provenance for the golden trajectories |
| [`data/figure2_balanced_full_factorial_summary.csv`](data/figure2_balanced_full_factorial_summary.csv), [`data/figure3_local_refinement_source.csv`](data/figure3_local_refinement_source.csv), [`data/figure4_source_data.csv`](data/figure4_source_data.csv), and [`data/figure5_source_data.csv`](data/figure5_source_data.csv) | Compact source data for the main numerical figures |
| [`data/figure5_source_provenance.json`](data/figure5_source_provenance.json) | Source hash and protocol metadata for the Figure 5 table |
| [`data/system_timing_summary.csv`](data/system_timing_summary.csv) | Retained computational cost summary |
| [`data/heating_principal_contrasts.csv`](data/heating_principal_contrasts.csv) | Retained aggregate contrasts for the measured heating experiment |

The original DaISy heating record is not redistributed. Generated result directories, checkpoints, rendered figures, and sample-level heating predictions are also not included.

## Requirements

- Python 3.11
- NumPy, SciPy, pandas, Matplotlib, threadpoolctl, CMA-ES, and pytest
- CPU execution is sufficient. No GPU packages are required

The frozen estimator traces were generated with Python 3.11, NumPy 2.4.2, and SciPy 1.17.1. The complete reproduction environment is defined in [`environment.yml`](environment.yml), which also pins Matplotlib 3.10.6, pandas 2.3.3, threadpoolctl 3.5.0, CMA-ES 4.4.4, and pytest 8.4.2.

Create the version pinned Conda environment from the repository root:

```bash
conda env create --name misg-p-and-lambda-code -f environment.yml
conda activate misg-p-and-lambda-code
```

Alternatively, install the dependencies from [`requirements.txt`](requirements.txt) in a Python environment of your choice:

```bash
pip install -r requirements.txt
```

The versions in `requirements.txt` are suitable for general execution, but its NumPy and SciPy pins differ from the golden trace environment. Use the Conda environment above for version-sensitive and exact output checks.

The repository itself does not need to be installed as a Python package. Run the commands below from the repository root.

## Verification

Run the repository integrity checks:

```bash
python -m unittest tests/test_package.py -v
```

Run the complete test suite:

```bash
python -m pytest tests
```

The tests cover the frozen system family, retained inputs, estimator behavior, deterministic random number streams, numerical metrics, search and resume behavior, experiment helpers and the shallow repository layout. Exact golden output tests require the pinned reproduction environment.

Several long running experiment programs provide a fast internal self check that does not repeat the full simulation:

```bash
python src/experiments/run_p_lambda_interaction_AUTO.py --self-test
python src/experiments/run_full_train_gamma_audit.py --self-test
python src/experiments/run_figure3_local_refinement_201.py --self-test
python src/experiments/run_final_frozen_comparison_AUTO.py --self-test
python src/experiments/run_new_system_holdout_AUTO.py --self-test
```

## Analysis and experiment programs

| Study component | Principal programs |
|---|---|
| Core estimator and system generation | [`src/core/estimators.py`](src/core/estimators.py), [`src/core/systems.py`](src/core/systems.py) |
| Initial simulation and reproducibility checks | [`src/experiments/run_e0_checks.py`](src/experiments/run_e0_checks.py), [`src/experiments/run_e1.py`](src/experiments/run_e1.py), [`src/experiments/run_repro2017.py`](src/experiments/run_repro2017.py) |
| Innovation length and forgetting-factor profile | [`src/experiments/run_lambda_profile_p5_FINAL_v5.py`](src/experiments/run_lambda_profile_p5_FINAL_v5.py), [`src/experiments/run_lambda_closeout_steps1_2_FINAL.py`](src/experiments/run_lambda_closeout_steps1_2_FINAL.py), [`src/experiments/run_p_lambda_interaction_AUTO.py`](src/experiments/run_p_lambda_interaction_AUTO.py) |
| Statistical closeout and multiplier selection | [`src/experiments/run_p_lambda_statistical_closeout_AUTO.py`](src/experiments/run_p_lambda_statistical_closeout_AUTO.py), [`src/experiments/run_full_train_gamma_audit.py`](src/experiments/run_full_train_gamma_audit.py), [`src/experiments/run_gamma_audit_closeout.py`](src/experiments/run_gamma_audit_closeout.py) |
| Frozen-trajectory and new-system confirmation | [`src/experiments/run_final_frozen_comparison_AUTO.py`](src/experiments/run_final_frozen_comparison_AUTO.py), [`src/experiments/run_new_system_holdout_AUTO.py`](src/experiments/run_new_system_holdout_AUTO.py) |
| Local refinement and robustness | [`src/experiments/run_figure3_local_refinement_201.py`](src/experiments/run_figure3_local_refinement_201.py), [`src/experiments/run_shortlist_K_reevaluation.py`](src/experiments/run_shortlist_K_reevaluation.py), [`src/experiments/run_tolerance_sensitivity_2pct_3pct.py`](src/experiments/run_tolerance_sensitivity_2pct_3pct.py) |
| Computational cost and power | [`src/experiments/benchmark_p_cost.py`](src/experiments/benchmark_p_cost.py), [`src/experiments/run_power.py`](src/experiments/run_power.py), [`src/experiments/run_tost_power.py`](src/experiments/run_tost_power.py) |
| Branch, sensitivity, record-length, and RLS analyses | [`src/analysis/run_branch_validation.py`](src/analysis/run_branch_validation.py), [`src/analysis/run_comprehensive_sensitivity.py`](src/analysis/run_comprehensive_sensitivity.py), [`src/analysis/run_review_1_4.py`](src/analysis/run_review_1_4.py), [`src/analysis/run_review_followup.py`](src/analysis/run_review_followup.py) |
| Conditional-correction and revision checks | [`src/analysis/run_prop3_diagnostics.py`](src/analysis/run_prop3_diagnostics.py), [`src/analysis/run_revision_checks.py`](src/analysis/run_revision_checks.py) |
| Measured DaISy heating experiment | [`src/experiments/run_daisy_heating_misg.py`](src/experiments/run_daisy_heating_misg.py), [`src/analysis/check_heating_residuals.py`](src/analysis/check_heating_residuals.py) |
| Stability support | [`src/theory/`](src/theory) |

The full simulation chain is computationally expensive and produces intermediate files that are not part of this compact release. Therefore, some downstream programs require results generated by earlier stages. Inspect an individual program's command line options before a full run:

```bash
python src/experiments/run_p_lambda_interaction_AUTO.py --help
python src/experiments/run_daisy_heating_misg.py --help
```

Choose an explicit output directory when the program supports one. Generated result directories are ignored by Git and should not be committed.

## Figure regeneration

The following figure programs operate on the compact tables retained in [`data/`](data):

```bash
python src/figures/regenerate_figure2.py
python src/figures/regenerate_figure3.py
python src/figures/regenerate_figure4_compact.py
python src/figures/regenerate_figure5_heatmaps.py
python src/figures/make_figure7.py
```

Additional scripts in `src/figures/` reproduce review stage figures and require their corresponding generated analysis directories. Generated figure files and alignment audit reports are excluded from version control and can be regenerated using the scripts above.

## Measured heating experiment

Run a reduced smoke check with:

```bash
python src/experiments/run_daisy_heating_misg.py --quick
```

The public DaISy heating record is not redistributed. Pass a local record with `--data-file`, or allow the script to retrieve it when network access and source terms permit. To use an existing local copy:

```bash
python src/experiments/run_daisy_heating_misg.py \
  --data-file /path/to/heating_system.dat.gz \
  --output-dir /path/to/output
```

### Dataset source, citation, and reuse

The heating record is the **Heating system** dataset, code **99-001**, in the Thermic Systems section of the [DaISy Database for the Identification of Systems](https://homes.esat.kuleuven.be/~smc/daisy/daisydata.html), maintained by KU Leuven ESAT/STADIUS. DaISy makes the record publicly downloadable for system identification research and asks users to cite the dataset in resulting publications.

Suggested citation:

> De Moor, B. L. R. (ed.). *DaISy: Database for the Identification of Systems*. Department of Electrical Engineering, ESAT/STADIUS, KU Leuven, Belgium. [https://homes.esat.kuleuven.be/~smc/daisy/](https://homes.esat.kuleuven.be/~smc/daisy/). Used dataset: Heating system, Thermic Systems, code 99-001. Accessed 24 October 2026.

The DaISy page does not state a standard open data license for this record or expressly grant redistribution, modification or commercial use rights. For that reason:

- the original `heating_system.dat.gz` file is not included in this repository;
- users obtain the record from the official source or supply their own local copy;
- only derived aggregate results, such as [`data/heating_principal_contrasts.csv`](data/heating_principal_contrasts.csv), are retained here;
- the repository's MIT License does not apply to the original DaISy data.

Before redistributing the raw record, incorporating it into another public dataset, or using it commercially, verify the current terms with the [DaISy maintainers](https://homes.esat.kuleuven.be/~smc/daisy/). Public availability alone should not be treated as permission for unrestricted reuse.

## Reproducibility notes

- Fixed random number streams and seed registries are implemented in `src/core/rng.py`.
- The primary frozen system family contains 201 training, 102 validation, and 201 test systems; tests verify these counts.
- The retained factorial result covers `p = {1, 2, 5, 8, 12}` and `lambda = {1.0, 0.99, 0.95, 0.925, 0.90, 0.85}`.
- The `src/theory/` directory preserves the numerical certification programs. Some of these programs consume generated certificates or platform specific compiled artifacts that are not included in the compact repository.
- Large experiments can write many intermediate records. Preserve the supplied files in `data/` and direct new runs to separate output directories.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). If you use this code, please cite this repository and the accompanying manuscript when available.

## License

This source code is released under the [MIT License](LICENSE). The license does not grant rights to third party datasets. The DaISy heating record and any other externally sourced data remain subject to their original ownership and use terms.
