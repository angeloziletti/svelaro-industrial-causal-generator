# Industrial causal generator

A generator of synthetic industrial batch-process data with known causes, for testing causal discovery methods and
industrial AI on the question of which recorded variables directly drive a process target such as quality, yield or
energy use.

Each plant is drawn from a structural causal model built around one target. It comes with its full causal graph, the
role of every recorded variable with respect to the target (direct cause, ancestor, sibling or descendant) and a record
of how the data were generated. A plant is fixed by a configuration file and its graph number.

## Contents

- `advanced_scm/`, `time_series/`: the generator and the ground-truth audit
- `configs/`: the configurations and seeds behind the published data
- `checks/`: the check that every share is realised in the generated data as configured, and the paper's difficulty
  map
- `tests/`: unit tests of the confounder share, the estimator, the check configurations and the difficulty map
- `samples/`: 10 sample plants of the reference configuration

## Install

Python 3.12.

```bash
pip install -r requirements.txt
```

On Windows, keep the repository in a short path (e.g. `C:\src\`): plant folders are deep, and Windows limits paths to
260 characters unless long paths are enabled.

## Generate plants

Run from the repository root. Everything is set in the configuration file.

```bash
python -m advanced_scm.examples.generate_batch_datasets --config configs/reference.yaml
```

This writes the 10 plants of the configuration to the folder named by `output.directory`, in a subfolder
`experiment_<timestamp>`. To generate only some plants, add their graph numbers to the configuration:

```yaml
batch_generation:
  graphs: [0, 3]
```

A plant depends only on its configuration and graph number, so a selected plant is the same as in a full run.

## What a plant contains

The files below are in `batch_000/graph_<number>/realization_0/`.

| File | Content |
| --- | --- |
| `full_timeseries.parquet` | One row per time step. `context_index` is the batch, `temp_*` are the recorded variables and `defect_ts` is the target. |
| `datasets/dataset_metadata.json` | The ground truth, including the causal graph, the role of each variable (`target_driver_labels`) and the generation record. |
| `temporal_validation/` | Generation parameters and small diagnostic files. |

## Audit the ground truth

The audit replays each plant, intervenes on its variables and checks that the labels match the effects on the target.
Set `experiment_dir` in `configs/audit.yaml` to the folder written by the generator, then run:

```bash
python -m advanced_scm.examples.audit_ground_truth --config configs/audit.yaml
```

It writes `datasets/ground_truth_audit.json` for each plant and exits with an error if any plant fails.

## Check the shares

Measures each share on the generated plants and prints it beside its setting (paper, appendix). `configs/checks/`
holds the variants the check needs, such as the configurations without the within-batch confounder. Generate every
configuration listed in `configs/checks/realised_shares.yaml`, then run:

```bash
python -m checks.realised_shares --config configs/checks/realised_shares.yaml
```

It writes `output/checks/realised_shares.json`.

## Difficulty map

Computes the two difficulty measures of the paper (Appendix "Difficulty measured on the data") for every
configuration and, optionally, three public process datasets. Generate the configurations, then run:

```bash
python -m checks.difficulty_map --config configs/checks/difficulty_map.yaml
```

It writes `output/checks/difficulty_map.json` and reports the largest difference from the paper's points. To measure
the sample plants instead, list `configs/reference.yaml` under `configurations` and map it to `samples` under
`experiment_dirs`.

The public datasets are read from the files their authors publish; set their paths in the configuration:

| Dataset | Files | Problems |
| --- | --- | --- |
| Tennessee Eastman | the classic test runs `d00_te.dat` ... `d21_te.dat` (Downs and Vogel; Braatz et al.), and `TEGroundTruth.txt` from CIPCaD-Bench | each variable with one parent in the graph is a target, its parent the direct cause; XMEAS 1-22 and XMV 1-11 as X1-X33, one run per episode |
| IndPenSim | the 100-batch release (Mendeley, doi 10.17632/pdnjz7zz5x.1) | target: change of penicillin concentration within a batch; one problem per control regime; roles from the simulator's equations, listed in the configuration |
| Food processing | `UF.csv` from CIPCaD-Bench | target X17; the last stage, X13-X16, as direct causes |

## Tests

```bash
pip install pytest
python -m pytest tests
```

## Configurations

There are 119 configurations with 10 plants each. Every file is complete and can be run on its own. Each file is named
after its setting, e.g. `shares_extreme/confound_0.8.yaml` is the confounder share 0.8 and
`fewer_batches/combo_interaction_b25.yaml` the combined difficulties with 25 batches; the paper's table of all results
lists them in the same groups.

| Folder | Files | What varies |
| --- | --- | --- |
| `reference.yaml` | 1 | The reference plant |
| `shares/` | 11 | One difficulty at a time |
| `shares_extreme/` | 7 | The hard end of each difficulty, and combinations |
| `interactions/` | 7 | Product terms in the target, weak signal, weak control |
| `forty_variables/` | 9 | 40 recorded variables instead of 20 |
| `fewer_batches/` | 27 | 25, 50 or 100 batches instead of 200 |
| `imperfections/` | 38 | Sensor error, timing jitter, operator reaction and other departures from the clean plant |
| `edge_types/` | 2 | Causal edges of only one kind |
| `recipe_phases/` | 7 | Recipe phases shared by all variables |
| `lots_and_drift/` | 10 | Raw-material lots and campaign drift |

`configs/checks/` holds the configurations of the share check (above); they are not part of the 119.

## Sample plants

`samples/` holds plants 0 to 9 of `configs/reference.yaml`, in the same layout as the generator's output. Only
`full_timeseries.parquet` and `datasets/dataset_metadata.json` are kept for each plant. Absolute paths in the metadata
were removed. To audit the samples, set `experiment_dir: samples` in `configs/audit.yaml`.

## Licence

Released for non-commercial research use.

- **Code:** [PolyForm Noncommercial 1.0.0](LICENSE)
- **Data, including the sample plants:** [CC BY-NC 4.0](LICENSE-DATA)

For commercial use, contact the copyright holder.

Copyright (c) 2026 Svelaro GmbH
