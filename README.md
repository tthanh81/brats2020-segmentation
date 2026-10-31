# BraTS2020 Brain Tumor Segmentation — 3D U-Net and Swin UNETR

**Research code and documented results** for multimodal 3D MRI brain tumor segmentation with PyTorch and MONAI. This repository began as a 3D U-Net baseline and was extended to seven evaluated configurations (M1, M2, M3, M4a, M4b, M5, M6). All comparisons below use a fixed **internal** validation split; M6 alone was selected for the final test of the current comparison workflow.

> **Release status (2026-11):** M1–M6 experiments and their internal validation comparisons are complete, and the **frozen M6** checkpoint was evaluated on all 56 internal test subjects. The newly added notebooks 06–11 are **review-only research summaries**, not fully executable end-to-end training notebooks. Some original training/analysis scripts are archived separately in Kaggle and the research preparation archive; the complete M1–M6 workflow has **not** been independently reproduced from this refactored GitHub branch alone.
>
> **Important test caveat:** The internal held-out cases come from the labeled BraTS2020 training collection; they are not the official BraTS challenge test set. Historical test results for earlier experiments are already present in this repository. Consequently, this test set must **not** be described as never previously accessed across the entire project.

## 1. Study design

| Item | Protocol |
|---|---|
| Data | BraTS2020 labeled training subjects (Kaggle: `awsaf49/brats20-dataset-training-validation`) |
| Split | Fixed seed 42: **257 train / 55 validation / 56 internal test** subjects |
| MRI inputs | T1, T1ce, T2, FLAIR |
| Labels | Background, NCR/NET, ED, ET; original label `4` remapped to `3` |
| Training patches | `96 × 96 × 96`, batch size 1 |
| Evaluation | Full-volume sliding-window inference, ROI `96 × 96 × 96`, overlap `0.25` |
| Region metrics | Dice: WT (whole tumor), TC (tumor core), ET (enhancing tumor) |
| Checkpoint selection | Internal validation only; M6 epoch 20 frozen before the test in the current workflow |

M2–M5 extended the 3D U-Net training lineage from M1's epoch-10 checkpoint to epoch 20; M6 was initialized **from scratch** and trained for 20 epochs. Epoch counts therefore do not imply comparable computational cost. Consult the experiment records and [research overview](docs/M1_M6_RESEARCH_OVERVIEW.md) for detailed constraints.

## 2. Internal validation results — all seven configurations

All metrics below are means on the **same 55 internal validation subjects**, using the reported epoch-20 checkpoints.

| Model | Main experimental change | WT | TC | ET | Mean Dice |
|---|---|---:|---:|---:|---:|
| M1 | 3D U-Net baseline (20 epochs) | 0.861699 | 0.743321 | 0.667613 | 0.757544 |
| M2 | 3D data augmentation | 0.827139 | 0.727427 | 0.660358 | 0.738308 |
| M3 | Cosine LR scheduler | 0.865705 | 0.753051 | 0.671339 | 0.763365 |
| M4a | Dice + weighted CE + focal loss | 0.862155 | 0.743206 | 0.682176 | 0.762512 |
| M4b | Class-balanced patch sampling | 0.857675 | 0.733379 | 0.661771 | 0.750942 |
| M5 | LR schedule + combined loss | 0.868797 | 0.752799 | 0.678467 | 0.766687 |
| **M6** | **Swin UNETR (15,705,646 parameters)** | **0.870206** | **0.760337** | **0.697429** | **0.775991** |

**M6** was selected based on the predefined validation Mean Dice criterion. The per-patient paired validation analysis includes Wilcoxon tests and Holm correction. Statistical significance and uncertainty should be interpreted from the original verified analysis artifacts, not inferred from this aggregate table alone.

- [Validation summary CSV](results/m1_m6_summary/validation_seven_model_means_summary.csv)
- [Summary data provenance](results/m1_m6_summary/README.md)

These repository-level CSV/JSON summaries were **transcribed from previously verified Kaggle outputs**; they are **not byte-identical exports** of the source result files.

## 3. Frozen M6 final test (56 internal test subjects)

One-time evaluation using the selected **Swin UNETR / epoch 20** checkpoint. No further model selection, training, or hyperparameter tuning was performed on the basis of the test results.

| Region | Dice | Patient-bootstrap 95% confidence interval |
|---|---:|---:|
| WT | 0.871992 | [0.849028, 0.892578] |
| TC | 0.808354 | [0.757233, 0.852442] |
| ET | 0.723875 | [0.648604, 0.790752] |
| **Mean Dice** | **0.801407** | **[0.762350, 0.836853]** |

- Evaluated subjects: **56/56**; **4** subjects had empty ET ground-truth masks.
- Frozen checkpoint SHA256: `02a2930122c3994fa72342921c719f1f001d540f4fb2e106fe293607ed527421`.
- Remote backup: `tthanhh811/brats2020-m6-finaltest-56cases-seed42` (Kaggle private dataset; access may be restricted). All four 14-subject chunks and all four final artifacts were checked with SHA256.
- [Public-safe final test summary](results/m1_m6_summary/m6_final_test_public_summary.json)

These confidence intervals quantify uncertainty in the subject-level mean metric under the study's bootstrap procedure; they are **not** predictions for the Dice of 95% of individual patients. Results from the validation and test cohorts are descriptive summaries of different subject samples, not paired comparisons.

## 4. Repository guide

```text
brats2020-segmentation/
├── notebooks/
│   ├── 01–05: baseline and original pipeline notebooks
│   └── 06–11: cleaned, review-only notebooks for M2–M6
├── src/
│   ├── data/        # MRI loading, augmentation, patch sampling
│   ├── models/      # 3D U-Net, M4a combined loss, Swin UNETR factory
│   ├── training/    # existing training helpers
│   └── evaluation/  # existing full-volume evaluation helpers
├── splits/          # fixed patient-level split manifest
├── results/
│   ├── metrics/ and extended_20epochs/  # historical baseline artifacts
│   └── m1_m6_summary/                    # new public aggregate summaries
├── tests/           # synthetic CPU smoke checks
├── docs/            # research overview and archived legacy documentation
└── requirements.txt
```

### What is reproducible here?

- The earlier baseline pipeline and existing modules are preserved on historical commits, including the pinned experiment source commit `79a95a08f11694278114a89348cf8303a1e39d67` (branch `experiment/unet3d-optimization`).
- The additional M4a, M4b and Swin UNETR modules passed **synthetic CPU smoke tests** with Python 3.13.15, PyTorch 2.11.0+cpu, and MONAI 1.6.1. These tests check selected behaviors (sampler geometry, loss forward/backward, Swin UNETR parameter count), **not** full experiment reproducibility.
- Review notebooks `06-...` through `11-...` are primarily explanatory documents. References inside them to `original_notebooks/` or `scripts/historical_cells/` point to a **separate research preparation archive**, not directories shipped in this public repository.
- Full M1–M6 historical execution scripts, source Kaggle analysis artifacts, and trained weights remain available in the project archives/backups. Access to private Kaggle datasets is not guaranteed for third parties; this repository alone does not currently constitute a complete from-scratch reproduction package.

**Do not use `Run All` on the review notebooks to attempt retraining.** No re-evaluation on the final 56-case test split is required for this repository update.

## 5. Checkpoints, data, and research transparency

- Large model checkpoints (`.pt`, `.pth`, `.ckpt`), MRI volumes (`.nii`, `.nii.gz`), private credentials, and subject-level result tables are **not included** in this release.
- Kaggle Dataset identifiers and SHA256 records are included where available for provenance and integrity checking. Private datasets require their owner's access permissions.
- The 55-case validation results supported M1–M6 comparison and frozen checkpoint selection; test results must **not** be used to revise that selection.
- The experiment source commit and subsequent refactor commit serve different purposes: historical execution provenance vs. code/documentation cleanup. Do not present the new module files as having been used verbatim for all past training runs.

## 6. Limitations and documentation

This is an internal, single-split research comparison with limited evidence for external generalization; model configurations also differ in compute cost. The test set had previous historical exposure elsewhere in the project. No result should be treated as evidence of clinical safety or diagnostic fitness.

Further background and experimental details:

- [M1–M6 research overview](docs/M1_M6_RESEARCH_OVERVIEW.md)
- [Original baseline README archived for historical reference](docs/README_LEGACY_BASELINE.md) — *describes the earlier state of the project and contains superseded status statements.*

**Research/education use only; not a medical device.**
