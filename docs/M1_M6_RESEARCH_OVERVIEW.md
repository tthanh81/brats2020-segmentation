# BraTS2020 — M1–M6 research extension (2026)

This document summarizes the completed M1–M6 validation study and the frozen final evaluation of M6. It supplements the earlier 3D U-Net baseline and extended-training material in the main README; it does not replace that experiment history.

## Study scope

- **Data:** BraTS2020 labeled training cases, with a fixed subject-level seed-42 split: 257 training, 55 internal validation, and 56 internal held-out test subjects.
- **Inputs:** T1, T1ce, T2, and FLAIR MRI. Segmentation labels are mapped to the four model output classes (background, NCR/NET, ED, ET).
- **Validation protocol:** full-volume sliding-window inference, ROI `96 × 96 × 96`, overlap `0.25`; Dice for whole tumor (WT), tumor core (TC), and enhancing tumor (ET).
- **Comparison:** validation results were used for model selection. The final M6 checkpoint was fixed before its 56-case test evaluation in this workflow.
- **Dataset caveat:** the 56 cases form an **internal hold-out** from the publicly labeled BraTS2020 training collection. This is **not** the official BraTS challenge test set. The repository also contains older historical test results; the hold-out should **not** be described as untouched throughout the entire project history.

## Experimental configurations and validation results

All numbers below refer to the same 55-case **internal validation split**. The reported checkpoints are at epoch 20.

| Model | Main change | WT | TC | ET | Mean Dice |
|---|---|---:|---:|---:|---:|
| M1 | 3D U-Net baseline, 20 epochs | 0.861699 | 0.743321 | 0.667613 | 0.757544 |
| M2 | 3D augmentation | 0.827139 | 0.727427 | 0.660358 | 0.738308 |
| M3 | Cosine learning-rate schedule | 0.865705 | 0.753051 | 0.671339 | 0.763365 |
| M4a | Dice + weighted CE + focal loss | 0.862155 | 0.743206 | 0.682176 | 0.762512 |
| M4b | Class-balanced patch sampling | 0.857675 | 0.733379 | 0.661771 | 0.750942 |
| M5 | Cosine schedule + combined loss | 0.868797 | 0.752799 | 0.678467 | 0.766687 |
| **M6** | **Swin UNETR** | **0.870206** | **0.760337** | **0.697429** | **0.775991** |

**Training comparability:** M2–M5 continued from M1's epoch-10 checkpoint through epoch 20; M6 was initialized from scratch and trained for 20 epochs. Equal epoch counts do not imply identical computational budgets. The selected M6 used **15,705,646 parameters**.

See [`../results/m1_m6_summary/validation_seven_model_means_summary.csv`](../results/m1_m6_summary/validation_seven_model_means_summary.csv) for higher precision. Aggregate values in this repository were transcribed from verified Kaggle outputs and **are not byte-identical exports of source result files**.

## Frozen final test: M6 only

The M6 epoch-20 checkpoint was selected **before** the 56-case internal test evaluation.

| Region | Test Dice | Patient-level bootstrap 95% CI |
|---|---:|---:|
| WT | 0.871992 | [0.849028, 0.892578] |
| TC | 0.808354 | [0.757233, 0.852442] |
| ET | 0.723875 | [0.648604, 0.790752] |
| **Mean Dice** | **0.801407** | **[0.762350, 0.836853]** |

- **Test subjects:** 56/56; four had empty ET ground-truth masks.
- **Frozen checkpoint SHA256:** `02a2930122c3994fa72342921c719f1f001d540f4fb2e106fe293607ed527421`.
- **Evaluation restriction:** do not select another model, tune parameters, or retrain based on these test results.
- **Provenance:** Kaggle Dataset `tthanhh811/brats2020-m6-finaltest-56cases-seed42`; four independently verified 14-case chunks and four final summary artifacts. Access may be restricted because the project backups are private.

For the detailed public-safe aggregate summary, see [`../results/m1_m6_summary/m6_final_test_public_summary.json`](../results/m1_m6_summary/m6_final_test_public_summary.json).

## Code, notebooks, and reproducibility

- [`../notebooks/`](../notebooks/) includes the historical notebooks 01–05 and **review-only** notebooks 06–11. The latter are **not** independently tested end-to-end training scripts and are not intended for `Run All` to reproduce the study.
- The refactor candidates [`../src/models/m4a_combined_loss.py`](../src/models/m4a_combined_loss.py), [`../src/data/patches_class_balanced.py`](../src/data/patches_class_balanced.py), and [`../src/models/swin_unetr.py`](../src/models/swin_unetr.py) passed synthetic CPU smoke checks using PyTorch 2.11.0+cpu and MONAI 1.6.1.
- The historical execution code and original notebooks are **archived separately** in the research preparation package and Kaggle versions. Paths such as `original_notebooks/` and `scripts/historical_cells/` mentioned in the review notebooks refer to that **separate local archive; these directories are not included in this GitHub release**. Therefore, the review notebooks must not be represented as self-contained reproducibility artifacts.
- The shared-source commit used for the experiments is `79a95a08f11694278114a89348cf8303a1e39d67`, on branch `experiment/unet3d-optimization`. This differs from the commit that will contain the later refactor/documentation.
- Model weights (`.pt`), MRI volumes (`.nii`/`.nii.gz`), patient-level data, and credentials are intentionally omitted from this release. Model checkpoints are backed up in Kaggle Datasets; private Dataset IDs are references, not guarantees of public downloads.
- Paired validation significance tests, detailed training curves, checkpoint dataset manifest for **all** models, and independently tested full training runners are **additional publication tasks**; their absence must not be obscured by the aggregate results above.

## Limitations

Results are from one internal subject split and one training run per configuration, with distinct computational budgets. The retrospective existence of older test information in the repository limits claims of a perfectly untouched hold-out across the project's entire history. These results are research benchmarks, **not** evidence of clinical readiness.
