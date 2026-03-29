# Hierarchical Semantic Segmentation (Pascal-Part)

This repository contains a hierarchical segmentation pipeline and baseline/fine-tuning experiments for the test assignment:

This is the result of both Codex and hierarchical segmentation approaches testing.

- dataset: prepared Pascal-Part subset (`JPEGImages` + `gt_masks` as `.npy`)
- task: hierarchical semantic segmentation
- required metrics: `mIoU^0`, `mIoU^1`, `mIoU^2` (all excluding background)

TODO:
 - add access to MLFlow logs
 - configure DVC and best weights downloading
 - Docker image with the best model inference
 - details report on segmentation approaches and losses


## Hierarchy

Fine classes (`level 2`):

- `0`: background
- `1`: low_hand
- `2`: torso
- `3`: low_leg
- `4`: head
- `5`: up_leg
- `6`: up_hand

Derived hierarchy:

- `level 0`: `body` (`mask != 0`)
- `level 1`: `upper_body = {1,2,4,6}`, `lower_body = {3,5}`

## Project Structure

```text
configs/         # YAML configs (common + model-specific overrides)
scripts/         # train / evaluate / infer entrypoints
src/hseg/        # core package (data, model, loss, metrics, trainer)
reports/         # experiment notes and result artifacts
Plan.md          # research notes + execution plan
```

Config layout:

- `configs/common.yaml`: shared defaults (data/training/mlflow/inference palette)
- `configs/*.yaml`: experiment/model overrides using `base_config: common.yaml`

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Pre-Commit and CI Checks

Install and enable local git hooks:

```bash
python3 -m pip install pre-commit
pre-commit install
PRE_COMMIT_HOME=/tmp/pre-commit-cache pre-commit run --all-files
```

What is configured:

- hook config: `.pre-commit-config.yaml`
- linter config: `pyproject.toml` (Ruff on `src/` and `scripts/`)
- CI workflow: `.github/workflows/pre-commit.yml` (runs same hooks on `push` and `pull_request`)
- if pre-commit cache permissions fail in sandbox, set `PRE_COMMIT_HOME=/tmp/pre-commit-cache`

## Dataset Layout

Place extracted data under `data/Pascal-part` (or override with `--data-root`):

```text
data/Pascal-part/
├── JPEGImages/
│   ├── xxx.jpg
│   └── ...
└── gt_masks/
    ├── xxx.npy
    └── ...
```

File stems must match (`xxx.jpg` <-> `xxx.npy`).

## Train

```bash
python3 scripts/train.py \
  --config configs/baseline.yaml \
  --data-root data/Pascal-part \
  --output-dir outputs/baseline
```

What happens:

- if `train_id.txt` and `val_id.txt` are present in `data-root`, `splits.json` is created from those files
- otherwise a deterministic train/val/test split is generated and saved to `splits.json`
- model checkpoints are stored in `outputs/baseline/` (`best.pt`, `last.pt`)
- full epoch history is stored in `history.json`
- final summary is stored in `summary.json`
- early stopping is enabled by default and monitors validation `loss` (`training.early_stopping`)
- optional class-aware crop can be enabled via `data.class_aware_crop` in YAML

Class-aware crop config (example):

```yaml
data:
  class_aware_crop:
    enabled: true
    probability: 0.4
    crop_size: [320, 320]
    target_class_ids: [1, 3, 5]
    min_target_pixels: 48
    max_tries: 12
    fallback_to_random_crop: true
```

## MLflow Monitoring

Training supports optional MLflow logging (disabled by default).

Start UI (from repo root):

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Then launch training with MLflow enabled:

```bash
python3 scripts/train.py \
  --config configs/deeplabv3_mnv3_ft_tuned.yaml \
  --data-root data/Pascal-part \
  --output-dir outputs/deeplabv3_mnv3_ft_tuned_bs28_384 \
  --mlflow \
  --mlflow-tracking-uri file:./mlruns \
  --mlflow-experiment pascal-part-segmentation \
  --mlflow-run-name mnv3_bs28_384
```

Or use the preconfigured baseline test file:

```bash
python3 scripts/train.py \
  --config configs/unet_baseline_mlflow_test.yaml \
  --data-root data/Pascal-part \
  --output-dir outputs/unet_baseline_mlflow_test_bs12_c48_320
```

Logged to MLflow:

- train and validation metrics each epoch (`train/*`, `val/*`)
- final metrics under split-aware namespace (`final/<split>/*`), plus `test/*` alias when final split is `test`
- optimizer LR (`train/lr`) and per-param-group LR (`train/lr_group_*`) + best-checkpoint metric tracking
- config and output artifacts (`history.json`, `summary.json`, `best.pt`, split file)

## Evaluate

```bash
python3 scripts/evaluate.py \
  --checkpoint outputs/baseline/best.pt \
  --split val \
  --data-root data/Pascal-part \
  --output-json outputs/baseline/val_metrics.json
```

The script prints and optionally saves:

- `miou_l0` = `mIoU^0` (`body`)
- `miou_l1` = `mIoU^1` (`upper_body`, `lower_body`)
- `miou_l2` = `mIoU^2` (6 fine classes)
- by default, uses config embedded in checkpoint (or pass `--config ...`)
- evaluation requires existing split file; use `--allow-generate-split` only when intentional

## Inference (Single Image)

```bash
python3 scripts/infer.py \
  --checkpoint outputs/baseline/best.pt \
  --image data/Pascal-part/JPEGImages/example.jpg \
  --output-dir outputs/infer
```

Outputs:

- raw predictions as `.npy` for `l0/l1/l2`
- colorized masks as `.png` for `l0/l1/l2`
- by default, uses config embedded in checkpoint (or pass `--config ...`)

## Notes on the Baseline

- model: U-Net style encoder-decoder with three heads (`l0`, `l1`, `l2`)
- loss: weighted CE at all levels + consistency regularization from fine predictions to coarse heads
- metrics: IoU via confusion matrices, with background excluded from required mIoU averages

This is a robust baseline intended for fast iteration. Next quality gains typically come from:

- better augmentation policy
- stronger encoder
- class re-weighting / sampling for rare parts
- controlled ablations on consistency loss

## Fine-Tuning DeepLabV3-MobileNetV3

Use the provided fine-tuning config:

```bash
python3 scripts/train.py \
  --config configs/deeplabv3_mnv3_ft.yaml \
  --data-root data/Pascal-part \
  --output-dir outputs/deeplabv3_mnv3_ft_bs16_cosine
```

This run stores:

- train/validation metrics per epoch in `outputs/deeplabv3_mnv3_ft_bs16_cosine/history.json`
- best/last checkpoints in `outputs/deeplabv3_mnv3_ft_bs16_cosine/{best.pt,last.pt}`
- final evaluation (uses `test` split if present) in `outputs/deeplabv3_mnv3_ft_bs16_cosine/summary.json`
- an additional tuned DeepLabV3 variant is available at `outputs/deeplabv3_mnv3_ft_tuned_bs28_384/`

## Fine-Tuning LR-ASPP-MobileNetV3

```bash
python3 scripts/train.py \
  --config configs/lraspp_mnv3_ft.yaml \
  --data-root data/Pascal-part \
  --output-dir outputs/lraspp_mnv3_ft_bs28_384
```

## Fine-Tuning DeepLabV3+ (SMP + timm Encoder)

Use the SMP-based DeepLabV3+ config with a timm MobileNetV3 encoder:

```bash
python3 scripts/train.py \
  --config configs/deeplabv3plus_timm_mnv3.yaml \
  --data-root data/Pascal-part \
  --output-dir outputs/deeplabv3plus_timm_mnv3_bs20_384
```

This model keeps the same hierarchical API and losses (`logits_l0/l1/l2`) but uses a shared
DeepLabV3+ encoder/decoder from `segmentation_models_pytorch` and three custom hierarchy heads.
You can switch backbones via `model.encoder_name` (for example, `tu-mobilenetv3_large_100`).
The strongest baseline continuation artifact for this family is
`outputs/deeplabv3plus_timm_mnv3_continue40_from_last_e26_bs20/`.

## DeepLabV3+ + Dice (35 Epochs, From Scratch)

Dice on level-2 logits can be enabled from config:

```bash
python3 scripts/train.py \
  --config configs/deeplabv3plus_timm_mnv3_dice_35.yaml \
  --data-root data/Pascal-part \
  --output-dir outputs/deeplabv3plus_timm_mnv3_dice35_from_scratch_bs20_nw0
```

Key artifacts for this run:

- training outputs: `outputs/deeplabv3plus_timm_mnv3_dice35_from_scratch_bs20_nw0/`
- test metrics JSON: `reports/deeplabv3plus_timm_mnv3_dice35_from_scratch_bs20_nw0_test_metrics.json`
- qualitative test predictions: `reports/test_predictions_deeplabv3plus_timm_mnv3_dice35_from_scratch_bs20_nw0/`

## DeepLabV3+ + Class-Aware Crop + Present-Only Dice

Run command:

```bash
python3 scripts/train.py \
  --config configs/deeplabv3plus_timm_mnv3_dice35_present_crop_wu4_lr12e5_pat6.yaml \
  --data-root data/Pascal-part \
  --output-dir outputs/deeplabv3plus_timm_mnv3_dice35_present_crop_wu4_lr12e5_pat6
```

Key artifacts for this run:

- training outputs: `outputs/deeplabv3plus_timm_mnv3_dice35_present_crop_wu4_lr12e5_pat6/`
- test metrics JSON: `reports/deeplabv3plus_timm_mnv3_dice35_present_crop_wu4_lr12e5_pat6_test_metrics.json`
- qualitative test predictions: `reports/test_predictions_deeplabv3plus_timm_mnv3_dice35_present_crop_wu4_lr12e5_pat6/`

## SegFormer (SMP MiT-B1) + Class-Aware Crop + Lovasz + Present-Only Dice (Current Best)

Run command:

```bash
python3 scripts/train.py \
  --config configs/segformer_mitb1_lovasz_dice35_present_crop_wu4_lr6e4_pat6.yaml \
  --data-root data/Pascal-part \
  --output-dir outputs/segformer_mitb1_lovasz_dice35_present_crop_wu4_lr6e4_pat6_bs14_acc2
```

Key artifacts for this run:

- training outputs: `outputs/segformer_mitb1_lovasz_dice35_present_crop_wu4_lr6e4_pat6_bs14_acc2/`
- test metrics JSON: `reports/segformer_mitb1_lovasz_dice35_present_crop_wu4_lr6e4_pat6_bs14_acc2_test_metrics.json`
- qualitative test predictions: `reports/test_predictions_segformer_mitb1_lovasz_dice35_present_crop_wu4_lr6e4_pat6_bs14_acc2/`

## SegFormer (SMP MiT-B1) + Added Augmentations + Bidirectional Consistency

Run command:

```bash
python3 scripts/train.py \
  --config configs/segformer_mitb1_lovasz_dice35_present_crop_aug_bidir.yaml \
  --data-root data/Pascal-part \
  --output-dir outputs/segformer_mitb1_lovasz_dice35_present_crop_aug_bidir
```

Key artifacts for this run:

- training outputs: `outputs/segformer_mitb1_lovasz_dice35_present_crop_aug_bidir/`
- training summary: `outputs/segformer_mitb1_lovasz_dice35_present_crop_aug_bidir/summary.json`
- test metrics JSON: `reports/segformer_mitb1_lovasz_dice35_present_crop_aug_bidir_test_metrics.json`
- qualitative test predictions: `reports/test_predictions_segformer_mitb1_lovasz_dice35_present_crop_aug_bidir/`

Notes for this run:

- keeps the previous SegFormer warmup schedule and class-aware crop
- adds scale jitter + photometric distortion
- switches consistency from one-way to bidirectional MSE
- final test metrics: `mIoU^2=0.5704`, `mIoU^1=0.6899`, `mIoU^0=0.8296`, `loss=0.4680`
- this did not exceed the current best SegFormer test `mIoU^2=0.5784`

## Qualitative Test Examples (SegFormer Best Run)

Each row shows exactly: original image, original image with true mask overlay, and original image with predicted mask overlay.

| Sample | Original Image | Original + True Mask | Original + Predicted Mask |
|---|---|---|---|
| 2008_000003 | ![orig-2008_000003](reports/readme_examples_segformer/001_2008_000003_original.jpg) | ![gt-overlay-2008_000003](reports/readme_examples_segformer/001_2008_000003_gt_overlay.jpg) | ![pred-overlay-2008_000003](reports/readme_examples_segformer/001_2008_000003_pred_overlay.jpg) |
| 2008_000090 | ![orig-2008_000090](reports/readme_examples_segformer/002_2008_000090_original.jpg) | ![gt-overlay-2008_000090](reports/readme_examples_segformer/002_2008_000090_gt_overlay.jpg) | ![pred-overlay-2008_000090](reports/readme_examples_segformer/002_2008_000090_pred_overlay.jpg) |
| 2008_000096 | ![orig-2008_000096](reports/readme_examples_segformer/003_2008_000096_original.jpg) | ![gt-overlay-2008_000096](reports/readme_examples_segformer/003_2008_000096_gt_overlay.jpg) | ![pred-overlay-2008_000096](reports/readme_examples_segformer/003_2008_000096_pred_overlay.jpg) |
| 2008_000367 | ![orig-2008_000367](reports/readme_examples_segformer/007_2008_000367_original.jpg) | ![gt-overlay-2008_000367](reports/readme_examples_segformer/007_2008_000367_gt_overlay.jpg) | ![pred-overlay-2008_000367](reports/readme_examples_segformer/007_2008_000367_pred_overlay.jpg) |
| 2008_000406 | ![orig-2008_000406](reports/readme_examples_segformer/008_2008_000406_original.jpg) | ![gt-overlay-2008_000406](reports/readme_examples_segformer/008_2008_000406_gt_overlay.jpg) | ![pred-overlay-2008_000406](reports/readme_examples_segformer/008_2008_000406_pred_overlay.jpg) |

## Final Results (Test Split)

Your understanding is correct: the main trained model families are:
- U-Net
- DeepLabV3
- LRASPP
- DeepLabV3+
- SegFormer

Some runs were trained in stages (resume/continuation). The table below includes only the final selected run per model family.

| Model family | Final selected run | Training stage note | `mIoU^2` | `mIoU^1` | `mIoU^0` | `loss` |
|---|---|---|---:|---:|---:|---:|
| U-Net | `outputs/unet_baseline_mlflow_test_bs12_c48_320/` | base run + resumed to 30 epochs | 0.2821 | 0.4322 | 0.6347 | 0.5900 |
| DeepLabV3 | `outputs/deeplabv3_mnv3_ft_bs16_cosine/` | final single-stage run (selected by best `mIoU^2` in DeepLabV3 family) | 0.4551 | 0.6094 | 0.7730 | 0.3940 |
| LRASPP | `outputs/lraspp_mnv3_ft_bs28_384/` | single-stage run with early stop | 0.4331 | 0.5768 | 0.7448 | 0.4469 |
| DeepLabV3+ | `outputs/deeplabv3plus_timm_mnv3_dice35_present_crop_wu4_lr12e5_pat6/` | class-aware rare-class crop + present-only Dice; stopped at epoch 28 (patience 6) | 0.5097 | 0.6424 | 0.7912 | 0.4459 |
| SegFormer | `outputs/segformer_mitb1_lovasz_dice35_present_crop_wu4_lr6e4_pat6_bs14_acc2/` | MiT-B1 pretrained + class-aware crop + Lovasz + present-only Dice; stopped at epoch 34 (patience 6) | 0.5784 | 0.6964 | 0.8262 | 0.4847 |

Canonical metric source for each run is listed in `reports/README.md` (U-Net final row uses `outputs/unet_baseline_mlflow_test_bs12_c48_320/summary.json`).

Latest non-selected experiment:
- `outputs/segformer_mitb1_lovasz_dice35_present_crop_aug_bidir/` reached test `mIoU^2=0.5704`, `mIoU^1=0.6899`, `mIoU^0=0.8296`, `loss=0.4680`

## Best Model

Best overall model (by primary fine-grained metric `mIoU^2`):
- `SegFormer (SMP) + MiT-B1 + Class-Aware Crop + Lovasz + Present-Only Dice(L2)` from `outputs/segformer_mitb1_lovasz_dice35_present_crop_wu4_lr6e4_pat6_bs14_acc2/`
- Final test metrics: `mIoU^2=0.5784`, `mIoU^1=0.6964`, `mIoU^0=0.8262`, `loss=0.4847`
