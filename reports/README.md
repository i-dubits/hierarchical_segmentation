# Reports Registry

This directory contains dataset analysis, evaluation summaries, and qualitative visualization artifacts.

## Table of Contents
- [Report to Model Map](#report-to-model-map)
- [Model-Centric View](#model-centric-view)
- [Notes](#notes)

## Report to Model Map

| Report path | Artifact type | Connected output folder | Model / config | Notes |
|---|---|---|---|---|
| `dataset_report.md` | Dataset analysis (markdown) | n/a | n/a | Inventory, class balance, split stats |
| `dataset_stats.json` | Dataset analysis (raw JSON) | n/a | n/a | Source data for `dataset_report.md` |
| `deeplabv3_mnv3_ft_bs16_cosine_test_metrics.json` | Test metrics JSON | `outputs/deeplabv3_mnv3_ft_bs16_cosine/` | DeepLabV3 + MobileNetV3, `configs/deeplabv3_mnv3_ft.yaml` | Quantitative eval |
| `deeplabv3_mnv3_ft_tuned_bs28_384_test_metrics.json` | Test metrics JSON | `outputs/deeplabv3_mnv3_ft_tuned_bs28_384/` | DeepLabV3 + MobileNetV3, `configs/deeplabv3_mnv3_ft_tuned.yaml` | Quantitative eval |
| `deeplabv3plus_timm_mnv3_bs20_384_test_metrics.json` | Test metrics JSON | `outputs/deeplabv3plus_timm_mnv3_bs20_384/` | DeepLabV3+ (SMP) + timm MobileNetV3, `configs/deeplabv3plus_timm_mnv3.yaml` | Quantitative eval |
| `deeplabv3plus_timm_mnv3_dice35_from_scratch_bs20_nw0_test_metrics.json` | Test metrics JSON | `outputs/deeplabv3plus_timm_mnv3_dice35_from_scratch_bs20_nw0/` | DeepLabV3+ (SMP) + timm MobileNetV3 + Dice(L2), `configs/deeplabv3plus_timm_mnv3_dice_35.yaml` | Quantitative eval |
| `deeplabv3plus_timm_mnv3_dice35_present_crop_wu4_lr12e5_pat6_test_metrics.json` | Test metrics JSON | `outputs/deeplabv3plus_timm_mnv3_dice35_present_crop_wu4_lr12e5_pat6/` | DeepLabV3+ (SMP) + timm MobileNetV3 + Class-Aware Crop + Present-Only Dice(L2), `configs/deeplabv3plus_timm_mnv3_dice35_present_crop_wu4_lr12e5_pat6.yaml` | Quantitative eval |
| `segformer_mitb1_lovasz_dice35_present_crop_wu4_lr6e4_pat6_bs14_acc2_test_metrics.json` | Test metrics JSON | `outputs/segformer_mitb1_lovasz_dice35_present_crop_wu4_lr6e4_pat6_bs14_acc2/` | SegFormer (SMP) + MiT-B1 + Class-Aware Crop + Lovasz + Present-Only Dice(L2), `configs/segformer_mitb1_lovasz_dice35_present_crop_wu4_lr6e4_pat6.yaml` | Quantitative eval |
| `segformer_mitb1_lovasz_dice35_present_crop_aug_bidir_test_metrics.json` | Test metrics JSON | `outputs/segformer_mitb1_lovasz_dice35_present_crop_aug_bidir/` | SegFormer (SMP) + MiT-B1 + Class-Aware Crop + Scale Jitter + Photometric Distortion + Bidirectional Consistency + Lovasz + Present-Only Dice(L2), `configs/segformer_mitb1_lovasz_dice35_present_crop_aug_bidir.yaml` | Quantitative eval |
| `deeplabv3plus_timm_mnv3_continue40_from_last_e26_bs20_test_metrics.json` | Test metrics JSON | `outputs/deeplabv3plus_timm_mnv3_continue40_from_last_e26_bs20/` | DeepLabV3+ (SMP) + timm MobileNetV3, `configs/deeplabv3plus_timm_mnv3_continue_40_from26.yaml` | Quantitative eval |
| `deeplabv3plus_timm_mnv3_continue40_lr5e5_bs20_test_metrics.json` | Test metrics JSON | `outputs/deeplabv3plus_timm_mnv3_continue40_lr5e5_bs20/` | DeepLabV3+ (SMP) + timm MobileNetV3, `configs/deeplabv3plus_timm_mnv3_continue_40.yaml` | Quantitative eval |
| `lraspp_mnv3_ft_bs28_384_test_metrics.json` | Test metrics JSON | `outputs/lraspp_mnv3_ft_bs28_384/` | LRASPP + MobileNetV3, `configs/lraspp_mnv3_ft.yaml` | Quantitative eval |
| `unet_baseline_mlflow_test_bs12_c48_320_test_metrics.json` | Test metrics JSON | `outputs/unet_baseline_mlflow_test_bs12_c48_320/` | U-Net baseline, `configs/unet_baseline_mlflow_test.yaml` | Historical eval artifact (non-canonical for final reporting) |
| `test_predictions_bs16_cosine/` | Qualitative test predictions | `outputs/deeplabv3_mnv3_ft_bs16_cosine/` | DeepLabV3 + MobileNetV3, `configs/deeplabv3_mnv3_ft.yaml` | 282 test panels |
| `test_predictions_deeplabv3plus_timm_mnv3_bs20_384/` | Qualitative test predictions | `outputs/deeplabv3plus_timm_mnv3_bs20_384/` | DeepLabV3+ (SMP), `configs/deeplabv3plus_timm_mnv3.yaml` | 282 test panels |
| `test_predictions_deeplabv3plus_timm_mnv3_dice35_from_scratch_bs20_nw0/` | Qualitative test predictions | `outputs/deeplabv3plus_timm_mnv3_dice35_from_scratch_bs20_nw0/` | DeepLabV3+ (SMP) + Dice(L2), `configs/deeplabv3plus_timm_mnv3_dice_35.yaml` | 282 test panels |
| `test_predictions_deeplabv3plus_timm_mnv3_dice35_present_crop_wu4_lr12e5_pat6/` | Qualitative test predictions | `outputs/deeplabv3plus_timm_mnv3_dice35_present_crop_wu4_lr12e5_pat6/` | DeepLabV3+ (SMP) + Class-Aware Crop + Present-Only Dice(L2), `configs/deeplabv3plus_timm_mnv3_dice35_present_crop_wu4_lr12e5_pat6.yaml` | 282 test panels |
| `test_predictions_segformer_mitb1_lovasz_dice35_present_crop_wu4_lr6e4_pat6_bs14_acc2/` | Qualitative test predictions | `outputs/segformer_mitb1_lovasz_dice35_present_crop_wu4_lr6e4_pat6_bs14_acc2/` | SegFormer (SMP) + MiT-B1 + Class-Aware Crop + Lovasz + Present-Only Dice(L2), `configs/segformer_mitb1_lovasz_dice35_present_crop_wu4_lr6e4_pat6.yaml` | 282 test panels |
| `test_predictions_segformer_mitb1_lovasz_dice35_present_crop_aug_bidir/` | Qualitative test predictions | `outputs/segformer_mitb1_lovasz_dice35_present_crop_aug_bidir/` | SegFormer (SMP) + MiT-B1 + Class-Aware Crop + Scale Jitter + Photometric Distortion + Bidirectional Consistency + Lovasz + Present-Only Dice(L2), `configs/segformer_mitb1_lovasz_dice35_present_crop_aug_bidir.yaml` | 282 test panels |
| `test_predictions_deeplabv3plus_timm_mnv3_continue40_from_last_e26_bs20/` | Qualitative test predictions | `outputs/deeplabv3plus_timm_mnv3_continue40_from_last_e26_bs20/` | DeepLabV3+ (SMP), `configs/deeplabv3plus_timm_mnv3_continue_40_from26.yaml` | 282 test panels |
| `test_predictions_lraspp_mnv3_ft_bs28_384/` | Qualitative test predictions | `outputs/lraspp_mnv3_ft_bs28_384/` | LRASPP + MobileNetV3, `configs/lraspp_mnv3_ft.yaml` | 282 test panels |
| `test_predictions_mnv3_tuned_bs28_384/` | Qualitative test predictions | `outputs/deeplabv3_mnv3_ft_tuned_bs28_384/` | DeepLabV3 + MobileNetV3, `configs/deeplabv3_mnv3_ft_tuned.yaml` | 282 test panels |
| `test_predictions_unet_baseline_mlflow_resume_30/` | Qualitative test predictions | `outputs/unet_baseline_mlflow_test_bs12_c48_320/` | U-Net baseline, `configs/unet_baseline_mlflow_resume_30.yaml` | 282 test panels |
| `test_predictions_unet_baseline_mlflow_test_bs12_c48_320/` | Qualitative test predictions | `outputs/unet_baseline_mlflow_test_bs12_c48_320/` | U-Net baseline, `configs/unet_baseline_mlflow_test.yaml` | 282 test panels |
| `visual_analysis/` | Qualitative dataset/mask visualization | n/a (dataset-level artifact) | n/a | Uses class palettes and overlays |

## Model-Centric View

| Model run (outputs) | Metrics source | Prediction folder |
|---|---|---|
| `deeplabv3_mnv3_ft_bs16_cosine` | `deeplabv3_mnv3_ft_bs16_cosine_test_metrics.json` | `test_predictions_bs16_cosine/` |
| `deeplabv3_mnv3_ft_tuned_bs28_384` | `deeplabv3_mnv3_ft_tuned_bs28_384_test_metrics.json` | `test_predictions_mnv3_tuned_bs28_384/` |
| `deeplabv3plus_timm_mnv3_bs20_384` | `deeplabv3plus_timm_mnv3_bs20_384_test_metrics.json` | `test_predictions_deeplabv3plus_timm_mnv3_bs20_384/` |
| `deeplabv3plus_timm_mnv3_dice35_from_scratch_bs20_nw0` | `deeplabv3plus_timm_mnv3_dice35_from_scratch_bs20_nw0_test_metrics.json` | `test_predictions_deeplabv3plus_timm_mnv3_dice35_from_scratch_bs20_nw0/` |
| `deeplabv3plus_timm_mnv3_dice35_present_crop_wu4_lr12e5_pat6` | `deeplabv3plus_timm_mnv3_dice35_present_crop_wu4_lr12e5_pat6_test_metrics.json` | `test_predictions_deeplabv3plus_timm_mnv3_dice35_present_crop_wu4_lr12e5_pat6/` |
| `segformer_mitb1_lovasz_dice35_present_crop_wu4_lr6e4_pat6_bs14_acc2` | `segformer_mitb1_lovasz_dice35_present_crop_wu4_lr6e4_pat6_bs14_acc2_test_metrics.json` | `test_predictions_segformer_mitb1_lovasz_dice35_present_crop_wu4_lr6e4_pat6_bs14_acc2/` |
| `segformer_mitb1_lovasz_dice35_present_crop_aug_bidir` | `segformer_mitb1_lovasz_dice35_present_crop_aug_bidir_test_metrics.json` | `test_predictions_segformer_mitb1_lovasz_dice35_present_crop_aug_bidir/` |
| `deeplabv3plus_timm_mnv3_continue40_from_last_e26_bs20` | `deeplabv3plus_timm_mnv3_continue40_from_last_e26_bs20_test_metrics.json` | `test_predictions_deeplabv3plus_timm_mnv3_continue40_from_last_e26_bs20/` |
| `deeplabv3plus_timm_mnv3_continue40_lr5e5_bs20` | `deeplabv3plus_timm_mnv3_continue40_lr5e5_bs20_test_metrics.json` | not generated yet |
| `lraspp_mnv3_ft_bs28_384` | `lraspp_mnv3_ft_bs28_384_test_metrics.json` | `test_predictions_lraspp_mnv3_ft_bs28_384/` |
| `unet_baseline_mlflow_test_bs12_c48_320` | `outputs/unet_baseline_mlflow_test_bs12_c48_320/summary.json` | `test_predictions_unet_baseline_mlflow_test_bs12_c48_320/`, `test_predictions_unet_baseline_mlflow_resume_30/` |

## Notes

- The best current test `mIoU^2` artifact is connected to `outputs/segformer_mitb1_lovasz_dice35_present_crop_wu4_lr6e4_pat6_bs14_acc2/` (`miou_l2=0.5784`).
- The latest SegFormer augmentation + bidirectional-consistency run is connected to `outputs/segformer_mitb1_lovasz_dice35_present_crop_aug_bidir/` (`miou_l2=0.5704`).
- Canonical metrics source per run is listed in the **Model-Centric View** table.
- `unet_baseline_mlflow_test_bs12_c48_320_test_metrics.json` and `outputs/unet_baseline_mlflow_test_bs12_c48_320/summary.json` report different values; use `summary.json` as canonical for the final U-Net row.
