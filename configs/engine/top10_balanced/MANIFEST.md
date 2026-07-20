# IronFlow Balanced GPU Configs

Generated/maintained for the `tank_armor_prepared_v20260703_original_classifier_all_v1` dataset.

Current experiment profile:

- Runtime: SSH GPU / Vast AI native Python
- Remote dataset root: `/workspace/ironflow/prestaged/tank_armor_prepared_v20260703_original_classifier_all_v1`
- Detection dataset: `detector_tank_av/detection`
- Classification dataset: `classifier_all/images`
- Classification uses the combined MBT + armored-vehicle ImageFolder dataset under `classifier_all/images`.
- External wrappers: native mode
- Training profile: RTX 5090 32GB balanced defaults from native params
- Detection epochs: 30
- Classification epochs: 30
- Early stopping patience: 5
- Detector batch policy: YOLO-family detectors use batch 16; DETR/D-FINE/VIT-style detectors use batch 8.
- Classifier batch policy: classifier experiments use batch 32.
- Prediction/evaluation split: `test`
- Artifact collection: light, with best/last weights where enabled

Dataset format expectations:

- YOLO-family detection reads YOLO `data.yaml` and normalized `.txt` labels.
- DETR/D-FINE/VIT-style detection reads COCO annotations under `detector_tank_av/detection/coco/annotations`.
- Classification reads ImageFolder-style original detector images under `classifier_all/images/train`, `val`, and `test`.
- Classification predict tasks consume the `classification_input_manifest.json` emitted by the train task when no source manifest exists in the ImageFolder root.
