# Testing Results and Limitations

## Package verification performed during finalization

- Python source compilation: passed.
- Frontend JavaScript syntax check (`node --check`): passed.
- Pytest: 16 passed, 1 skipped in the packaging environment.
- The skipped test depends on the bundled video smoke-test file being present.
- TensorFlow was not installed in the packaging environment, so model loading and end-to-end neural inference could not be executed here. Install `requirements/backend.txt` in the target runtime and verify `/health` reports `experimental` before demonstrating inference.

## Supplied model reports

The ZIP contained several reports from different training and evaluation experiments. Their metrics are not directly interchangeable because they use different split procedures and may represent different checkpoints.

| Report | Recorded result | Interpretation |
|---|---|---|
| `uadfv_mobilenetv2_report.json` | Frame validation accuracy 75.6%, AUC 0.825 | Frame-level validation on a small UADFV split |
| `uadfv_video_evaluation.json` | 20 videos, accuracy 90%, AUC 0.94 | One recorded video-level split; not external validation |
| `uadfv_5fold_cv_report.json` | Mean accuracy 40.84%, mean AUC 0.467 | Five-fold video-level cross-validation; shows poor stability/generalization |
| `uadfv_fold1_finetuned_report.json` | Accuracy 50%, AUC 0.60 | One fold-1 fine-tuning experiment |

The separate FaceForensics++ c23 notebook experiment discussed during development reported validation video-level accuracy 50% and AUC 0.50 on 16 videos after fine-tuning. Its trained model file was not present in the uploaded ZIP.

## Known technical limitations

1. Small datasets make estimates unstable.
2. UADFV results vary substantially by split; the 90% result should not be presented without the cross-validation caveat.
3. FF++ validation performance was near chance in the recorded experiment.
4. Haar Cascade is an older face-localization method. It may miss small, side-facing, or occluded faces and may return false positives.
5. Face detection counts are diagnostics only and are not fed into the current full-frame MobileNetV2 classifier.
6. The current classifier is frame-based and does not model temporal inconsistencies.
7. The live camera mode scores individual frames; it is not a full video-call implementation.
8. Risk thresholds are heuristic and not calibrated.
9. No audio deepfake detector is included.
10. Model loading/inference requires a compatible TensorFlow runtime; test the exact target machine before a live demonstration.

## Recommended academic wording

Describe DeepShield as an **experimental AI-assisted deepfake video risk analysis prototype**. State that it demonstrates video sampling, face localization, MobileNetV2 frame classification, API integration, and a WebSocket camera-frame demo. Report the supplied metrics with their evaluation protocols and caveats. Do not claim production accuracy or guaranteed detection.
