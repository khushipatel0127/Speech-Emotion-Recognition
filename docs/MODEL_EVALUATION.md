# SpeechSense Model Evaluation

## Evaluation contract

Phase 2 uses six shared emotions and 115 dataset-prefixed speakers. The split is
frozen with seed 42. Model selection uses five-fold stratified grouped
cross-validation on the 80 training speakers followed by evaluation on 17
validation speakers. The 18 final-test speakers remain untouched.

The manifest audit found 8,498 recordings. Seven recordings are excluded from
modeling after exact decoded-PCM hashing: one redundant same-label repeat and all
six members of three conflicting-label duplicate pairs. There are no missing
files, unreadable WAVs, duplicate paths, cross-split content duplicates, or
speaker overlaps.

## Traditional-model results

| Model | CV accuracy | CV macro F1 | Validation accuracy | Validation macro F1 | Validation inference |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 47.12% | 46.74% | 47.22% | 45.95% | 0.004 ms/file |
| Calibrated RBF SVM | 47.66% | 47.25% | **48.51%** | **47.45%** | 1.340 ms/file |
| HistGradientBoosting | **49.09%** | **48.66%** | 48.27% | 47.21% | 0.302 ms/file |
| Extra Trees | 47.93% | 46.09% | 47.86% | 46.34% | 0.463 ms/file |

The selected handcrafted validation candidate is the calibrated RBF SVM. Its
per-emotion validation recall is:

| Emotion | Recall |
|---|---:|
| Angry | 75.23% |
| Disgust | 37.85% |
| Fearful | 42.06% |
| Happy | 37.38% |
| Neutral | 39.18% |
| Sad | 57.48% |

High angry recall alongside low happy recall remains a practical concern.

## Dataset-specific validation

For the selected SVM, mixed-dataset training produced 48.26% accuracy and 47.16%
macro F1 on CREMA-D validation speakers, and 50.00% accuracy and 45.21% macro F1
on RAVDESS validation speakers.

Transfer without mixed-dataset training was substantially worse:

| Training dataset | Validation dataset | Accuracy | Macro F1 |
|---|---|---:|---:|
| CREMA-D | RAVDESS | 31.25% | 24.74% |
| RAVDESS | CREMA-D | 25.07% | 18.07% |

This is direct evidence of domain shift between recording protocols, speakers,
and acted-expression styles.

## Gain robustness

| Gain | Accuracy | Macro F1 | Agreement with 1× | Happy recall | Happy → Angry |
|---:|---:|---:|---:|---:|---:|
| 0.5× | 48.03% | 47.11% | 77.68% | 44.39% | 28/214 |
| 1.0× | 48.51% | 47.45% | 100.00% | 37.38% | 53/214 |
| 2.0× | 43.27% | 40.79% | 70.27% | 21.50% | 87/214 |
| 3.0× | 36.99% | 31.80% | 52.22% | 13.08% | 98/214 |

Artificial gain is only one robustness stress and is not a substitute for real
WhatsApp, microphone, noise, and codec evaluation. The degradation is severe
enough that this candidate should not replace the deployed model.

### Training-only volume augmentation

One deterministic gain-augmented copy (0.5×, 0.75×, 1.5×, or 2.0×) was added
for each training recording. Validation and test audio were never augmented for
fitting. The selected augmented SVM reached 46.17% grouped-CV accuracy, 45.77%
grouped-CV macro F1, 47.62% validation accuracy, and 46.73% validation macro F1.
It therefore trails the unaugmented SVM on the normal validation condition.

The augmentation substantially improves the gain stress test:

| Gain | Accuracy | Macro F1 | Happy recall | Happy → Angry |
|---:|---:|---:|---:|---:|
| 0.5× | 46.82% | 46.01% | 40.65% | 45/214 |
| 1.0× | 47.62% | 46.73% | 39.25% | 49/214 |
| 2.0× | 47.30% | 46.27% | 35.98% | 60/214 |
| 3.0× | 47.22% | 46.05% | 35.05% | 62/214 |

This is a measured robustness/normal-performance tradeoff. The augmented model
is retained as an experiment, but it is not the primary validation winner and
has not been deployed.

## WavLM status

Frozen `microsoft/wavlm-base-plus` extraction and grouped classifier evaluation
are implemented. Local extraction is intentionally not started because PyTorch,
Transformers, model weights, and hours of CPU inference are inappropriate for
the target 8 GB, non-CUDA laptop. The supplied Colab notebook uses small batches,
64-recording resumable shards, and free GPU execution.

No WavLM score or final-test score is reported yet. The 70% accuracy and macro-F1
values remain aspirations rather than achieved results.

## Real-world evaluation

A preliminary manually labelled collection now contains three intentionally
Happy phone recordings from one speaker. All files were readable, 5.50–6.12
seconds long, normally audible, and unclipped. The deployed historical model
predicted Angry for all three, producing 0/3 accuracy; two results were flagged
as low-certainty. This establishes a specific Happy→Angry natural-recording
failure, not an overall natural-speech accuracy estimate.

The current collection is too small, includes only one speaker and one class,
and must be expanded before reporting aggregate real-world performance. The
manifest records speaker ID, human-supplied expected emotion, device, format,
and file path. `evaluate_real_world.py` converts each file through the deployed
FFmpeg/inference path and writes per-recording outputs plus accuracy, macro F1,
uncertainty counts, and a confusion matrix.
