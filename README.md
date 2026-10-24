# SpeechSense — Speech Emotion Recognition

SpeechSense is a React and FastAPI application for analyzing emotion-related
acoustic patterns in speech. It accepts uploaded audio or browser microphone
recordings, converts them to 16 kHz mono WAV with FFmpeg, returns model scores,
and stores prediction history in SQLite.

The active deployment is the leakage-safe six-class WavLM-Large model. The
historical eight-class RAVDESS ensemble remains preserved as an immediate
rollback option. Local WavLM inference uses the optional PyTorch/Transformers
runtime and a cached backbone download.

## Supported application features

- WAV, MP3, M4A, MP4, OGG, WebM, AAC, WMA, and FLAC upload conversion
- Browser microphone recording and audio playback
- Real backend predictions and score breakdowns
- SQLite analysis history, filtering, details, and downloadable text reports
- React 19 and Vite 8 frontend
- FastAPI backend with generated OpenAPI documentation
- Preserved eight-class model and Phase 2 versioned experiment directories

## Phase 2 classification design

The primary Phase 2 experiment uses the six labels shared by RAVDESS and
CREMA-D: angry, disgust, fearful, happy, neutral, and sad. RAVDESS calm and
surprised are excluded rather than merged because they are distinct source
labels.

The validated inventory contains 8,498 six-class recordings from 115 speakers.
Seven exact duplicates are ineligible for modeling: one redundant same-label
copy and six recordings belonging to three conflicting-label duplicate pairs.
The resulting modeling manifest contains 8,491 recordings.

| Split | Model-eligible recordings | Speakers |
|---|---:|---:|
| Train | 5,932 | 80 |
| Validation | 1,241 | 17 |
| Final test (evaluated once after selection) | 1,318 | 18 |

All recordings from a speaker remain in one split. Previously examined RAVDESS
actors were prohibited from the final test split. The test speakers were used
once, after the model family, pooling method, and regularization were frozen.

## Selected Phase 2 model

Frozen `microsoft/wavlm-large` mean-pooled embeddings with class-balanced
Logistic Regression (`C=0.01`) were selected. The model reached 73.39%
validation macro F1 and 73.92% grouped cross-validation macro F1. After
selection, it was refit on train plus validation speakers and evaluated once on
the 18 untouched test speakers. Its fitted preprocessing and linear classifier
are exported as a portable NumPy artifact so local inference is independent of
the scikit-learn version used in Colab:

| Final-test metric | Result |
|---|---:|
| Accuracy | **77.54%** |
| Macro F1 | **77.78%** |
| Weighted F1 | **77.57%** |
| CREMA-D macro F1 | 76.89% |
| RAVDESS macro F1 | 82.33% |

Happy recall is 75.77%; Angry recall is 87.22%. These results exceed the 70%
aspirational target on the public speaker-independent holdout, but they do not
establish 90% accuracy or natural-conversation accuracy.

## Measured handcrafted baseline results

| Model | Grouped-CV accuracy | Grouped-CV macro F1 | Validation accuracy | Validation macro F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 47.12% | 46.74% | 47.22% | 45.95% |
| Calibrated RBF SVM | 47.66% | 47.25% | **48.51%** | **47.45%** |
| HistGradientBoosting | **49.09%** | **48.66%** | 48.27% | 47.21% |
| Extra Trees | 47.93% | 46.09% | 47.86% | 46.34% |

The calibrated RBF SVM is the strongest handcrafted baseline, but WavLM-Large
improves validation macro F1 by 25.94 percentage points. The handcrafted model
is retained for comparison and low-resource rollback experiments.

A training-only volume experiment retained 46.05% macro F1 at 3× gain, compared
with 31.80% for the unaugmented model, but reduced normal validation macro F1
from 47.45% to 46.73%. It is preserved as a robustness candidate and is not
deployed.

See [MODEL_EVALUATION.md](docs/MODEL_EVALUATION.md) for full interpretation.

## Project layout

```text
backend/                    FastAPI, FFmpeg conversion, SQLite, inference
frontend/                   React + Vite dashboard
dataset/                    Local RAVDESS and CREMA-D data (not tracked by Git)
experiments/phase2/         Validation, features, training, robustness, WavLM
models/final_model/         Historical eight-class deployment model
models/phase2/              Phase 2 caches and candidate artifacts
reports/phase2/             Reproducible manifests and measured reports
docs/                       Architecture, setup, evaluation, and demo guidance
```

## Windows setup and startup

The project expects Python 3.11 and Node.js. From PowerShell:

```powershell
cd "C:\Users\khush\Desktop\Speech-Emotion-Recognition"
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
cd frontend
npm install
cd ..
.\start.ps1
```

The frontend is served at `http://localhost:5173`; the API and Swagger UI are at
`http://127.0.0.1:8000` and `http://127.0.0.1:8000/docs`.

Detailed setup and troubleshooting are in [SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md).

## Reproducing Phase 2

```powershell
.\.venv\Scripts\python.exe -m pip install -r experiments\phase2\requirements.txt
.\.venv\Scripts\python.exe experiments\phase2\dataset_validation.py
.\.venv\Scripts\python.exe -m unittest experiments.phase2.test_dataset_validation -v
.\.venv\Scripts\python.exe experiments\phase2\extract_features.py --jobs 2
.\.venv\Scripts\python.exe experiments\phase2\train_baselines.py
.\.venv\Scripts\python.exe experiments\phase2\cross_dataset_evaluation.py
.\.venv\Scripts\python.exe experiments\phase2\robustness_tests.py
.\.venv\Scripts\python.exe experiments\phase2\extract_augmented_features.py --jobs 2
.\.venv\Scripts\python.exe experiments\phase2\train_volume_augmented.py
.\.venv\Scripts\python.exe experiments\phase2\robustness_tests.py `
  --selection models/phase2/volume_augmented/selection.json `
  --output reports/phase2/handcrafted_baselines/volume_augmented_gain_robustness.json
```

To reproduce WavLM extraction and training, open
[SpeechSense_WavLM_Colab.ipynb](experiments/phase2/SpeechSense_WavLM_Colab.ipynb)
in Google Colab, select a free GPU runtime, configure the single project path,
and run the cells. Embeddings are saved in resumable 64-recording shards. The
one-time finalization script is
`experiments/phase2/finalize_embedding_model.py` and refuses to reuse a report
that already contains final-test results.

Natural microphone or phone recordings can be evaluated without treating model
output as ground truth. Copy
`experiments/phase2/real_world_manifest.template.csv` to
`reports/phase2/real_world_manifest.csv`, fill in manually assigned labels and
device details, and run `experiments/phase2/evaluate_real_world.py`.

## Limitations

- Emotion labels describe acted or perceived vocal expression; they do not prove
  a person's internal emotional state.
- The historical rollback model was trained only on RAVDESS and generalizes
  poorly to natural speech.
- The Phase 2 public-dataset holdout exceeds 70%, but the small natural-recording
  pilot is not sufficient to quantify real-world accuracy.
- A preliminary one-speaker Happy benchmark produced 0/3 correct predictions
  from the deployed model. It documents a real failure but is too narrow to
  establish overall natural-speech accuracy.
- WavLM-Large needs substantially more CPU time and memory than the historical
  84-feature model. First startup also downloads the pretrained backbone.

SpeechSense should be presented as an experimental machine-learning system, not
as a clinical, legal, employment, or safety assessment tool.
