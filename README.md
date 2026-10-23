# SpeechSense — Speech Emotion Recognition

SpeechSense is a React and FastAPI application for analyzing emotion-related
acoustic patterns in speech. It accepts uploaded audio or browser microphone
recordings, converts them to 16 kHz mono WAV with FFmpeg, returns model scores,
and stores prediction history in SQLite.

The application currently deploys the historical eight-class RAVDESS ensemble.
A leakage-safe six-class Phase 2 pipeline is implemented and evaluated, but its
handcrafted candidate is not deployed because validation and robustness results
remain insufficient. Frozen WavLM evaluation is prepared for a free Colab GPU.

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
| Reserved final test | 1,318 | 18 |

All recordings from a speaker remain in one split. Previously examined RAVDESS
actors are prohibited from the new final test split. The reserved final test has
not been evaluated during current model development.

## Measured handcrafted baseline results

| Model | Grouped-CV accuracy | Grouped-CV macro F1 | Validation accuracy | Validation macro F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 47.12% | 46.74% | 47.22% | 45.95% |
| Calibrated RBF SVM | 47.66% | 47.25% | **48.51%** | **47.45%** |
| HistGradientBoosting | **49.09%** | **48.66%** | 48.27% | 47.21% |
| Extra Trees | 47.93% | 46.09% | 47.86% | 46.34% |

The calibrated RBF SVM is the current validation winner by a narrow margin. It
is still unsuitable for deployment: validation macro F1 falls to 31.80% at 3×
gain, and cross-dataset macro F1 is only 24.74% for CREMA-D → RAVDESS and 18.07%
for RAVDESS → CREMA-D. These measured failures are why Phase 2 has not replaced
the historical website model.

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
.\.venv\Scripts\python.exe experiments\phase2\dataset_validation.py
.\.venv\Scripts\python.exe -m unittest experiments.phase2.test_dataset_validation -v
.\.venv\Scripts\python.exe experiments\phase2\extract_features.py --jobs 2
.\.venv\Scripts\python.exe experiments\phase2\train_baselines.py
.\.venv\Scripts\python.exe experiments\phase2\cross_dataset_evaluation.py
.\.venv\Scripts\python.exe experiments\phase2\robustness_tests.py
```

For WavLM, open
[SpeechSense_WavLM_Colab.ipynb](experiments/phase2/SpeechSense_WavLM_Colab.ipynb)
in Google Colab, select a free GPU runtime, configure the single project path,
and run the cells. Embeddings are saved in resumable 64-recording shards.

## Limitations

- Emotion labels describe acted or perceived vocal expression; they do not prove
  a person's internal emotional state.
- The current website model was trained only on RAVDESS and generalizes poorly
  to natural speech.
- Phase 2 handcrafted models remain below the 70% aspirational target.
- No manually labeled natural-microphone benchmark has been supplied, so
  real-world accuracy has not been established.
- WavLM validation and the one-time Phase 2 final-test evaluation remain pending.

SpeechSense should be presented as an experimental machine-learning system, not
as a clinical, legal, employment, or safety assessment tool.
