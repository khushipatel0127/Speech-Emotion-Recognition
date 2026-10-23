# SpeechSense Architecture

## Runtime flow

```mermaid
flowchart LR
    U[Browser upload or microphone] --> R[React and Vite frontend]
    R -->|multipart audio| A[FastAPI /api/predict]
    A --> V[Size, extension, duration, silence validation]
    V --> F[FFmpeg: 16 kHz mono PCM WAV]
    F --> P[Feature extraction and model inference]
    P --> D[(SQLite prediction history)]
    P --> R
    D --> H[History, detail, audio and report endpoints]
    H --> R
```

The frontend never creates predictions. Every displayed emotion score originates
from the backend model response and is stored with the corresponding prediction.

## Current deployment

`backend/predictor.py` loads the historical artifacts from `models/final_model/`
once per process. It extracts 84 values in a fixed order: 40 MFCC means, 40 MFCC
standard deviations, RMS mean and standard deviation, and zero-crossing-rate mean
and standard deviation. A calibrated SVM and HistGradientBoosting classifier are
combined with equal-weight probability averaging.

The historical classifier supports eight RAVDESS labels. It remains available as
the rollback model until a stronger six-class candidate completes validation,
final-test evaluation, and integration testing.

## Phase 2 experiment architecture

```mermaid
flowchart TD
    RD[RAVDESS: 24 speakers] --> M[Validated six-class manifest]
    CD[CREMA-D: 91 speakers] --> M
    M --> S[Speaker-independent frozen split]
    S --> T[Train: model selection folds]
    S --> V[Validation: candidate selection]
    S --> X[Final test: sealed until finalization]
    T --> HF[84 handcrafted features]
    T --> WE[Frozen WavLM embeddings]
    HF --> HC[CPU classifiers]
    WE --> EC[Lightweight classifiers]
    HC --> C[Validation comparison]
    EC --> C
    C --> X
    X --> VM[Versioned final model]
    VM --> API[FastAPI integration]
```

Dataset-prefixed speaker identifiers prevent collisions between corpora. Exact
PCM hashes detect duplicates. All learned preprocessing, SVM calibration, and
classifier fitting occur within training speakers; validation and test speakers
are excluded from those fit operations.

## Persistence

SQLite stores prediction IDs, filenames, emotion results, serialized score maps,
duration, input size, stored-audio reference, and timestamp. Phase 2 integration
will add model-version and processing-time fields through a backward-compatible
migration so historical predictions remain readable.

## Artifact safety

Datasets, virtual environments, uploaded audio, databases, feature caches,
embeddings, and binary model files are excluded from Git. Source, small reports,
manifests, configuration, and evaluation summaries are tracked. The original
eight-class artifacts are preserved under `models/final_model/`.
