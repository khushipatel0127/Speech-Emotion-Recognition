# SpeechSense Windows Setup

## Requirements

- Windows 10 or 11
- Python 3.11, 64 bit
- Node.js with npm
- Approximately 4 GB free for dependencies and build artifacts
- Additional local space for RAVDESS and CREMA-D

The current project environment was created with Python 3.11.9. A virtual
environment copied from another machine is not portable; recreate it if its base
interpreter is missing.

## Backend environment

From PowerShell in the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

To reproduce the Phase 2 traditional-model experiments, also install their
locked scientific environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r experiments\phase2\requirements.txt
```

The WavLM packages are intentionally separate because they are large. The
validated classifier can run on CPU, while embedding extraction and training
remain better suited to a free Colab GPU.

To install the Phase 2 inference runtime:

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements-wavlm.txt
```

The first Phase 2 startup downloads `microsoft/wavlm-large` (about 1.26 GB) to
the Hugging Face cache. The repository configuration activates
`phase2_wavlm`; set `active_model` in `models/deployment.json` back to
`historical` for instant rollback. The fitted classifier itself is the portable
`portable_classifier.npz` artifact, so it does not depend on the exact
scikit-learn version used by Colab.

Verify model and audio imports:

```powershell
.\.venv\Scripts\python.exe -c "import fastapi, librosa, sklearn, soundfile; print('Backend dependencies OK')"
```

The backend uses `imageio-ffmpeg` as a bundled FFmpeg fallback. A system FFmpeg
installation is optional.

## Frontend environment

```powershell
cd frontend
npm install
npm run build
cd ..
```

## Start the application

```powershell
.\start.ps1
```

Manual startup is also supported:

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

In a second PowerShell window:

```powershell
cd frontend
npm run dev
```

Open `http://localhost:5173`. API documentation is available at
`http://127.0.0.1:8000/docs`.

## Dataset placement

RAVDESS actor folders must be located at `dataset/Actor_01` through
`dataset/Actor_24`. CREMA-D must contain `dataset/CREMA-D/wav/` and
`dataset/CREMA-D/metadata.csv`. Do not re-extract or rewrite source recordings.

Validate the dataset with:

```powershell
.\.venv\Scripts\python.exe experiments\phase2\dataset_validation.py
.\.venv\Scripts\python.exe -m unittest experiments.phase2.test_dataset_validation -v
```

## Common failures

- **Unable to create process:** the `.venv` points to a missing Python install.
  Restore Python 3.11 at the path in `.venv/pyvenv.cfg` or recreate the venv.
- **FFmpeg unavailable:** reinstall backend requirements so `imageio-ffmpeg` can
  provide its bundled executable.
- **Model file missing:** retain `models/final_model/` for rollback. Confirm that
  the Phase 2 paths in `models/deployment.json` exist before switching.
- **WavLM import/download error:** install `backend/requirements-wavlm.txt` and
  make sure the first startup has Internet access and at least 2 GB free.
- **Microphone unavailable:** grant microphone permission to the browser and use
  `localhost` or another secure browser context.
