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

The WavLM packages are intentionally separate in
`experiments/phase2/requirements-wavlm.txt` because they are large and the
recommended execution target is a free Colab GPU runtime.

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
- **Model file missing:** retain `models/final_model/` for the current app. Phase
  2 candidates are separate and are not automatic replacements.
- **Microphone unavailable:** grant microphone permission to the browser and use
  `localhost` or another secure browser context.
