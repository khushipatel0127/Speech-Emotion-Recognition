# SpeechSense — Full-Stack Speech Emotion Recognition Web Application

SpeechSense is a production-grade web application for Speech Emotion Recognition (SER) built with **React**, **Vite**, **FastAPI**, and **SQLite**, powered by an ensemble machine learning model trained on the RAVDESS dataset.

---

## 🌟 Key Features

1. **Multi-Format Audio & Video Uploads**:
   - Supports **WAV, MP3, M4A, MP4 (with audio), OGG, and WebM**.
   - Handles recordings exported from **WhatsApp**, iPhone Voice Memos, Android voice recorders, and video notes.
   - Converts all incoming audio streams into standard 16,000 Hz mono 16-bit PCM WAV using **FFmpeg** before ML feature extraction.
2. **In-Browser Microphone Recording**:
   - Record speech directly from the browser using the HTML5 `MediaRecorder` API.
   - Live recording timer, stop control, instant audio preview, and one-click submission.
3. **Overview Dashboard**: High-level project metrics, welcome banner with animated audio waveform illustration, architecture feature cards, and real-time recent analyses list connected to SQLite.
4. **Probability Breakdown**: Sorted multi-class probability scores across all 8 emotional categories (Angry, Calm, Disgust, Fearful, Happy, Neutral, Sad, Surprised) with emotion-specific color themes.
5. **Analysis History**: SQLite-backed persistent history with search by filename, filter by predicted emotion, confidence levels, audio duration, pagination, and secure bulk clear modal.
6. **Detailed Result View**: Complete inspection page with full probability breakdown, in-browser playback of stored converted WAV recordings, metadata summary (original filename, duration, file size, timestamp), and downloadable `.txt` analysis report.
7. **No Mock Data / No Fake Endpoints**: All predictions use the actual trained ensemble of Calibrated SVM and HistGradientBoosting via 50/50 soft voting.

---

## 🏗️ Architecture & Technology Stack

### Frontend
- **Framework**: React 19 + Vite 8
- **Routing**: React Router DOM 7
- **Icons**: Lucide React
- **Design System**: Dark navy fixed sidebar (`#0F172A`), royal blue primary accents (`#2563EB`), card-based layout, subtle elevation shadows, Plus Jakarta Sans typography.

### Backend
- **Framework**: FastAPI (Python 3.11)
- **Audio Preprocessing**: FFmpeg (via system PATH or bundled `imageio-ffmpeg` binary)
- **ML / Audio Libraries**: `scikit-learn`, `librosa`, `joblib`, `numpy`, `soundfile`
- **Database**: SQLite3 (`speechsense.db`)

### Machine Learning Ensemble
- **Model 1**: Calibrated Support Vector Classifier (`SVC(kernel="rbf", C=10)`) with `StandardScaler` and `SimpleImputer`.
- **Model 2**: HistGradientBoostingClassifier (`max_iter=200, learning_rate=0.05, max_leaf_nodes=15`) with `SimpleImputer`.
- **Combining Strategy**: 50/50 soft voting probability averaging.
- **Acoustic Features (84 total)**:
  - 40 MFCC means
  - 40 MFCC standard deviations
  - RMS energy mean and standard deviation
  - Zero-crossing rate (ZCR) mean and standard deviation
  - Sample rate: 16,000 Hz (mono)

---

## 🎥 FFmpeg Availability & Installation

SpeechSense automatically detects FFmpeg in two ways:
1. **Bundled Binary (Ready out of the box)**:
   The backend virtual environment includes `imageio-ffmpeg` which bundles a standalone, official 64-bit Windows FFmpeg v7.1 binary. **No manual installation is required.**
2. **Optional System-wide FFmpeg**:
   If you wish to install FFmpeg system-wide in Windows PowerShell, run:
   ```powershell
   winget install Gyan.FFmpeg
   ```
   or via Chocolatey:
   ```powershell
   choco install ffmpeg
   ```

---

## 🚀 How to Run the Application

### Option A: One-Click Launcher (Recommended)
From PowerShell in the project directory, run:
```powershell
cd "C:\Users\khush\Desktop\Speech-Emotion-Recognition"
.\start.ps1
```
*This starts both the FastAPI backend and the Vite frontend in separate terminal windows and automatically opens `http://localhost:5173` in your browser.*

---

### Option B: Run Servers Manually

#### 1. Start the FastAPI Backend
Open a PowerShell terminal:
```powershell
cd "C:\Users\khush\Desktop\Speech-Emotion-Recognition"
.\.venv\Scripts\python.exe -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```
- API Base: `http://localhost:8000`
- Swagger UI Documentation: `http://localhost:8000/docs`

#### 2. Start the React Frontend
Open a second PowerShell terminal:
```powershell
cd "C:\Users\khush\Desktop\Speech-Emotion-Recognition\frontend"
npm run dev
```
- Open `http://localhost:5173` in your browser.

---

## 🧪 Verification & Testing

All verification tests are pre-configured and can be run anytime:

```powershell
# 1. Run core database and prediction unit tests
.\.venv\Scripts\python.exe backend\test_backend.py

# 2. Run multi-format audio conversion tests (WAV, MP3, M4A, MP4, WebM, Corrupt validation)
.\.venv\Scripts\python.exe backend\test_multiformat.py

# 3. Run FastAPI endpoint integration tests
.\.venv\Scripts\python.exe backend\test_api.py

# 4. Verify frontend production build
cd frontend; npm run build
```
