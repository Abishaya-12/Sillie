# Sillie

Sillie is a real-time sign language translation project designed for live video-call scenarios. It combines a browser-based webcam workflow, MediaPipe hand landmark detection, and a Python Flask backend for signaling and serving the interface.

## Stack

- Frontend: HTML, CSS, JavaScript, WebRTC-ready UI, browser speech synthesis
- Hand tracking: MediaPipe Hands
- Sign classification: TensorFlow/Keras model trained on extracted landmarks
- Speech-to-text: Web Speech API first, then Whisper if needed
- Backend: Flask for the app shell and WebRTC signaling
- Runtime: Python 3.11

## Why these dependencies

- Flask: lightweight web server for routing and serving the UI
- OpenCV: camera input and preprocessing utilities
- MediaPipe: robust hand landmark detection in-browser or in Python preprocessing
- TensorFlow/Keras: small model training and inference for sign classification
- NumPy: efficient landmark-array processing

## Project structure

- `app/` – Flask application package with routes, templates, and static assets
- `config.py` – central configuration object with environment overrides
- `server.py` – local entrypoint to run the app
- `requirements.txt` – Python dependency list

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python server.py
```

For model training or offline landmark processing, install the ML extras separately:

```bash
pip install -r requirements-ml.txt
```

The ML requirements include scikit-learn for session-aware splits and confusion-matrix evaluation. Training keeps sessions separate between train, validation, and test where each label has at least three sessions. Otherwise, it warns and falls back to a stratified window-level split, whose accuracy may be optimistic.

```bash
python -m ml.train
```

Running server-side predictions also requires TensorFlow and the other ML packages; install them with `pip install -r requirements-ml.txt` before training or requesting predictions.

Run the server with the same Python environment used to install the ML requirements and train the model. On Windows, from the repository root, use `.venv\Scripts\python.exe server.py`; starting it with a different Python can leave TensorFlow unavailable to prediction requests.

The app will be available at `http://127.0.0.1:5000`.

For an offline demo, download the Press Start 2P and VT323 font files into `app/static/fonts/` and replace the Google Fonts link in each page with local `@font-face` declarations in `app/static/css/style.css`.

## Collecting data

Open `http://127.0.0.1:5000/collect`, optionally enter your name, and enter a label using letters, numbers, dashes, or underscores. Record each sign as a short motion: the page counts down, samples at the configured 30 Hz, then captures a three-second recording and saves valid overlapping sequence windows. Repeat each sign 2-3 times per recording; collect at least three separate recordings per label and include data from at least two people. Record an `idle` label too (hand resting or moving randomly) so the model has a background class. **Save samples** stores the collected windows in `data/<label>/<session>.csv`.

The model now expects 30 frames × 66 features per frame. Old single-frame data and files with the previous format are skipped by training; delete old 63-value CSVs (including old `data/<label>/<session>.csv` files) or re-record them with the sequence collector before training. Set `SILLIE_DATA_DIR` to use a different dataset directory.

Workflow: collect several sessions for each label at `/collect`, install the ML dependencies, run `python -m ml.train`, then start the app with `python server.py` and test live translation at `/`.

## Current status

The webcam hand-tracking page, sequence data collection, GRU training, and live sequence prediction are in place.
