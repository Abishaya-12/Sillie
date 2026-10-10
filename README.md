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

The ML requirements include scikit-learn for session-aware splits and confusion-matrix evaluation. Training keeps sessions separate between train, validation, and test where each label has at least three sessions. Otherwise, it warns and falls back to a stratified sample-level split, whose accuracy may be optimistic. To train the sign classifier after collecting samples, run:

```bash
python -m ml.train
```

Running server-side predictions also requires TensorFlow and the other ML packages; install them with `pip install -r requirements-ml.txt` before training or requesting predictions.

The app will be available at `http://127.0.0.1:5000`.

For an offline demo, download the Press Start 2P and VT323 font files into `app/static/fonts/` and replace the Google Fonts link in each page with local `@font-face` declarations in `app/static/css/style.css`.

## Collecting data

Open `http://127.0.0.1:5000/collect`, optionally enter your name, enter a label using letters, numbers, dashes, or underscores, then select **Record**. The browser records 100 frames where exactly one hand is detected. **Save samples** assigns that batch a unique session and stores it in `data/<label>/<session>.csv`; the collection page shows total sample and session counts. Existing flat `data/<label>.csv` files remain included and count as one `legacy` session. Set `SILLIE_DATA_DIR` to use a different storage directory. For meaningful session-isolated evaluation, collect at least three sessions for every label.

## Current status

The webcam hand-tracking page and browser-based sample collection are in place. The next milestone is training and testing a sign classifier on the collected landmarks.
