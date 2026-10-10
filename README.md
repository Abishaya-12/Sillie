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

The app will be available at `http://127.0.0.1:5000`.

## Current status

This repository is in the initial scaffold phase. The next milestone is to build the webcam page, add MediaPipe landmark detection, and collect a small sign dataset for training.
