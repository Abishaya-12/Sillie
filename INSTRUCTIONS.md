Step 1: Pick one stack

Your deck shows two stacks (Flask + gTTS vs. WebRTC + Whisper + VITS). I'd go with this:

Frontend: plain HTML/JS or React, with WebRTC for the call
Hand tracking: MediaPipe Hands (it has a JS version, so recognition can run in the browser with no server lag)
Sign classifier: a small model on hand landmarks (TensorFlow.js, or Keras trained in Python)
Speech to text: the browser's Web Speech API first, Whisper later
Text to speech: the browser's speechSynthesis first, since gTTS adds latency
Backend: Flask or FastAPI, just for the WebRTC signalling serveR

Step 2: Build order
Repo skeleton, README, and .gitignore
Webcam page with MediaPipe landmarks drawn on screen
Data collection script (record about 100 samples per sign)
Train the classifier and test it offline
Live prediction with a smoothing buffer, then text-to-speech
Speech-to-captions panel
WebRTC call between two tabs or devices
Wire everything into the call, then polish the UI and write the demo script

Commit after each step so your history shows steady progress.

Project: SILLIE, a real-time sign language <-> speech translator for video calls.
Stack: Python 3.11, Flask, MediaPipe, OpenCV, TensorFlow/Keras; frontend in vanilla JS with WebRTC.
Rules: keep functions small, add type hints and docstrings, no hardcoded paths, put config in config.py, explain any new dependency.