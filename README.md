# Sillie

Sillie is a real-time sign language translation project designed for live video-call scenarios. It combines a browser-based webcam workflow, MediaPipe hand landmark detection, and a Python Flask backend for signaling and serving the interface.

## Stack

- Frontend: HTML, CSS, JavaScript, 1:1 WebRTC calls and browser speech synthesis
- Hand tracking: MediaPipe Hands
- Sign classification: TensorFlow/Keras model trained on extracted landmarks
- Speech-to-text: Web Speech API first, then Whisper if needed
- Backend: Flask for the app shell and WebRTC signaling
- Runtime: Python 3.11

## Why these dependencies

- Flask: lightweight web server for routing and serving the UI
- Flask-SQLAlchemy: persist account profiles in the configured SQL database
- Flask-Login: manage authenticated user sessions
- Flask-WTF: protect account and lobby forms with CSRF tokens
- Flask-SocketIO: authenticate users and relay WebRTC signaling over Socket.IO
- simple-websocket: provide WebSocket support with Flask-SocketIO's threading mode
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
export SILLIE_DEBUG=true
export SILLIE_SECRET_KEY=local-development-only-change-me
python server.py
```

On Windows PowerShell, set the same local-only values with
`$env:SILLIE_DEBUG = "true"` and
`$env:SILLIE_SECRET_KEY = "local-development-only-change-me"`.

Set a strong `SILLIE_SECRET_KEY` in the environment before running outside debug
mode. Set `SILLIE_DATABASE_URL` to a SQLAlchemy database URI to use a database
other than the default SQLite file in the Flask `instance/` directory. For
example, local PowerShell development can use:

```powershell
$env:SILLIE_DEBUG = "true"
$env:SILLIE_SECRET_KEY = "replace-with-a-local-development-secret"
python server.py
```

In production, set `SILLIE_SECRET_KEY` and `SILLIE_DATABASE_URL` in the hosting
provider's environment configuration. SQLite files on Render's free instances
are wiped on every deploy or restart; use persistent storage or an external
database if account data must survive. PostgreSQL deployments also need
`psycopg2-binary` installed in the server environment.

The app uses `db.create_all()` to create missing tables, but it does not add new
columns to an existing table. After updating an older local database for the
admin fields, delete `instance/sillie.sqlite3` once (or adopt Flask-Migrate
before preserving existing local data).

For model training and server inference, install the ML extras separately:

```bash
pip install -r requirements-ml.txt
```

The ML requirements include scikit-learn for session-aware splits and evaluation. Install pytest to run the feature tests:

```bash
pip install -r requirements-dev.txt
pytest
```

Training keeps sessions separate between train, validation, and test. It reports per-class metrics and the most common confusions.

```bash
python -m ml.train
```

### Importing video datasets

#### Fetching selected WLASL videos

Install the ML and Hugging Face dependencies with `pip install -r requirements-ml.txt`.
The fetcher uses `HF_TOKEN` when set, then a cached Hugging Face login, and otherwise
continues anonymously with lower rate limits. To save a login for later use, run
`hf auth login`.

```powershell
$env:HF_TOKEN = "your-hugging-face-token"
python -m ml.fetch_wlasl --words yes,hello,father,listen,bathroom --dry-run
python -m ml.fetch_wlasl --words yes,hello,father,listen,bathroom --max-per-word 12
```

Videos are written to `datasets/wlasl/<word>/`; the `datasets/` directory is ignored
by Git. WLASL is intended for **academic and computational use only**; commercial
use is not allowed under its Computational Use of Data Agreement. Credit the WLASL
authors, Dongxu Li and Hongdong Li, when using the dataset, and follow the dataset's
license and citation requirements.

Download sign videos manually and place them under
`datasets/<dataset-name>/<label>/` as `.mp4`, `.webm`, `.mov`, or `.avi` files.
You may add `datasets/<dataset-name>/signers.csv` with `file,signer` columns to
group clips by signer. Videos are not bundled with Sillie: verify each video's
license and the dataset's terms, and download/use them only as those terms allow.
The importer writes `ds-`-prefixed session CSV files into `data/<label>/`; those
sessions contribute to training only, never validation or test.

```bash
python -m ml.import_videos --source datasets/<dataset-name>
python -m ml.import_videos --source datasets/<dataset-name> --labels yes,hello --max-per-label 40 --dry-run
python -m ml.train
```

Collect at least two of your own sessions per label for validation and test.
When a label has fewer, training stops unless explicitly allowed with
`python -m ml.train --allow-ds-only`; that option prints a warning and may leave
evaluation incomplete.

### Administrator access

Grant or remove administrator access from the server command line only:

```bash
flask --app server make-admin <username>
flask --app server remove-admin <username>
```

Alternatively, set `SILLIE_ADMIN_USERNAMES` to comma-separated usernames in the
server environment; matching users are granted admin access when they log in.
Registration and browser/API requests cannot assign this role.

Running server-side predictions also requires TensorFlow and the other ML packages; install them with `pip install -r requirements-ml.txt` before training or requesting predictions.

Run the server with the same Python environment used to install the ML requirements and train the model. On Windows, from the repository root, use `.venv\Scripts\python.exe server.py`; starting it with a different Python can leave TensorFlow unavailable to prediction requests.

The app will be available at `http://127.0.0.1:5000`.

## Video calls

Open the same room from two signed-in browser sessions. For two devices on
different networks, use an HTTPS tunnel such as Cloudflare Tunnel or ngrok and
share its HTTPS URL. Browsers require HTTPS for camera and microphone access,
except on localhost. For two local browser windows, launch Chrome with
`--use-fake-device-for-media-stream` to use a simulated camera and microphone,
using separate Chrome profiles so each window can sign into a different account.
The call page loads the Socket.IO browser client required by Flask-SocketIO; no
other JavaScript packages are needed.

Calls use a public STUN server by default. For networks that require TURN,
configure all three optional environment variables:

- `SILLIE_TURN_URL` – for example, `turn:turn.example.com:3478`
- `SILLIE_TURN_USERNAME`
- `SILLIE_TURN_CREDENTIAL`

Run one Gunicorn worker with threaded handling for WebSocket connections:

```bash
gunicorn server:app --bind 0.0.0.0:$PORT --workers 1 --threads 100 --timeout 120
```

Exactly one worker is required because call rooms and signaling participants are
stored in memory in that process; additional workers would have separate room
state and could not signal each other. Do not configure eventlet or gevent.

For an offline demo, download the Press Start 2P and VT323 font files into `app/static/fonts/` and replace the Google Fonts link in each page with local `@font-face` declarations in `app/static/css/style.css`.

## Collecting data

Sign in, open **COLLECT SAMPLES** from the lobby, optionally enter your name, and enter a label using letters, numbers, dashes, or underscores. Sit far enough back that your full head and both shoulders remain visible, and keep a consistent distance from the camera. Each recording samples raw MediaPipe hand and pose coordinates at the configured 30 Hz, then creates overlapping one-second windows. Repeat each sign 2-3 times per recording; collect at least three separate recordings per sign from at least two people. Record an `idle` label too (hand resting or moving randomly). **Save samples** stores raw windows in `data/<label>/<session>.csv`.

This pipeline computes body-aware features from raw hand and pose windows in `ml/features.py`. Delete or move the old `data/` folder before collecting; old single-frame and 66-feature sequence rows are not compatible with this raw format. Training skips incompatible CSV files with a warning. Set `SILLIE_DATA_DIR` to use a different dataset directory.

Workflow: collect raw hand-and-pose windows for each label from the authenticated lobby, install the ML dependencies, run `python -m ml.train`, then start the app with the same Python environment (`.venv\Scripts\python.exe server.py` on Windows) and test live translation at `/translate`.

## Current status

The webcam hand-and-pose tracking page, raw sequence collection, body-aware feature extraction, and Conv1D-GRU training and prediction are in place. Authenticated users can also make 1:1 WebRTC calls; captions are currently limited to the in-call test data channel.
