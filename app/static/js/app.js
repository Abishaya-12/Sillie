"use strict";

import { fetchJson } from "./api.js";
import { FrameClock } from "./frame-clock.js";
import { startHandTracking } from "./hand-tracking.js";
import { flattenWindow } from "./landmarks.js";
import { PredictionStabilizer } from "./stabilizer.js";

const REQUIRED_STABLE_PREDICTIONS = 3;
const MIN_PREDICTION_CONFIDENCE = 0.8;
const DUPLICATE_COMMIT_WINDOW_MS = 1500;
const MIN_PREDICTION_INTERVAL_MS = 150;
const MODEL_STATUS_INTERVAL_MS = 10000;
const NO_MODEL_ERROR = "No trained model found. Run `python -m ml.train` first.";

const video = document.getElementById("camera");
const overlay = document.getElementById("overlay");
const transcript = document.getElementById("transcript");
const statusPill = document.querySelector(".status-pill");
const predictionStatus = document.getElementById("prediction-status");
const currentGuess = document.getElementById("current-guess");
const topGuesses = document.getElementById("top-guesses");
const confidenceTrack = document.getElementById("confidence-track");
const confidenceFill = document.getElementById("confidence-fill");
const clearButton = document.getElementById("clear-button");
const undoButton = document.getElementById("undo-button");
const autoSpeak = document.getElementById("auto-speak");
const speakButton = document.getElementById("speak-button");

let config = null;
let frameClock = null;
let frameBuffer = [];
const stabilizer = new PredictionStabilizer({
  requiredCount: REQUIRED_STABLE_PREDICTIONS,
  minConfidence: MIN_PREDICTION_CONFIDENCE,
  duplicateWindowMs: DUPLICATE_COMMIT_WINDOW_MS,
});
const committedWords = [];
let predictionInFlight = false;
let lastPredictionAt = Number.NEGATIVE_INFINITY;
let modelAvailable = false;
let modelIncompatible = false;
let handPresent = false;
let handPresenceGeneration = 0;

/**
 * Render committed words and keep the transcript controls synchronized.
 */
function renderTranscript() {
  transcript.textContent = committedWords.length
    ? committedWords.join(" ")
    : "Waiting for input...";
  undoButton.disabled = committedWords.length === 0;
}

/**
 * Update the current prediction text and accessible confidence bar.
 * @param {string | null} label The current label, or null to clear the guess.
 * @param {number} confidence Confidence value in the range 0 through 1.
 */
function renderPrediction(label, confidence) {
  if (!label) {
    currentGuess.textContent = "Waiting for a hand...";
    topGuesses.textContent = "";
    confidenceFill.style.width = "0%";
    confidenceFill.dataset.level = "low";
    confidenceTrack.setAttribute("aria-valuenow", "0");
    return;
  }

  const percentage = Math.round(confidence * 100);
  currentGuess.textContent = `${label} ${percentage}%`;
  confidenceFill.style.width = `${percentage}%`;
  confidenceFill.dataset.level = percentage < 50 ? "low" : percentage < 85 ? "mid" : "high";
  confidenceTrack.setAttribute("aria-valuenow", String(percentage));
}

/**
 * Render the top model alternatives under the confidence bar.
 * @param {{label: string, confidence: number}[]} predictions Ranked predictions.
 */
function renderTopGuesses(predictions) {
  topGuesses.textContent = predictions
    .map(({ label, confidence }) => `${label.toUpperCase()} ${Math.round(confidence * 100)}`)
    .join(" / ");
}

/**
 * Display buffer fill or the current model-ready state.
 */
function renderListeningStatus() {
  if (!config) {
    return;
  }
  if (modelIncompatible) {
    predictionStatus.textContent = "RECORD RAW DATA AND RETRAIN MODEL";
    return;
  }
  if (frameBuffer.length < config.sequence_length) {
    predictionStatus.textContent =
      `LISTENING ${frameBuffer.length}/${config.sequence_length}`;
    return;
  }
  predictionStatus.textContent = modelAvailable
    ? `LISTENING ${frameBuffer.length}/${config.sequence_length}`
    : "TRAIN A MODEL FIRST";
}

/**
 * Check model availability without requesting a prediction.
 */
async function refreshModelStatus() {
  try {
    const status = await fetchJson("/api/model/status");
    modelIncompatible = status.compatible === false;
    modelAvailable =
      !modelIncompatible && Array.isArray(status.labels) && status.labels.length > 0;
    renderListeningStatus();
  } catch (error) {
    console.error(error);
    predictionStatus.textContent = "CANNOT CHECK MODEL STATUS";
  }
}

/**
 * Speak one committed word when auto-speech is enabled.
 * @param {string} word The newly committed word.
 */
function speakWord(word) {
  if (!autoSpeak.checked || !("speechSynthesis" in window)) {
    return;
  }

  window.speechSynthesis.speak(new SpeechSynthesisUtterance(word));
}

/**
 * Record a stable prediction in the transcript and optionally speak it.
 * @param {string | null} label Predicted label, or null when the hand left.
 * @param {number} confidence Model confidence.
 */
function handleStablePrediction(label, confidence) {
  const committed = stabilizer.update(label, confidence);
  if (!committed) {
    return;
  }
  committedWords.push(committed);
  renderTranscript();
  speakWord(committed);
}

/**
 * Request a prediction when the interval and single-flight constraints allow.
 * @param {number[][]} frames One raw hand-and-pose window.
 */
async function requestPrediction(frames) {
  if (!modelAvailable || predictionInFlight) {
    return;
  }

  const now = performance.now();
  if (now - lastPredictionAt < MIN_PREDICTION_INTERVAL_MS) {
    return;
  }

  lastPredictionAt = now;
  predictionInFlight = true;
  const requestGeneration = handPresenceGeneration;
  try {
    const prediction = await fetchJson("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ window: flattenWindow(frames, config) }),
    });
    if (!handPresent || requestGeneration !== handPresenceGeneration) {
      return;
    }
    renderPrediction(prediction.label, prediction.confidence);
    renderTopGuesses(prediction.top || []);
    if (prediction.label.toLowerCase() === config.idle_label.toLowerCase()) {
      stabilizer.resetPending();
    } else {
      handleStablePrediction(prediction.label, prediction.confidence);
    }
  } catch (error) {
    stabilizer.reset();
    if (error.message.includes(NO_MODEL_ERROR)) {
      modelAvailable = false;
      renderListeningStatus();
      renderPrediction(null, 0);
    } else if (error.message.includes("incompatible feature format")) {
      modelAvailable = false;
      modelIncompatible = true;
      renderListeningStatus();
    } else {
      console.error(error);
      predictionStatus.textContent = error.message;
    }
  } finally {
    predictionInFlight = false;
  }
}

/**
 * Store one fixed-clock frame, reset on missing hands, and predict full windows.
 * @param {number[] | null} frame Fixed-rate raw landmark frame.
 */
function handleClockTick(frame) {
  if (frame === null) {
    if (handPresent) {
      handPresenceGeneration += 1;
    }
    handPresent = false;
    frameBuffer = [];
    stabilizer.reset();
    renderPrediction(null, 0);
    renderListeningStatus();
    return;
  }

  handPresent = true;
  frameBuffer.push(frame);
  if (frameBuffer.length > config.sequence_length) {
    frameBuffer.shift();
  }
  renderListeningStatus();
  if (
    frameBuffer.length === config.sequence_length
    && !predictionInFlight
    && performance.now() - lastPredictionAt >= MIN_PREDICTION_INTERVAL_MS
  ) {
    void requestPrediction(frameBuffer.slice());
  }
}

/**
 * Start the shared camera and MediaPipe hand-tracking pipeline.
 */
async function initialiseCamera() {
  try {
    config = await fetchJson("/api/config");
    frameClock = new FrameClock(config, handleClockTick);
    await refreshModelStatus();
    await startHandTracking({
      video,
      overlay,
      onStatus: (status) => {
        statusPill.textContent = status.toUpperCase();
      },
    });
    frameClock.start();
  } catch (error) {
    console.error(error);
    statusPill.textContent = "CAMERA UNAVAILABLE";
    predictionStatus.textContent = error.message.toUpperCase();
  }
}

/**
 * Speak the full committed transcript.
 */
function speakTranscript() {
  const sentence = committedWords.join(" ");
  if (!sentence || !("speechSynthesis" in window)) {
    return;
  }
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(new SpeechSynthesisUtterance(sentence));
}

clearButton.addEventListener("click", () => {
  committedWords.length = 0;
  renderTranscript();
});

undoButton.addEventListener("click", () => {
  committedWords.pop();
  renderTranscript();
});

speakButton.addEventListener("click", speakTranscript);
void refreshModelStatus();
window.setInterval(refreshModelStatus, MODEL_STATUS_INTERVAL_MS);
window.addEventListener("DOMContentLoaded", initialiseCamera);
