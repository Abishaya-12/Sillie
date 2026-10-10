"use strict";

import { fetchJson } from "./api.js";
import { startHandTracking } from "./hand-tracking.js";
import { normalizeLandmarks } from "./landmarks.js";
import { PredictionStabilizer } from "./stabilizer.js";

const REQUIRED_STABLE_PREDICTIONS = 8;
const MIN_PREDICTION_CONFIDENCE = 0.85;
const DUPLICATE_COMMIT_WINDOW_MS = 1500;
const MIN_PREDICTION_INTERVAL_MS = 100;
const MODEL_STATUS_INTERVAL_MS = 10000;
const REQUIRED_HAND_COUNT = 1;
const NO_MODEL_ERROR = "No trained model found. Run `python -m ml.train` first.";

const video = document.getElementById("camera");
const overlay = document.getElementById("overlay");
const transcript = document.getElementById("transcript");
const statusPill = document.querySelector(".status-pill");
const predictionStatus = document.getElementById("prediction-status");
const currentGuess = document.getElementById("current-guess");
const confidenceTrack = document.getElementById("confidence-track");
const confidenceFill = document.getElementById("confidence-fill");
const clearButton = document.getElementById("clear-button");
const undoButton = document.getElementById("undo-button");
const autoSpeak = document.getElementById("auto-speak");
const speakButton = document.getElementById("speak-button");

const stabilizer = new PredictionStabilizer({
  requiredCount: REQUIRED_STABLE_PREDICTIONS,
  minConfidence: MIN_PREDICTION_CONFIDENCE,
  duplicateWindowMs: DUPLICATE_COMMIT_WINDOW_MS,
});
const committedWords = [];
let predictionInFlight = false;
let lastPredictionAt = Number.NEGATIVE_INFINITY;
let modelAvailable = false;
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
    confidenceFill.style.width = "0%";
    confidenceTrack.setAttribute("aria-valuenow", "0");
    return;
  }

  const percentage = Math.round(confidence * 100);
  currentGuess.textContent = `${label} ${percentage}%`;
  confidenceFill.style.width = `${percentage}%`;
  confidenceTrack.setAttribute("aria-valuenow", String(percentage));
}

/**
 * Check model availability without requesting a prediction.
 */
async function refreshModelStatus() {
  try {
    const status = await fetchJson("/api/model/status");
    modelAvailable = Array.isArray(status.labels) && status.labels.length > 0;
    predictionStatus.textContent = modelAvailable
      ? "Model ready"
      : "Train a model first";
  } catch (error) {
    console.error(error);
    predictionStatus.textContent = "Cannot check model status";
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
 * @param {{x: number, y: number, z: number}[]} landmarks One hand's 21 landmarks.
 */
async function requestPrediction(landmarks) {
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
      body: JSON.stringify({ features: normalizeLandmarks(landmarks) }),
    });
    if (!handPresent || requestGeneration !== handPresenceGeneration) {
      return;
    }
    renderPrediction(prediction.label, prediction.confidence);
    handleStablePrediction(prediction.label, prediction.confidence);
  } catch (error) {
    stabilizer.reset();
    if (error.message.includes(NO_MODEL_ERROR)) {
      modelAvailable = false;
      predictionStatus.textContent = "Train a model first";
      renderPrediction(null, 0);
    } else {
      console.error(error);
      predictionStatus.textContent = error.message;
    }
  } finally {
    predictionInFlight = false;
  }
}

/**
 * Update tracking behavior and request predictions from hand-tracking results.
 * @param {{x: number, y: number, z: number}[][]} hands Detected hand landmarks.
 */
function handleHands(hands) {
  if (hands.length !== REQUIRED_HAND_COUNT) {
    if (handPresent) {
      handPresenceGeneration += 1;
    }
    handPresent = false;
    stabilizer.reset();
    renderPrediction(null, 0);
    return;
  }
  handPresent = true;
  void requestPrediction(hands[0]);
}

/**
 * Start the shared camera and MediaPipe hand-tracking pipeline.
 */
async function initialiseCamera() {
  try {
    await startHandTracking({
      video,
      overlay,
      onHands: handleHands,
      onStatus: (status) => {
        statusPill.textContent = status;
      },
    });
  } catch (error) {
    console.error(error);
    statusPill.textContent = "Camera unavailable";
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
