"use strict";

import { startHandTracking } from "./hand-tracking.js";
import { normalizeLandmarks } from "./landmarks.js";

const REQUIRED_SAMPLES = 100;
const video = document.getElementById("camera");
const overlay = document.getElementById("overlay");
const labelInput = document.getElementById("sign-label");
const recordButton = document.getElementById("record-button");
const saveButton = document.getElementById("save-button");
const progress = document.getElementById("progress");
const message = document.getElementById("collection-message");
const statusPill = document.querySelector(".status-pill");
const summaryList = document.getElementById("summary-list");

let isRecording = false;
let samples = [];
let sampleLabel = "";

function updateProgress() {
  progress.textContent = `${samples.length} / ${REQUIRED_SAMPLES}`;
}

function recordHandFrame(hands) {
  if (!isRecording || hands.length !== 1) {
    return;
  }

  samples.push(normalizeLandmarks(hands[0]));
  updateProgress();
  if (samples.length === REQUIRED_SAMPLES) {
    isRecording = false;
    saveButton.disabled = false;
    message.textContent = "Recording complete. Save the samples when ready.";
  }
}

async function startCamera() {
  try {
    await startHandTracking({
      video,
      overlay,
      onHands: recordHandFrame,
      onStatus: (status) => {
        statusPill.textContent = status;
      },
    });
  } catch (error) {
    statusPill.textContent = "Camera unavailable";
    message.textContent = error.message;
    console.error(error);
  }
}

async function refreshSummary() {
  const response = await fetch("/api/samples/summary");
  const counts = await response.json();
  if (!response.ok) {
    throw new Error(counts.error || "Could not load sample summary.");
  }

  summaryList.replaceChildren();
  const labels = Object.entries(counts);
  if (labels.length === 0) {
    summaryList.textContent = "No samples saved yet.";
    return;
  }
  labels.forEach(([label, count]) => {
    const item = document.createElement("li");
    item.textContent = `${label}: ${count} samples`;
    summaryList.append(item);
  });
}

function beginRecording() {
  const label = labelInput.value.trim();
  if (!/^[A-Za-z0-9_-]+$/.test(label)) {
    message.textContent = "Enter a label using only letters, numbers, dashes, or underscores.";
    return;
  }
  if (!video.srcObject) {
    message.textContent = "Wait for the camera to start before recording.";
    return;
  }

  samples = [];
  sampleLabel = label;
  isRecording = true;
  labelInput.disabled = true;
  recordButton.disabled = true;
  saveButton.disabled = true;
  updateProgress();
  message.textContent = "Recording: hold one hand in view.";
}

async function saveSamples() {
  saveButton.disabled = true;
  message.textContent = "Saving samples...";
  try {
    const response = await fetch("/api/samples", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label: sampleLabel, samples }),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "Could not save samples.");
    }
    message.textContent = `Saved ${samples.length} samples for "${result.label}". Total: ${result.count}.`;
    samples = [];
    sampleLabel = "";
    updateProgress();
    labelInput.disabled = false;
    recordButton.disabled = false;
    try {
      await refreshSummary();
    } catch (summaryError) {
      message.textContent += ` Sample summary could not be refreshed: ${summaryError.message}`;
    }
  } catch (error) {
    message.textContent = error.message;
    saveButton.disabled = false;
  }
}

recordButton.addEventListener("click", beginRecording);
saveButton.addEventListener("click", saveSamples);
refreshSummary().catch((error) => {
  message.textContent = error.message;
});
window.addEventListener("DOMContentLoaded", startCamera);
