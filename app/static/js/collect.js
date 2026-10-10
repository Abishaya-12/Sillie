"use strict";

import { startHandTracking } from "./hand-tracking.js";
import { normalizeLandmarks } from "./landmarks.js";
import { fetchJson } from "./api.js";

const REQUIRED_SAMPLES = 100;
const HEALTH_CHECK_INTERVAL_MS = 5000;
const video = document.getElementById("camera");
const overlay = document.getElementById("overlay");
const collectorName = document.getElementById("collector-name");
const labelInput = document.getElementById("sign-label");
const recordButton = document.getElementById("record-button");
const saveButton = document.getElementById("save-button");
const progress = document.getElementById("progress");
const message = document.getElementById("collection-message");
const statusPill = document.querySelector(".status-pill");
const summaryList = document.getElementById("summary-list");
const serverStatus = document.getElementById("server-status");

let isRecording = false;
let samples = [];
let sampleLabel = "";
let collectorNameForBatch = "";

/** Update the sample capture counter. */
function updateProgress() {
  progress.textContent = `${samples.length} / ${REQUIRED_SAMPLES}`;
}

/**
 * Store a normalized frame when exactly one hand is visible.
 * @param {{x: number, y: number, z: number}[][]} hands Detected hand landmarks.
 */
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

/** Start the shared camera and hand-tracking pipeline. */
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

/** Refresh the visible sample counts from the server. */
async function refreshSummary() {
  const counts = await fetchJson("/api/samples/summary");
  summaryList.replaceChildren();
  const labels = Object.entries(counts);
  if (labels.length === 0) {
    summaryList.textContent = "No samples saved yet.";
    return;
  }
  labels.forEach(([label, counts]) => {
    const item = document.createElement("li");
    item.textContent = `${label}: ${counts.samples} samples, ${counts.sessions} sessions`;
    summaryList.append(item);
  });
}

/** Validate the chosen label and begin capturing hand frames. */
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
  collectorNameForBatch = collectorName.value.trim();
  isRecording = true;
  collectorName.disabled = true;
  labelInput.disabled = true;
  recordButton.disabled = true;
  saveButton.disabled = true;
  updateProgress();
  message.textContent = "Recording: hold one hand in view.";
}

/** Submit the recorded sample batch and refresh the saved-count summary. */
async function saveSamples() {
  saveButton.disabled = true;
  message.textContent = "Saving samples...";
  try {
    const safeName = collectorNameForBatch.replace(/[^A-Za-z0-9_-]/g, "") || "anonymous";
    const session = `${safeName}-${Date.now()}`.replace(/[^A-Za-z0-9_-]/g, "");
    const result = await fetchJson("/api/samples", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label: sampleLabel, samples, session }),
    });
    message.textContent = `Saved ${samples.length} samples for "${result.label}". Total: ${result.samples} samples across ${result.sessions} sessions.`;
    samples = [];
    sampleLabel = "";
    collectorNameForBatch = "";
    updateProgress();
    collectorName.disabled = false;
    labelInput.disabled = false;
    recordButton.disabled = false;
    try {
      await refreshSummary();
    } catch (summaryError) {
      console.error(summaryError);
      message.textContent += ` Sample summary could not be refreshed: ${summaryError.message}`;
    }
  } catch (error) {
    console.error(error);
    message.textContent = error.message;
    saveButton.disabled = false;
  }
}

/** Update the server-status indicator using the health endpoint. */
async function refreshServerStatus() {
  try {
    await fetchJson("/api/health");
    serverStatus.textContent = "Server online";
    serverStatus.classList.add("online");
    serverStatus.classList.remove("offline");
  } catch (error) {
    console.error(error);
    serverStatus.textContent = "Server offline";
    serverStatus.classList.add("offline");
    serverStatus.classList.remove("online");
  }
}

recordButton.addEventListener("click", beginRecording);
saveButton.addEventListener("click", saveSamples);
refreshSummary().catch((error) => {
  console.error(error);
  message.textContent = error.message;
});
refreshServerStatus();
window.setInterval(refreshServerStatus, HEALTH_CHECK_INTERVAL_MS);
window.addEventListener("DOMContentLoaded", startCamera);
