"use strict";

import { fetchJson } from "./api.js";
import { FrameClock } from "./frame-clock.js";
import { flattenWindow } from "./landmarks.js";
import { startHandTracking } from "./hand-tracking.js";

const COLLECTION_DURATION_SECONDS = 3;
const COUNTDOWN_SECONDS = 3;
const WINDOW_STRIDE = 5;
const HEALTH_CHECK_INTERVAL_MS = 5000;

const video = document.getElementById("camera");
const overlay = document.getElementById("overlay");
const collectorName = document.getElementById("collector-name");
const labelInput = document.getElementById("sign-label");
const recordButton = document.getElementById("record-button");
const saveButton = document.getElementById("save-button");
const progress = document.getElementById("progress");
const collectionStats = document.getElementById("collection-stats");
const message = document.getElementById("collection-message");
const statusPill = document.querySelector(".status-pill");
const summaryList = document.getElementById("summary-list");
const serverStatus = document.getElementById("server-status");

let config = null;
let frameClock = null;
let isRecording = false;
let captureFrames = [];
let samples = [];
let sampleLabel = "";
let collectorNameForBatch = "";
let recordingCount = 0;

/** Update the capture tick counter using the configured fixed sample rate. */
function updateProgress() {
  const expectedTicks = config.sample_rate_hz * COLLECTION_DURATION_SECONDS;
  progress.textContent = `${captureFrames.length} / ${expectedTicks} TICKS`;
}

/** Update the accumulated recording and accepted-window counter. */
function updateCollectionStats() {
  collectionStats.textContent = `${recordingCount} RECORDINGS / ${samples.length} WINDOWS`;
}

/**
 * Process one fixed-clock frame while a recording is active.
 * @param {object | null} frame Normalized frame data or null when no single hand is visible.
 */
function handleClockTick(frame) {
  if (!isRecording) {
    return;
  }

  captureFrames.push(frame);
  updateProgress();
  const expectedTicks = config.sample_rate_hz * COLLECTION_DURATION_SECONDS;
  if (captureFrames.length >= expectedTicks) {
    finishRecording();
  }
}

/**
 * Convert a captured recording into overlapping valid sequence windows.
 * @param {(object | null)[]} recording The fixed-rate frame-clock sample list.
 * @returns {{windows: number[][], dropped: number}} Accepted and rejected window counts.
 */
function sliceRecording(recording) {
  const windows = [];
  let dropped = 0;
  for (
    let start = 0;
    start + config.sequence_length <= recording.length;
    start += WINDOW_STRIDE
  ) {
    const window = recording.slice(start, start + config.sequence_length);
    if (window.some((frame) => frame === null)) {
      dropped += 1;
      continue;
    }
    windows.push(flattenWindow(window, config));
  }
  return { windows, dropped };
}

/** Finish recording, retain valid windows, and update the UI. */
function finishRecording() {
  isRecording = false;
  recordingCount += 1;
  const { windows, dropped } = sliceRecording(captureFrames);
  samples.push(...windows);
  updateCollectionStats();
  recordButton.disabled = false;
  saveButton.disabled = samples.length === 0;
  message.textContent =
    `RECORDING ${recordingCount}: ${windows.length} WINDOWS KEPT, ` +
    `${dropped} DROPPED (HAND OR BODY LOST). REPEAT THE SIGN 2-3 TIMES DURING EACH RECORDING.`;
}

/**
 * Delay countdown stages by one second each.
 * @param {number} milliseconds Delay length.
 * @returns {Promise<void>} Resolves after the delay.
 */
function wait(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

/**
 * Start the countdown and collect one three-second sample-clock recording.
 */
async function beginRecording() {
  const label = labelInput.value.trim();
  if (!/^[A-Za-z0-9_-]+$/.test(label)) {
    message.textContent = "ENTER A LABEL USING ONLY LETTERS, NUMBERS, DASHES, OR UNDERSCORES.";
    return;
  }
  if (!frameClock) {
    message.textContent = "WAIT FOR THE CAMERA AND CONFIGURATION TO START.";
    return;
  }

  if (!sampleLabel) {
    sampleLabel = label;
    collectorNameForBatch = collectorName.value.trim();
    collectorName.disabled = true;
    labelInput.disabled = true;
  } else if (sampleLabel !== label) {
    message.textContent = "SAVE OR DISCARD THIS LABEL'S WINDOWS BEFORE CHANGING LABEL.";
    return;
  }

  recordButton.disabled = true;
  saveButton.disabled = true;
  captureFrames = [];
  for (let remaining = COUNTDOWN_SECONDS; remaining > 0; remaining -= 1) {
    message.textContent = `STARTING IN ${remaining}... REPEAT THE SIGN 2-3 TIMES DURING RECORDING.`;
    await wait(1000);
  }

  isRecording = true;
  message.textContent = "RECORDING... REPEAT THE SIGN 2-3 TIMES.";
  updateProgress();
}

/** Refresh the visible sample-window counts from the server. */
async function refreshSummary() {
  const counts = await fetchJson("/api/samples/summary");
  summaryList.replaceChildren();
  const labels = Object.entries(counts);
  if (labels.length === 0) {
    summaryList.textContent = "NO SAMPLES SAVED YET.";
    return;
  }
  labels.forEach(([label, labelCounts]) => {
    const item = document.createElement("li");
    item.textContent =
      `${label}: ${labelCounts.samples} WINDOWS / ${labelCounts.sessions} SESS`;
    summaryList.append(item);
  });
}

/** Save every valid window collected for this label as one session. */
async function saveSamples() {
  saveButton.disabled = true;
  message.textContent = "SAVING WINDOWS...";
  try {
    const safeName = collectorNameForBatch.replace(/[^A-Za-z0-9_-]/g, "") || "anonymous";
    const session = `${safeName}-${Date.now()}`.replace(/[^A-Za-z0-9_-]/g, "");
    const result = await fetchJson("/api/samples", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label: sampleLabel, samples, session }),
    });
    message.textContent =
      `SAVED ${samples.length} WINDOWS FOR "${result.label}". ` +
      `TOTAL: ${result.samples} WINDOWS ACROSS ${result.sessions} SESSIONS.`;
    samples = [];
    sampleLabel = "";
    collectorNameForBatch = "";
    recordingCount = 0;
    captureFrames = [];
    progress.textContent = "0 / 0 TICKS";
    updateCollectionStats();
    collectorName.disabled = false;
    labelInput.disabled = false;
    recordButton.disabled = false;
    try {
      await refreshSummary();
    } catch (summaryError) {
      console.error(summaryError);
      message.textContent += ` SUMMARY COULD NOT BE REFRESHED: ${summaryError.message}`;
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
    serverStatus.textContent = "SERVER ONLINE";
    serverStatus.classList.add("online");
    serverStatus.classList.remove("offline");
  } catch (error) {
    console.error(error);
    serverStatus.textContent = "SERVER OFFLINE";
    serverStatus.classList.add("offline");
    serverStatus.classList.remove("online");
  }
}

/** Load shared feature config, start tracking, and begin fixed-rate sampling. */
async function initialiseCollector() {
  try {
    config = await fetchJson("/api/config");
    progress.textContent = `0 / ${config.sample_rate_hz * COLLECTION_DURATION_SECONDS} TICKS`;
    frameClock = new FrameClock(config, handleClockTick);
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
    frameClock = null;
    recordButton.disabled = true;
    statusPill.textContent = "CAMERA UNAVAILABLE";
    message.textContent = error.message;
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
window.addEventListener("DOMContentLoaded", initialiseCollector);
