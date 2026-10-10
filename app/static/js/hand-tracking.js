"use strict";

const HANDS_VERSION = "0.4.1675469240";
const POSE_VERSION = "0.5.1675469404";
const HAND_CONNECTION_COLOR = "#73eff7";
const LANDMARK_COLOR = "#ffcd75";
const POSE_ANCHOR_COLOR = "#94b0c2";
const LANDMARK_PIXEL_SIZE = 6;
const POSE_ANCHOR_INDICES = [0, 7, 8, 11, 12];
const REQUIRED_SHOULDER_VISIBILITY = 0.5;

let latestHand = null;
let latestPose = null;
let anyHandDetected = false;
let handTimestamp = Number.NEGATIVE_INFINITY;
let poseTimestamp = Number.NEGATIVE_INFINITY;

/**
 * Return latest detector results with ages measured against the performance clock.
 * @returns {{hand: object[] | null, pose: object[] | null, handAgeMs: number, poseAgeMs: number}} Current landmarks and age in milliseconds.
 */
export function getLatest() {
  const now = performance.now();
  return {
    hand: latestHand,
    pose: latestPose,
    handAgeMs: now - handTimestamp,
    poseAgeMs: now - poseTimestamp,
  };
}

/**
 * Draw detected hand landmarks and body pose anchors on the mirrored camera overlay.
 * @param {CanvasRenderingContext2D} context Canvas drawing context.
 * @param {HTMLCanvasElement} overlay Landmark canvas.
 * @param {HTMLVideoElement} video Camera video.
 * @param {object[] | null} hand One hand's landmarks, when present.
 * @param {object[] | null} pose Body pose landmarks, when present.
 */
function drawOverlay(context, overlay, video, hand, pose) {
  const width = video.videoWidth || 640;
  const height = video.videoHeight || 480;
  if (overlay.width !== width || overlay.height !== height) {
    overlay.width = width;
    overlay.height = height;
  }
  context.clearRect(0, 0, overlay.width, overlay.height);
  if (hand && window.drawConnectors) {
    context.lineCap = "square";
    window.drawConnectors(context, hand, window.HAND_CONNECTIONS, {
      color: HAND_CONNECTION_COLOR,
      lineWidth: 4,
      lineCap: "square",
    });
    context.fillStyle = LANDMARK_COLOR;
    hand.forEach((landmark) => {
      const left = Math.round(landmark.x * overlay.width - LANDMARK_PIXEL_SIZE / 2);
      const top = Math.round(landmark.y * overlay.height - LANDMARK_PIXEL_SIZE / 2);
      context.fillRect(left, top, LANDMARK_PIXEL_SIZE, LANDMARK_PIXEL_SIZE);
    });
  }
  if (pose) {
    context.fillStyle = POSE_ANCHOR_COLOR;
    POSE_ANCHOR_INDICES.forEach((index) => {
      const landmark = pose[index];
      const left = Math.round(landmark.x * overlay.width - LANDMARK_PIXEL_SIZE / 2);
      const top = Math.round(landmark.y * overlay.height - LANDMARK_PIXEL_SIZE / 2);
      context.fillRect(left, top, LANDMARK_PIXEL_SIZE, LANDMARK_PIXEL_SIZE);
    });
  }
}

/**
 * Report whether the hand and shoulders are sufficiently visible.
 * @param {(status: string) => void} onStatus Status callback.
 */
function updateTrackingStatus(onStatus) {
  if (anyHandDetected) {
    if (
      !latestPose
      || ![latestPose[11], latestPose[12]].every(
        (shoulder) =>
          Number.isFinite(shoulder?.visibility)
          && shoulder.visibility >= REQUIRED_SHOULDER_VISIBILITY,
      )
    ) {
      onStatus("STEP BACK: SHOW HEAD + SHOULDERS");
      return;
    }
    onStatus("Tracking hands");
    return;
  }
  onStatus("Looking for hands");
}

/**
 * Start MediaPipe Hands and Pose processing on the same camera frames.
 * @param {{video: HTMLVideoElement, overlay: HTMLCanvasElement, onStatus: (status: string) => void}} options Camera elements and status callback.
 * @returns {Promise<{stop: () => void}>} Resolves once camera processing starts.
 */
export async function startHandTracking({ video, overlay, onStatus }) {
  if (!window.Hands || !window.Pose || !window.Camera || !window.drawConnectors) {
    throw new Error("MediaPipe failed to load. Please refresh the page.");
  }
  const context = overlay.getContext("2d");
  if (!context) {
    throw new Error("Could not create the camera overlay.");
  }

  latestHand = null;
  latestPose = null;
  anyHandDetected = false;
  handTimestamp = Number.NEGATIVE_INFINITY;
  poseTimestamp = Number.NEGATIVE_INFINITY;

  const hands = new window.Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@${HANDS_VERSION}/${file}`,
  });
  hands.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7,
  });
  hands.onResults((results) => {
    const detectedHands = results.multiHandLandmarks || [];
    anyHandDetected = detectedHands.length > 0;
    latestHand = detectedHands.length === 1 ? detectedHands[0] : null;
    handTimestamp = performance.now();
    drawOverlay(context, overlay, video, latestHand, latestPose);
    updateTrackingStatus(onStatus);
  });

  const pose = new window.Pose({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${POSE_VERSION}/${file}`,
  });
  pose.setOptions({
    modelComplexity: 0,
    smoothLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  pose.onResults((results) => {
    latestPose = results.poseLandmarks || null;
    poseTimestamp = performance.now();
    drawOverlay(context, overlay, video, latestHand, latestPose);
    updateTrackingStatus(onStatus);
  });

  const camera = new window.Camera(video, {
    onFrame: async () => {
      await hands.send({ image: video });
      await pose.send({ image: video });
    },
    width: 640,
    height: 480,
  });
  await camera.start();
  return { stop: () => camera.stop() };
}
