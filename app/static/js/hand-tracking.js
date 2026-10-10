"use strict";

const HANDS_VERSION = "0.4.1675469240";
const HAND_CONNECTION_COLOR = "#73eff7";
const LANDMARK_COLOR = "#ffcd75";
const LANDMARK_PIXEL_SIZE = 6;

/**
 * Start the shared MediaPipe hand detector and camera.
 * @param {{
 *   video: HTMLVideoElement,
 *   overlay: HTMLCanvasElement,
 *   onHands: (hands: {x: number, y: number, z: number}[][]) => void,
 *   onStatus: (status: string) => void
 * }} options Camera elements and callbacks for hand results and status.
 * @returns {Promise<void>} Resolves after the camera starts.
 */
export async function startHandTracking({ video, overlay, onHands, onStatus }) {
  if (!window.Hands || !window.Camera || !window.drawConnectors) {
    throw new Error("MediaPipe failed to load. Please refresh the page.");
  }

  const context = overlay.getContext("2d");
  if (!context) {
    throw new Error("Could not create the camera overlay.");
  }

  const detector = new window.Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@${HANDS_VERSION}/${file}`,
  });
  detector.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7,
  });

  detector.onResults((results) => {
    const hands = results.multiHandLandmarks || [];
    const width = video.videoWidth || 640;
    const height = video.videoHeight || 480;
    if (overlay.width !== width || overlay.height !== height) {
      overlay.width = width;
      overlay.height = height;
    }

    context.save();
    context.clearRect(0, 0, overlay.width, overlay.height);
    if (results.image) {
      context.drawImage(results.image, 0, 0, overlay.width, overlay.height);
    }
    hands.forEach((landmarks) => {
      context.lineCap = "square";
      window.drawConnectors(context, landmarks, window.HAND_CONNECTIONS, {
        color: HAND_CONNECTION_COLOR,
        lineWidth: 4,
        lineCap: "square",
      });
      context.fillStyle = LANDMARK_COLOR;
      landmarks.forEach((landmark) => {
        const left = Math.round(landmark.x * overlay.width - LANDMARK_PIXEL_SIZE / 2);
        const top = Math.round(landmark.y * overlay.height - LANDMARK_PIXEL_SIZE / 2);
        context.fillRect(left, top, LANDMARK_PIXEL_SIZE, LANDMARK_PIXEL_SIZE);
      });
    });
    context.restore();

    onStatus(hands.length > 0 ? "Tracking hands" : "Looking for hands");
    onHands(hands);
  });

  const camera = new window.Camera(video, {
    onFrame: async () => {
      await detector.send({ image: video });
    },
    width: 640,
    height: 480,
  });
  await camera.start();
}
