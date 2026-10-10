"use strict";

const video = document.getElementById("camera");
const overlay = document.getElementById("overlay");
const transcript = document.getElementById("transcript");
const speakButton = document.getElementById("speak-button");
const statusPill = document.querySelector(".status-pill");

const canvasContext = overlay.getContext("2d");

async function initialiseCamera() {
  if (!window.Hands || !window.Camera || !window.drawConnectors || !window.drawLandmarks) {
    transcript.textContent = "MediaPipe failed to load. Please refresh the page.";
    return;
  }

  try {
    video.addEventListener("loadedmetadata", () => {
      overlay.width = video.videoWidth || 640;
      overlay.height = video.videoHeight || 480;
    });

    const handDetector = new window.Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`,
    });

    handDetector.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.7,
    });

    handDetector.onResults((results) => {
      canvasContext.save();
      canvasContext.clearRect(0, 0, overlay.width, overlay.height);

      if (results.image) {
        canvasContext.drawImage(results.image, 0, 0, overlay.width, overlay.height);
      }

      if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        statusPill.textContent = "Tracking hands";

        results.multiHandLandmarks.forEach((landmarks) => {
          window.drawConnectors(canvasContext, landmarks, window.HAND_CONNECTIONS, {
            color: "#22c55e",
            lineWidth: 4,
          });
          window.drawLandmarks(canvasContext, landmarks, {
            color: "#a78bfa",
            lineWidth: 2,
          });
        });
      } else {
        statusPill.textContent = "Looking for hands";
        transcript.textContent = "Waiting for input...";
      }

      canvasContext.restore();
    });

    const camera = new window.Camera(video, {
      onFrame: async () => {
        await handDetector.send({ image: video });
      },
      width: 640,
      height: 480,
    });

    await camera.start();
    statusPill.textContent = "Tracking hands";
  } catch (error) {
    transcript.textContent = "Camera access was denied. Please allow webcam permissions.";
    statusPill.textContent = "Camera blocked";
    console.error(error);
  }
}

function speakTranslation() {
  const message = transcript.textContent.trim();
  if (!message || !("speechSynthesis" in window)) {
    return;
  }

  const utterance = new SpeechSynthesisUtterance(message);
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
}

speakButton.addEventListener("click", speakTranslation);

window.addEventListener("DOMContentLoaded", initialiseCamera);
