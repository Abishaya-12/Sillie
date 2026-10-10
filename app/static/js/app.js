"use strict";

import { startHandTracking } from "./hand-tracking.js";

const video = document.getElementById("camera");
const overlay = document.getElementById("overlay");
const transcript = document.getElementById("transcript");
const speakButton = document.getElementById("speak-button");
const statusPill = document.querySelector(".status-pill");

async function initialiseCamera() {
  try {
    await startHandTracking({
      video,
      overlay,
      onHands: () => {},
      onStatus: (status) => {
        statusPill.textContent = status;
      },
    });
  } catch (error) {
    statusPill.textContent = "Camera unavailable";
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
