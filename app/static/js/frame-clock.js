"use strict";

import { makeFrame } from "./landmarks.js";

/**
 * Sample the latest MediaPipe result on a fixed-rate clock.
 */
export class FrameClock {
  /**
   * Create the clock with its runtime sample rate and frame callback.
   * @param {number} sampleRateHz The sample rate supplied by /api/config.
   * @param {(frame: {shape: number[], wrist: [number, number, number], scale: number} | null) => void} onTick Called for every fixed-rate tick.
   */
  constructor(sampleRateHz, onTick) {
    this.intervalMs = 1000 / sampleRateHz;
    this.onTick = onTick;
    this.latestLandmarks = null;
    this.timerId = null;
  }

  /**
   * Update the most recent detected hand result.
   * @param {{x: number, y: number, z: number}[][]} hands Detected hand landmarks.
   */
  updateHands(hands) {
    this.latestLandmarks = hands.length === 1 ? hands[0] : null;
  }

  /** Begin emitting fixed-rate frame samples. */
  start() {
    if (this.timerId !== null) {
      return;
    }
    this.timerId = window.setInterval(() => {
      this.onTick(this.latestLandmarks ? makeFrame(this.latestLandmarks) : null);
    }, this.intervalMs);
  }

  /** Stop emitting samples and clear any in-progress timer. */
  stop() {
    if (this.timerId !== null) {
      window.clearInterval(this.timerId);
      this.timerId = null;
    }
  }
}
