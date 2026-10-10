"use strict";

/**
 * Commit a predicted word only after a stable sequence of confident results.
 */
export class PredictionStabilizer {
  /**
   * Configure the stability and repeat-commit thresholds.
   * @param {{requiredCount: number, minConfidence: number, duplicateWindowMs: number}} options Stabilization thresholds.
   */
  constructor({ requiredCount, minConfidence, duplicateWindowMs }) {
    this.requiredCount = requiredCount;
    this.minConfidence = minConfidence;
    this.duplicateWindowMs = duplicateWindowMs;
    this.currentLabel = null;
    this.currentCount = 0;
    this.lastCommittedLabel = null;
    this.lastCommittedAt = 0;
  }

  /**
   * Process one prediction and return a newly committed label, if any.
   * @param {string | null} label The predicted label, or null if no hand is present.
   * @param {number} confidence Prediction confidence from 0 to 1.
   * @param {number} [now] Current timestamp in milliseconds.
   * @returns {string | null} The label committed by this prediction, otherwise null.
   */
  update(label, confidence, now = Date.now()) {
    if (label === null) {
      this.reset();
      return null;
    }

    if (label === this.lastCommittedLabel) {
      if (now - this.lastCommittedAt < this.duplicateWindowMs) {
        this.currentLabel = null;
        this.currentCount = 0;
        return null;
      }
      this.lastCommittedLabel = null;
    }

    if (confidence < this.minConfidence) {
      this.currentLabel = null;
      this.currentCount = 0;
      return null;
    }

    if (label !== this.currentLabel) {
      this.currentLabel = label;
      this.currentCount = 1;
    } else {
      this.currentCount += 1;
    }

    if (this.currentCount < this.requiredCount) {
      return null;
    }

    this.lastCommittedLabel = label;
    this.lastCommittedAt = now;
    this.currentLabel = null;
    this.currentCount = 0;
    return label;
  }

  /** Reset the stability counter and repeat-commit lock after hand absence. */
  reset() {
    this.currentLabel = null;
    this.currentCount = 0;
    this.lastCommittedLabel = null;
    this.lastCommittedAt = 0;
  }
}
