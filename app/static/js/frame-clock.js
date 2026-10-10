"use strict";

import { getLatest } from "./hand-tracking.js";

const MIN_SHOULDER_VISIBILITY = 0.5;
const MAX_RESULT_AGE_MS = 200;

/**
 * Check that both shoulder landmarks have usable visibility scores.
 * @param {object[]} pose Pose landmarks.
 * @returns {boolean} True when both shoulders meet the configured threshold.
 */
function hasVisibleShoulders(pose) {
  return [pose[11], pose[12]].every(
    (shoulder) =>
      Number.isFinite(shoulder.visibility)
      && shoulder.visibility >= MIN_SHOULDER_VISIBILITY,
  );
}

/**
 * Sample fresh hand and pose landmarks on a fixed-rate timer.
 */
export class FrameClock {
  /**
   * Create the clock using the runtime config returned by /api/config.
   * @param {{sample_rate_hz: number, raw_frame_size: number}} settings Runtime sampling settings.
   * @param {(frame: number[] | null) => void} onTick Callback for every timer tick.
   */
  constructor(settings, onTick) {
    this.intervalMs = 1000 / settings.sample_rate_hz;
    this.rawFrameSize = settings.raw_frame_size;
    this.onTick = onTick;
    this.timerId = null;
  }

  /** Begin emitting fixed-rate raw-frame samples. */
  start() {
    if (this.timerId !== null) {
      return;
    }
    this.timerId = window.setInterval(() => {
      this.onTick(this.buildRawFrame(getLatest()));
    }, this.intervalMs);
  }

  /** Stop emitting samples and clear any in-progress timer. */
  stop() {
    if (this.timerId !== null) {
      window.clearInterval(this.timerId);
      this.timerId = null;
    }
  }

  /**
   * Return a raw frame only when hand and body landmarks are current and usable.
   * @param {{hand: object[] | null, pose: object[] | null, handAgeMs: number, poseAgeMs: number}} latest Current detector results.
   * @returns {number[] | null} Flattened hand xyz + pose xy coordinates.
   */
  buildRawFrame(latest) {
    if (
      !latest.hand
      || latest.hand.length !== 21
      || !latest.pose
      || latest.pose.length !== 33
      || latest.handAgeMs >= MAX_RESULT_AGE_MS
      || latest.poseAgeMs >= MAX_RESULT_AGE_MS
      || !hasVisibleShoulders(latest.pose)
    ) {
      return null;
    }

    const rawFrame = [
      ...latest.hand.flatMap(({ x, y, z }) => [x, y, z]),
      ...latest.pose.flatMap(({ x, y }) => [x, y]),
    ];
    if (
      rawFrame.length !== this.rawFrameSize
      || rawFrame.some((coordinate) => !Number.isFinite(coordinate))
    ) {
      return null;
    }
    return rawFrame;
  }
}
