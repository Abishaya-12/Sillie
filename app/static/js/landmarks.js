/**
 * Convert MediaPipe hand landmarks into wrist-relative, scale-normalized features.
 * @param {{x: number, y: number, z: number}[]} landmarks The 21 hand landmarks.
 * @returns {number[]} A flat array containing 63 normalized coordinates.
 */
export function normalizeLandmarks(landmarks) {
  const wrist = landmarks[0];
  const middleFingerMcp = landmarks[9];
  const distance = Math.hypot(
    middleFingerMcp.x - wrist.x,
    middleFingerMcp.y - wrist.y,
    middleFingerMcp.z - wrist.z,
  );
  const scale = distance === 0 ? 1 : distance;

  return landmarks.flatMap((point) => [
    (point.x - wrist.x) / scale,
    (point.y - wrist.y) / scale,
    (point.z - wrist.z) / scale,
  ]);
}

let sequenceConfig = null;

/**
 * Configure sequence dimensions using values returned by /api/config.
 * @param {{sequence_length: number, features_per_frame: number}} config Shared sequence settings.
 */
export function configureLandmarks(config) {
  sequenceConfig = config;
}

/**
 * Capture the normalized hand shape and the raw wrist position for one frame.
 * @param {{x: number, y: number, z: number}[]} landmarks The 21 hand landmarks.
 * @returns {{shape: number[], wrist: [number, number, number], scale: number}} Frame data.
 */
export function makeFrame(landmarks) {
  const wrist = landmarks[0];
  const middleFingerMcp = landmarks[9];
  const distance = Math.hypot(
    middleFingerMcp.x - wrist.x,
    middleFingerMcp.y - wrist.y,
    middleFingerMcp.z - wrist.z,
  );
  return {
    shape: normalizeLandmarks(landmarks),
    wrist: [wrist.x, wrist.y, wrist.z],
    scale: distance === 0 ? 1 : distance,
  };
}

/**
 * Convert one configured sequence into flat per-frame feature values.
 * Wrist displacement relative to the first frame preserves motion between frames.
 * @param {{shape: number[], wrist: [number, number, number], scale: number}[]} frames Exactly SEQUENCE_LENGTH frames.
 * @returns {number[]} A flat vector sized by the shared sequence configuration.
 */
export function buildWindow(frames) {
  if (!sequenceConfig) {
    throw new Error("Load sequence settings from /api/config before building windows.");
  }
  const { sequence_length: sequenceLength, features_per_frame: featuresPerFrame } =
    sequenceConfig;
  if (frames.length !== sequenceLength || sequenceLength < 1) {
    throw new Error(`A window must contain exactly ${sequenceLength} frames.`);
  }

  const referenceWrist = frames[0].wrist;
  const scale = frames[0].scale || 1;
  const flattened = frames.flatMap((frame) => [
    ...frame.shape,
    (frame.wrist[0] - referenceWrist[0]) / scale,
    (frame.wrist[1] - referenceWrist[1]) / scale,
    (frame.wrist[2] - referenceWrist[2]) / scale,
  ]);
  if (flattened.length !== sequenceLength * featuresPerFrame) {
    throw new Error("Generated window size does not match /api/config.");
  }
  return flattened;
}
