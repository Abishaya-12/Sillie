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
