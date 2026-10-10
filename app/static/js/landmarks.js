/**
 * Flatten consecutive raw landmark frames into the wire-format sequence window.
 * @param {number[][]} frames Non-null 129-value frames sampled by FrameClock.
 * @returns {number[]} One flattened window.
 */
export function flattenWindow(frames, settings) {
  if (
    frames.length !== settings.sequence_length
    || frames.some((frame) => frame.length !== settings.raw_frame_size)
  ) {
    throw new Error("Raw frames do not match the sequence settings from /api/config.");
  }
  return frames.flat();
}
