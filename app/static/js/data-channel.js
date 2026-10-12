"use strict";

let sillieChannel = null;
let messageHandler = null;

/**
 * Attach the reliable WebRTC data channel used for lightweight call messages.
 * @param {RTCDataChannel} channel The negotiated Sillie data channel.
 */
export function attachDataChannel(channel) {
  sillieChannel = channel;
  sillieChannel.onmessage = dispatchMessage;
}

/**
 * Register the callback invoked for valid JSON messages from the peer.
 * @param {(message: {type: string, text: string, ts: number}) => void} handler Message callback.
 */
export function onMessage(handler) {
  messageHandler = handler;
}

/**
 * Send a typed JSON message to the connected peer.
 * @param {{type: string, text: string, ts: number}} message Message to send.
 * @returns {boolean} Whether the message was sent on an open channel.
 */
export function sendMessage(message) {
  if (
    !sillieChannel
    || sillieChannel.readyState !== "open"
    || typeof message?.type !== "string"
    || typeof message.text !== "string"
    || typeof message.ts !== "number"
  ) {
    return false;
  }
  sillieChannel.send(JSON.stringify(message));
  return true;
}

/**
 * Parse and dispatch a well-formed message received over the data channel.
 * @param {MessageEvent<string>} event Incoming data-channel message.
 */
function dispatchMessage(event) {
  let message;
  try {
    message = JSON.parse(event.data);
  } catch {
    return;
  }
  if (
    message
    && typeof message.type === "string"
    && typeof message.text === "string"
    && typeof message.ts === "number"
  ) {
    messageHandler?.(message);
  }
}
