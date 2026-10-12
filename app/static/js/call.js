"use strict";

import { attachDataChannel, onMessage, sendMessage } from "./data-channel.js";

const MEDIA_CONSTRAINTS = {
  audio: { echoCancellation: true },
  video: {
    width: { ideal: 640 },
    height: { ideal: 480 },
    frameRate: { ideal: 30 },
  },
};
const SIGNALING_TIMEOUT_MS = 12000;
const ICE_RESTART_TIMEOUT_MS = 10000;
const CAPTION_PREVIEW = "This is a test caption.";

export const localVideo = document.getElementById("local-video");
const remoteVideo = document.getElementById("remote-video");
const callShell = document.querySelector(".call-shell");
const roomCode = callShell.dataset.roomCode;
const statusLabel = document.getElementById("call-status");
const statusLed = document.getElementById("call-status-led");
const messageBox = document.getElementById("call-message");
const captionPanel = document.getElementById("incoming-captions");
const remotePlaceholder = document.getElementById("remote-placeholder");

let localStream = null;
let socket = null;
let peerConnection = null;
let iceServers = [];
let pendingCandidates = [];
let frameProcessor = null;
let frameCallbackHandle = null;
let iceRestartAttempts = 0;
let isEnding = false;
let remotePeerPresent = false;
let hasConnectedSocket = false;
let dataChannelCreated = false;

/**
 * Install the single per-frame extension point for future camera processing.
 * @param {(video: HTMLVideoElement, now: number, metadata: VideoFrameCallbackMetadata) => void} processor Frame processor callback.
 */
export function setFrameProcessor(processor) {
  frameProcessor = processor;
  if (
    frameProcessor
    && localStream
    && frameCallbackHandle === null
    && typeof localVideo.requestVideoFrameCallback === "function"
  ) {
    frameCallbackHandle = localVideo.requestVideoFrameCallback(processLocalFrame);
  }
}

/**
 * Forward each local camera frame to the optional processing hook.
 * @param {number} now Frame timestamp.
 * @param {VideoFrameCallbackMetadata} metadata Video frame metadata.
 */
function processLocalFrame(now, metadata) {
  frameCallbackHandle = null;
  if (!localStream || !frameProcessor) {
    return;
  }
  frameProcessor(localVideo, now, metadata);
  frameCallbackHandle = localVideo.requestVideoFrameCallback(processLocalFrame);
}

/**
 * Start camera, load ICE configuration, establish signaling, and join the room.
 */
async function startCall() {
  if (!window.isSecureContext) {
    showMessage("Camera and microphone require HTTPS outside localhost.");
    setCallStatus("ENDED");
    return;
  }

  try {
    localStream = await navigator.mediaDevices.getUserMedia(MEDIA_CONSTRAINTS);
    localVideo.srcObject = localStream;
    await localVideo.play();
    iceServers = await loadIceServers();
    createPeerConnection();
    onMessage(handleDataMessage);
    socket = window.io({ timeout: SIGNALING_TIMEOUT_MS });
    socket.on("connect", handleSocketConnect);
    socket.on("disconnect", handleSocketDisconnect);
    socket.on("connect_error", handleSocketError);
    socket.on("room_full", handleRoomFull);
    socket.on("join_error", handleJoinError);
    socket.on("peer_present", handlePeerPresent);
    socket.on("peer_joined", handlePeerJoined);
    socket.on("peer_left", handlePeerLeft);
    socket.on("signal", handleSignal);
  } catch (error) {
    showMediaError(error);
    await endCall(false);
  }
}

/**
 * Retrieve the login-protected STUN/TURN configuration for this call.
 * @returns {Promise<RTCIceServer[]>} Configured ICE servers.
 */
async function loadIceServers() {
  const response = await fetch("/api/ice-config", { credentials: "same-origin" });
  if (!response.ok) {
    throw new Error("Could not load call network configuration.");
  }
  const configuration = await response.json();
  return configuration.iceServers;
}

/**
 * Create a peer connection and attach the already-open local media stream.
 * @returns {RTCPeerConnection} The room's peer connection.
 */
function createPeerConnection() {
  if (peerConnection) {
    return peerConnection;
  }
  peerConnection = new RTCPeerConnection({ iceServers });
  for (const track of localStream.getTracks()) {
    peerConnection.addTrack(track, localStream);
  }
  peerConnection.onicecandidate = handleIceCandidate;
  peerConnection.ontrack = handleRemoteTrack;
  peerConnection.onconnectionstatechange = handleConnectionState;
  peerConnection.ondatachannel = handleDataChannel;
  return peerConnection;
}

/**
 * Join the room after the authenticated signaling socket connects.
 */
function handleSocketConnect() {
  if (hasConnectedSocket) {
    closePeerConnection();
    createPeerConnection();
    remotePeerPresent = false;
  }
  hasConnectedSocket = true;
  showMessage("");
  socket.emit("join", { room: roomCode });
  setCallStatus("WAITING FOR PEER");
}

/**
 * Show reconnecting state after a signaling transport disconnect.
 * @param {string} reason Socket.IO disconnect reason.
 */
function handleSocketDisconnect(reason) {
  if (isEnding) {
    return;
  }
  setCallStatus("RECONNECTING");
  showMessage("Signalling disconnected. Reconnecting to the room...");
  if (reason === "io server disconnect") {
    socket.connect();
  }
}

/**
 * Explain failures to connect to the authenticated signaling endpoint.
 * @param {Error} error Socket.IO connection error.
 */
function handleSocketError(error) {
  showMessage(`Signalling disconnected: ${error.message || "connection failed"}.`);
}

/**
 * Show a friendly room-capacity message and stop local resources.
 */
function handleRoomFull() {
  showMessage("This room is full. A call can have only two people.");
  endCall(false);
}

/**
 * Show a friendly invalid-room error.
 * @param {{message?: string}} error Server validation error.
 */
function handleJoinError(error) {
  showMessage(error.message || "That room code is not valid.");
}

/**
 * Prepare the connection after joining a room that already has a peer.
 * @param {{display_name: string, profile: string}} peer Existing room participant.
 */
function handlePeerPresent(peer) {
  remotePeerPresent = true;
  showMessage(`Connecting to ${peer.display_name}...`);
  createPeerConnection();
  setCallStatus("CONNECTING");
}

/**
 * Create the offer when another participant joins this existing room.
 * @param {{display_name: string, profile: string}} peer Newly joined participant.
 */
async function handlePeerJoined(peer) {
  remotePeerPresent = true;
  showMessage(`Connecting to ${peer.display_name}...`);
  createPeerConnection();
  setCallStatus("CONNECTING");
  await createAndSendOffer(false);
}

/**
 * Close the old peer connection when the other participant leaves the room.
 */
function handlePeerLeft() {
  remotePeerPresent = false;
  closePeerConnection();
  remoteVideo.srcObject = null;
  remotePlaceholder.hidden = false;
  setCallStatus("ENDED");
  showMessage("The other participant ended the call.");
}

/**
 * Build an offer and optionally restart ICE after a failed connection.
 * @param {boolean} restartIce Whether this offer must restart ICE.
 */
async function createAndSendOffer(restartIce) {
  const connection = createPeerConnection();
  if (!dataChannelCreated) {
    const channel = connection.createDataChannel("sillie", { ordered: true });
    attachDataChannel(channel);
    dataChannelCreated = true;
  }
  const offer = await connection.createOffer(
    restartIce ? { iceRestart: true } : undefined,
  );
  await connection.setLocalDescription(offer);
  sendSignal("offer", connection.localDescription);
}

/**
 * Relay a gathered ICE candidate through the room signaling socket.
 * @param {RTCPeerConnectionIceEvent} event ICE candidate event.
 */
function handleIceCandidate(event) {
  if (event.candidate) {
    sendSignal("candidate", event.candidate.toJSON());
  }
}

/**
 * Attach the received peer media stream to the large remote video.
 * @param {RTCTrackEvent} event Remote track event.
 */
function handleRemoteTrack(event) {
  remoteVideo.srcObject = event.streams[0];
  remotePlaceholder.hidden = true;
  remoteVideo.play().catch(showPlaybackError);
}

/**
 * Update call status and attempt exactly one ICE restart for a failed link.
 */
async function handleConnectionState() {
  if (!peerConnection) {
    return;
  }
  if (peerConnection.connectionState === "connected") {
    setCallStatus("CONNECTED");
    showMessage("");
    return;
  }
  if (["connecting", "new"].includes(peerConnection.connectionState)) {
    setCallStatus("CONNECTING");
    return;
  }
  if (
    ["disconnected", "failed"].includes(peerConnection.connectionState)
    && iceRestartAttempts < 1
    && remotePeerPresent
  ) {
    iceRestartAttempts += 1;
    setCallStatus("RECONNECTING");
    showMessage("Trying to restore the connection...");
    try {
      await createAndSendOffer(true);
      window.setTimeout(checkIceRestart, ICE_RESTART_TIMEOUT_MS);
    } catch (error) {
      setCallStatus("ENDED");
      showMessage(`Could not restart the connection: ${error.message}`);
    }
  }
}

/**
 * Check whether the single ICE restart succeeded.
 */
function checkIceRestart() {
  if (
    peerConnection
    && peerConnection.connectionState !== "connected"
    && !isEnding
  ) {
    setCallStatus("ENDED");
    showMessage("The connection could not be restored. End the call and try again.");
  }
}

/**
 * Process a relayed offer, answer, or ICE candidate.
 * @param {{type: string, payload: RTCSessionDescriptionInit | RTCIceCandidateInit}} message Signaling message.
 */
async function handleSignal(message) {
  createPeerConnection();
  if (message.type === "offer") {
    await peerConnection.setRemoteDescription(message.payload);
    await flushPendingCandidates();
    const answer = await peerConnection.createAnswer();
    await peerConnection.setLocalDescription(answer);
    sendSignal("answer", peerConnection.localDescription);
    setCallStatus("CONNECTING");
  } else if (message.type === "answer") {
    await peerConnection.setRemoteDescription(message.payload);
    await flushPendingCandidates();
  } else if (message.type === "candidate") {
    if (peerConnection.remoteDescription) {
      await peerConnection.addIceCandidate(message.payload);
    } else {
      pendingCandidates.push(message.payload);
    }
  }
}

/**
 * Add ICE candidates received before the remote session description.
 */
async function flushPendingCandidates() {
  while (pendingCandidates.length > 0) {
    await peerConnection.addIceCandidate(pendingCandidates.shift());
  }
}

/**
 * Send a typed WebRTC signaling message to the server.
 * @param {"offer" | "answer" | "candidate"} type Signaling message kind.
 * @param {RTCSessionDescriptionInit | RTCIceCandidateInit} payload Signal content.
 */
function sendSignal(type, payload) {
  if (socket?.connected) {
    socket.emit("signal", { type, payload });
  }
}

/**
 * Attach the data channel created by the room offerer.
 * @param {RTCDataChannelEvent} event Incoming peer data channel.
 */
function handleDataChannel(event) {
  attachDataChannel(event.channel);
}

/**
 * Append incoming caption messages to the caption panel.
 * @param {{type: string, text: string, ts: number}} message Data-channel message.
 */
function handleDataMessage(message) {
  if (message.type !== "caption") {
    return;
  }
  const caption = document.createElement("p");
  caption.textContent = message.text;
  captionPanel.replaceChildren(caption);
}

/**
 * Toggle local audio track transmission.
 */
function toggleMicrophone() {
  const track = localStream?.getAudioTracks()[0];
  if (!track) {
    showMessage("No microphone was found.");
    return;
  }
  track.enabled = !track.enabled;
  updateToggleButton("mic-button", "MIC", track.enabled);
}

/**
 * Toggle local video track transmission.
 */
function toggleCamera() {
  const track = localStream?.getVideoTracks()[0];
  if (!track) {
    showMessage("No camera was found.");
    return;
  }
  track.enabled = !track.enabled;
  updateToggleButton("cam-button", "CAM", track.enabled);
}

/**
 * Update the text and pressed state of a media toggle.
 * @param {string} buttonId Control element identifier.
 * @param {"MIC" | "CAM"} label Control label.
 * @param {boolean} enabled Whether the track is enabled.
 */
function updateToggleButton(buttonId, label, enabled) {
  const button = document.getElementById(buttonId);
  button.textContent = `${label}: ${enabled ? "ON" : "OFF"}`;
  button.setAttribute("aria-pressed", String(enabled));
}

/**
 * Copy the room code to the local clipboard.
 */
async function copyRoomCode() {
  try {
    await navigator.clipboard.writeText(roomCode);
    showMessage("Room code copied.");
  } catch {
    showMessage(`Share this room code: ${roomCode}`);
  }
}

/**
 * Send a sample caption to verify the reliable peer data channel.
 */
function sendTestCaption() {
  const sent = sendMessage({
    type: "caption",
    text: CAPTION_PREVIEW,
    ts: Date.now(),
  });
  showMessage(sent ? "Test caption sent." : "Connect to your peer before testing captions.");
}

/**
 * Stop local media, close the peer connection, leave signaling, and return to lobby.
 * @param {boolean} navigate Whether to return to the lobby after cleanup.
 */
async function endCall(navigate = true) {
  if (isEnding) {
    if (navigate) {
      window.location.assign("/lobby");
    }
    return;
  }
  isEnding = true;
  setCallStatus("ENDED");
  closePeerConnection();
  if (localStream) {
    for (const track of localStream.getTracks()) {
      track.stop();
    }
  }
  if (frameCallbackHandle !== null) {
    localVideo.cancelVideoFrameCallback(frameCallbackHandle);
    frameCallbackHandle = null;
  }
  if (socket) {
    if (socket.connected) {
      socket.emit("leave");
    }
    socket.disconnect();
  }
  if (navigate) {
    window.location.assign("/lobby");
  }
}

/**
 * Close the current peer connection and discard queued candidate data.
 */
function closePeerConnection() {
  if (peerConnection) {
    peerConnection.onconnectionstatechange = null;
    peerConnection.close();
    peerConnection = null;
  }
  dataChannelCreated = false;
  pendingCandidates = [];
  iceRestartAttempts = 0;
}

/**
 * Display a friendly browser permission or camera failure.
 * @param {Error} error Media acquisition error.
 */
function showMediaError(error) {
  if (error.name === "NotAllowedError" || error.name === "PermissionDeniedError") {
    showMessage("Camera or microphone permission was denied. Allow access and reload.");
  } else if (error.name === "NotFoundError" || error.name === "DevicesNotFoundError") {
    showMessage("No camera or microphone was found on this device.");
  } else {
    showMessage(error.message || "Could not start your camera and microphone.");
  }
}

/**
 * Explain why browser autoplay did not start remote audio/video.
 * @param {Error} error Playback failure.
 */
function showPlaybackError(error) {
  showMessage(`Allow media playback to hear your peer. ${error.message}`);
}

/**
 * Render a call status and its corresponding LED state.
 * @param {"WAITING FOR PEER" | "CONNECTING" | "CONNECTED" | "RECONNECTING" | "ENDED"} status Current call state.
 */
function setCallStatus(status) {
  statusLabel.textContent = status;
  statusLed.dataset.state = status.toLowerCase().replaceAll(" ", "-");
}

/**
 * Display an inline call message or clear the message panel.
 * @param {string} message Friendly call feedback.
 */
function showMessage(message) {
  messageBox.textContent = message;
}

document.getElementById("mic-button").addEventListener("click", toggleMicrophone);
document.getElementById("cam-button").addEventListener("click", toggleCamera);
document.getElementById("copy-code-button").addEventListener("click", copyRoomCode);
document.getElementById("test-caption-button").addEventListener("click", sendTestCaption);
document.getElementById("end-call-button").addEventListener("click", handleEndButtonClick);

startCall();

/**
 * End the call in response to the user pressing the END CALL key.
 * @param {MouseEvent} event Button click event.
 */
function handleEndButtonClick(event) {
  event.preventDefault();
  endCall();
}
