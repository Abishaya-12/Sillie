"use strict";

const STATUS_REFRESH_INTERVAL_MS = 5000;
const statusUrl = "/admin/api/status";
const registeredUsers = document.getElementById("registered-users");
const onlineNow = document.getElementById("online-now");
const activeRooms = document.getElementById("active-rooms");
const roomList = document.getElementById("room-list");
const userTableBody = document.getElementById("user-table-body");

/**
 * Fetch and render safe live admin status.
 */
async function refreshAdminStatus() {
  try {
    const response = await fetch(statusUrl, {
      credentials: "same-origin",
      cache: "no-store",
    });
    if (!response.ok) {
      throw new Error(`Status request failed (${response.status}).`);
    }
    renderStatus(await response.json());
  } catch (error) {
    console.error("Could not refresh admin status:", error);
  }
}

/**
 * Render aggregate counters, room occupancy, and users without HTML injection.
 * @param {{registered_users: number, online_now: number, active_rooms: number, rooms: {room: string, participant_count: number}[], users: object[]}} status Current server status.
 */
function renderStatus(status) {
  registeredUsers.textContent = String(status.registered_users);
  onlineNow.textContent = String(status.online_now);
  activeRooms.textContent = String(status.active_rooms);
  renderRooms(status.rooms);
  renderUsers(status.users);
}

/**
 * Render active room codes and occupancy without participant names.
 * @param {{room: string, participant_count: number}[]} rooms Current rooms.
 */
function renderRooms(rooms) {
  roomList.replaceChildren();
  if (!rooms.length) {
    roomList.append(createEmptyState("NO ACTIVE ROOMS"));
    return;
  }
  for (const room of rooms) {
    const item = document.createElement("li");
    item.textContent = `ROOM ${room.room} // ${room.participant_count} PLAYERS`;
    roomList.append(item);
  }
}

/**
 * Render account directory rows using text nodes only.
 * @param {object[]} users Current safe user records.
 */
function renderUsers(users) {
  userTableBody.replaceChildren();
  if (!users.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 7;
    cell.className = "empty-state";
    cell.textContent = "NO PLAYERS YET";
    row.append(cell);
    userTableBody.append(row);
    return;
  }
  for (const user of users) {
    const row = document.createElement("tr");
    [
      user.id,
      user.username,
      user.display_name,
      user.accessibility_profile,
      formatTimestamp(user.created_at),
      formatTimestamp(user.last_seen),
    ].forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = String(value);
      row.append(cell);
    });
    const statusCell = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = `presence-badge ${user.online ? "online" : "offline"}`;
    badge.textContent = user.online ? "ONLINE" : "OFFLINE";
    statusCell.append(badge);
    row.append(statusCell);
    userTableBody.append(row);
  }
}

/**
 * Format a timestamp for the account directory.
 * @param {string | null} value ISO timestamp or null.
 * @returns {string} Localized timestamp or em dash.
 */
function formatTimestamp(value) {
  return value ? new Date(value).toLocaleString() : "—";
}

/**
 * Create a simple empty-state list item.
 * @param {string} message Empty-state text.
 * @returns {HTMLLIElement} Rendered empty item.
 */
function createEmptyState(message) {
  const item = document.createElement("li");
  item.className = "empty-state";
  item.textContent = message;
  return item;
}

refreshAdminStatus();
window.setInterval(refreshAdminStatus, STATUS_REFRESH_INTERVAL_MS);
