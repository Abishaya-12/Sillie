"use strict";

const SERVER_UNAVAILABLE_MESSAGE =
  "Cannot reach the Sillie server. Make sure `python server.py` is running and you opened http://127.0.0.1:5000";
const WRONG_SERVER_HINT = "The page may be served by the wrong server.";

/**
 * Fetch a JSON response and provide consistent network and server errors.
 * @param {string | URL} url The endpoint to request.
 * @param {RequestInit} [options] Fetch options.
 * @returns {Promise<any>} The parsed JSON response body.
 */
export async function fetchJson(url, options) {
  let response;
  try {
    response = await fetch(url, options);
  } catch (error) {
    throw new Error(SERVER_UNAVAILABLE_MESSAGE, { cause: error });
  }

  const contentType = response.headers.get("content-type") || "";
  const isJson = /(?:application\/json|\+json)/i.test(contentType);
  if (!response.ok) {
    if (!isJson) {
      throw new Error(`Request failed with HTTP ${response.status}. ${WRONG_SERVER_HINT}`);
    }

    const body = await parseJson(response);
    throw new Error(body?.error || `Request failed with HTTP ${response.status}.`);
  }

  if (!isJson) {
    throw new Error(`Expected a JSON response (HTTP ${response.status}). ${WRONG_SERVER_HINT}`);
  }

  return parseJson(response);
}

/**
 * Parse a response already confirmed to use a JSON content type.
 * @param {Response} response The HTTP response to parse.
 * @returns {Promise<any>} The parsed response body.
 */
async function parseJson(response) {
  try {
    return await response.json();
  } catch (error) {
    throw new Error(
      `Invalid JSON response (HTTP ${response.status}). ${WRONG_SERVER_HINT}`,
      { cause: error },
    );
  }
}
