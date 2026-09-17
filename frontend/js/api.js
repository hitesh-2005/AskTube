/**
 * AskTube API Client
 */

const API_BASE = "/api";

export async function processVideo(url, language = null, forceRefresh = false) {
  const response = await fetch(`${API_BASE}/videos/process`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, language, force_refresh: forceRefresh }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.error?.message || `HTTP ${response.status}: Failed to process video`);
  }

  return response.json();
}

export async function getVideo(videoId) {
  const response = await fetch(`${API_BASE}/videos/${encodeURIComponent(videoId)}`);
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.error?.message || `HTTP ${response.status}: Failed to fetch video`);
  }
  return response.json();
}

export async function askQuestion(videoId, question) {
  const response = await fetch(`${API_BASE}/videos/${encodeURIComponent(videoId)}/questions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.error?.message || `HTTP ${response.status}: Failed to answer question`);
  }

  return response.json();
}
