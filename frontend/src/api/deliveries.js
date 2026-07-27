import { getJSON } from "./client.js";

export function listDeliveries(userId) {
  const params = new URLSearchParams({ user_id: userId });
  return getJSON(`/api/deliveries?${params.toString()}`);
}

export function getDelivery(id) {
  return getJSON(`/api/deliveries/${encodeURIComponent(id)}`);
}

export function downloadUrl(id, versionNo) {
  return `/api/deliveries/${encodeURIComponent(id)}/versions/${versionNo}/download`;
}

export function regenerate(id) {
  return mutate(`/api/deliveries/${encodeURIComponent(id)}/regenerate`, "POST");
}

async function mutate(url, method) {
  const res = await fetch(url, { method });
  const text = await res.text();
  if (!res.ok) throw new Error(`${method} ${url} -> ${res.status} ${detail(text, res.statusText)}`);
  const streamError = sseError(text);
  if (streamError) throw new Error(streamError);
  return { ok: true };
}

function sseError(text) {
  const blocks = text.split("\n\n");
  for (const block of blocks) {
    const event = sseEvent(block);
    if (event?.name === "error") return payloadMessage(event.data);
  }
  return "";
}

function sseEvent(block) {
  const lines = block.split("\n");
  const data = [];
  let name = "message";
  for (const line of lines) {
    if (line.startsWith("event:")) name = line.slice(6).trim();
    if (line.startsWith("data:")) data.push(line.slice(5).trim());
  }
  return data.length ? { name, data: data.join("\n") } : null;
}

function payloadMessage(data) {
  try {
    const payload = JSON.parse(data);
    return payload.message || payload.detail || data;
  } catch {
    return data;
  }
}

function detail(text, fallback) {
  try {
    const body = JSON.parse(text);
    return body.detail ?? fallback;
  } catch {
    return text || fallback;
  }
}
