import { getJSON } from "./client.js";

export function listThreads(userId, q) {
  const params = new URLSearchParams({ user_id: userId });
  if (q) params.set("q", q);
  return getJSON(`/api/threads?${params.toString()}`);
}

export function getThread(id) {
  return getJSON(`/api/threads/${encodeURIComponent(id)}`);
}

export function renameThread(id, title) {
  return mutate(`/api/threads/${encodeURIComponent(id)}`, "PATCH", { title });
}

export function deleteThread(id) {
  return mutate(`/api/threads/${encodeURIComponent(id)}`, "DELETE");
}

async function mutate(url, method, body) {
  const res = await fetch(url, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`${method} ${url} -> ${res.status} ${await detail(res)}`);
  return res.status === 204 ? null : res.json();
}

async function detail(res) {
  try {
    const body = await res.json();
    return body.detail ?? res.statusText;
  } catch {
    return res.statusText;
  }
}
