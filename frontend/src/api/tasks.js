import { getJSON } from "./client.js";

export function listProjects(userId) {
  const params = new URLSearchParams({ user_id: userId });
  return getJSON(`/api/projects?${params.toString()}`);
}

export function getLedger(projectId) {
  return getJSON(`/api/ledger/${encodeURIComponent(projectId)}`);
}

export function listUnfiled(userId) {
  const params = new URLSearchParams({ user_id: userId });
  return getJSON(`/api/ledger/unfiled?${params.toString()}`);
}

export function promote(projectId, payload) {
  return mutate(`/api/projects/${encodeURIComponent(projectId)}/promote`, "POST", payload);
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
