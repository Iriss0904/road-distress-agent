export async function getJSON(path) {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`${path} -> ${res.status} ${await errorDetail(res)}`);
  }
  return res.json();
}

async function errorDetail(res) {
  try {
    const body = await res.json();
    return body.detail ?? res.statusText;
  } catch {
    return res.statusText;
  }
}
