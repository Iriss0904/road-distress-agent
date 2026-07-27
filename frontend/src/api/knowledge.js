import { getJSON } from "./client.js";

export function getKnowledgeSummary() {
  return getJSON("/api/knowledge/summary");
}

export function getKnowledgePreview(docId) {
  const params = new URLSearchParams({ doc_id: docId });
  return getJSON(`/api/knowledge/preview?${params.toString()}`);
}

export function knowledgeDocUrl(docId) {
  const params = new URLSearchParams({ doc_id: docId });
  return `/api/knowledge/doc?${params.toString()}`;
}
