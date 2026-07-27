import { defineStore } from "pinia";
import { computed, ref } from "vue";
import { getKnowledgePreview, getKnowledgeSummary } from "../api/knowledge.js";

const EMPTY_COUNT = 0;
const INDEXED_STATUS = "indexed";

export const useKnowledgeStore = defineStore("knowledge", () => {
  const summary = ref(null);
  const preview = ref(null);
  const activeDocId = ref("");
  const loading = ref(false);
  const refreshing = ref(false);
  const previewLoading = ref(false);
  const error = ref("");
  const previewError = ref("");
  const documents = computed(() => summary.value?.documents || []);
  const activeDocument = computed(() =>
    documents.value.find((doc) => doc.doc_id === activeDocId.value) || null,
  );

  async function loadSummary(options = {}) {
    if (summary.value && !options.force) return summary.value;
    setLoading(options.force);
    error.value = "";
    try {
      summary.value = normalizeSummary(await getKnowledgeSummary());
      return summary.value;
    } catch (err) {
      summary.value = null;
      error.value = err.message;
      throw err;
    } finally {
      clearLoading();
    }
  }

  async function refresh() {
    const previous = activeDocId.value;
    await loadSummary({ force: true });
    if (previous && documents.value.some((doc) => doc.doc_id === previous)) {
      await openDocument(previous, { force: true });
      return;
    }
    await openFirstDocument();
  }

  async function openFirstDocument() {
    const doc = documents.value.find((row) => row.status === INDEXED_STATUS) || documents.value[0];
    if (doc) await openDocument(doc.doc_id);
  }

  async function openDocument(docId, options = {}) {
    const doc = documents.value.find((row) => row.doc_id === docId);
    if (!doc) throw new Error(`unknown knowledge document: ${docId}`);
    activeDocId.value = docId;
    previewError.value = "";
    if (doc.status !== INDEXED_STATUS) {
      preview.value = null;
      return;
    }
    if (!options.force && preview.value?.doc_id === docId) return;
    await loadPreview(docId);
  }

  async function loadPreview(docId) {
    previewLoading.value = true;
    preview.value = null;
    try {
      preview.value = normalizePreview(await getKnowledgePreview(docId));
    } catch (err) {
      previewError.value = err.message;
      throw err;
    } finally {
      previewLoading.value = false;
    }
  }

  function setLoading(force) {
    loading.value = true;
    refreshing.value = Boolean(force);
  }

  function clearLoading() {
    loading.value = false;
    refreshing.value = false;
  }

  return {
    summary,
    preview,
    activeDocId,
    loading,
    refreshing,
    previewLoading,
    error,
    previewError,
    documents,
    activeDocument,
    loadSummary,
    refresh,
    openFirstDocument,
    openDocument,
  };
});

function normalizeSummary(payload) {
  if (!payload || typeof payload !== "object") {
    throw new Error("/api/knowledge/summary returned an invalid payload");
  }
  return {
    ...payload,
    documents: asArray(payload.documents, "/api/knowledge/summary.documents"),
    qdrant: payload.qdrant || { points_count: EMPTY_COUNT },
  };
}

function normalizePreview(payload) {
  if (!payload || typeof payload !== "object" || !payload.snippet) {
    throw new Error("/api/knowledge/preview returned an invalid payload");
  }
  return payload;
}

function asArray(value, endpoint) {
  if (Array.isArray(value)) return value;
  throw new Error(`${endpoint} returned a non-array payload`);
}
