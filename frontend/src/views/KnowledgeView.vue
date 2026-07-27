<script setup>
import { computed, onMounted, ref } from "vue";
import KnowledgeMetrics from "../components/KnowledgeMetrics.vue";
import KnowledgePreview from "../components/KnowledgePreview.vue";
import KnowledgeTable from "../components/KnowledgeTable.vue";
import { useRuntimeStore } from "../stores/runtime.js";
import { useKnowledgeStore } from "../stores/knowledge.js";

const runtime = useRuntimeStore();
const store = useKnowledgeStore();
const query = ref("");

const rows = computed(() => {
  const term = query.value.trim().toLowerCase();
  if (!term) return store.documents;
  return store.documents.filter((doc) => searchableText(doc).includes(term));
});

const subtitle = computed(() =>
  runtime.t("knowledge.subtitle", { collection: store.summary?.collection || "—" }),
);

const loadedAt = computed(() => {
  if (!store.summary?.loaded_at) return "";
  return runtime.t("knowledge.loadedAt", { time: store.summary.loaded_at.slice(0, 19) });
});

onMounted(load);

async function load() {
  try {
    await store.loadSummary();
    if (!store.activeDocId) await store.openFirstDocument();
  } catch (err) {
    store.error = err.message;
  }
}

async function refresh() {
  try {
    await store.refresh();
  } catch (err) {
    store.error = err.message;
  }
}

function open(doc) {
  store.openDocument(doc.doc_id).catch((err) => {
    store.previewError = err.message;
  });
}

function searchableText(doc) {
  return [doc.title, doc.filename, doc.lane, doc.status].filter(Boolean).join(" ").toLowerCase();
}
</script>

<template>
  <section class="knowledge-view">
    <header class="page-head">
      <div>
        <h1>{{ runtime.t("knowledge.title") }}</h1>
        <p>{{ subtitle }}</p>
      </div>
      <div class="head-actions">
        <span v-if="loadedAt" class="loaded-at">{{ loadedAt }}</span>
        <button type="button" :disabled="store.refreshing" @click="refresh">
          {{ store.refreshing ? runtime.t("knowledge.refreshing") : runtime.t("knowledge.refresh") }}
        </button>
      </div>
    </header>

    <KnowledgeMetrics />

    <p v-if="store.error" class="error">{{ store.error }}</p>
    <p v-if="store.loading && !store.summary" class="loading">{{ runtime.t("knowledge.loading") }}</p>

    <div class="workspace">
      <section class="documents-panel">
        <div class="toolbar">
          <label class="search">
            <span aria-hidden="true">⌕</span>
            <input v-model="query" type="search" :placeholder="runtime.t('knowledge.search')">
          </label>
        </div>
        <KnowledgeTable :rows="rows" @open="open" />
      </section>
      <KnowledgePreview />
    </div>
  </section>
</template>

<style scoped>
.knowledge-view {
  display: grid;
  min-height: 100%;
  grid-template-rows: auto auto auto minmax(0, 1fr);
  gap: 16px;
  padding: 24px;
}

.page-head,
.head-actions {
  display: flex;
  align-items: center;
  gap: 12px;
}

.page-head {
  justify-content: space-between;
}

h1,
p {
  margin: 0;
}

h1 {
  color: var(--rd-text-strong);
  font-size: 24px;
  line-height: 1.2;
}

.page-head p,
.loaded-at,
.loading {
  color: var(--rd-text-dim);
  font-size: 13px;
}

.page-head p {
  margin-top: 6px;
}

button {
  min-height: 32px;
  padding: 0 11px;
  border: 1px solid var(--rd-border-strong);
  border-radius: var(--rd-radius-control);
  background: var(--rd-surface);
  color: var(--rd-text);
  cursor: pointer;
  font-weight: 650;
}

button:disabled {
  cursor: wait;
  opacity: 0.65;
}

.workspace {
  display: grid;
  min-height: 0;
  grid-template-columns: minmax(620px, 1.7fr) minmax(330px, 0.9fr);
  gap: 16px;
}

.documents-panel {
  min-width: 0;
  overflow: hidden;
  border: 1px solid var(--rd-border);
  border-radius: var(--rd-radius-control);
  background: var(--rd-surface);
}

.toolbar {
  padding: 12px 14px;
  border-bottom: 1px solid var(--rd-border);
  background: linear-gradient(180deg, #fff, #fafbff);
}

.search {
  display: flex;
  height: 34px;
  align-items: center;
  gap: 8px;
  padding: 0 11px;
  border: 1px solid var(--rd-border-strong);
  border-radius: var(--rd-radius-control);
  background: var(--rd-surface);
  color: var(--rd-text-dim);
}

.search input {
  width: 100%;
  min-width: 0;
  border: 0;
  background: transparent;
  color: var(--rd-text);
  outline: none;
}

.error {
  margin: 0;
  color: var(--rd-danger);
  font-size: 13px;
}

@media (max-width: 1120px) {
  .workspace {
    grid-template-columns: 1fr;
  }
}
</style>
