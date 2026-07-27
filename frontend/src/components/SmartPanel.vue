<script setup>
import { computed, ref } from "vue";
import PdfPageViewer from "./PdfPageViewer.vue";
import { citationMeta } from "../lib/citations.js";
import { useRuntimeStore } from "../stores/runtime.js";

const props = defineProps({
  refMap: { type: Object, default: () => ({}) },
  snapshot: { type: Object, default: () => ({}) },
});

const activeTab = ref("processDraft");
const runtime = useRuntimeStore();
const currentId = ref("");
const currentRef = ref(null);
const current = computed(() => currentRef.value || props.refMap[currentId.value] || null);
const draftRows = computed(() => [
  [runtime.t("panel.distress"), fieldText(["defect_category", "disease", "distress_type"])],
  [runtime.t("panel.method"), fieldText(["chosen_method", "method", "final_answer.method_selection"])],
  [runtime.t("panel.steps"), fieldText(["procedure", "construction_steps", "final_answer.steps"])],
  [runtime.t("panel.acceptance"), fieldText(["acceptance", "acceptance_criteria", "final_answer.acceptance_criteria"])],
]);
const hasDraft = computed(() => draftRows.value.some((row) => row[1]));
const pdfHref = computed(() => {
  if (!current.value?.source_doc) return "";
  const docId = encodeURIComponent(current.value.source_doc);
  return `/api/knowledge/doc?doc_id=${docId}#page=${pageNumber(current.value)}`;
});
const pdfSrc = computed(() => {
  if (!current.value?.source_doc) return "";
  const docId = encodeURIComponent(current.value.source_doc);
  return `/api/knowledge/doc?doc_id=${docId}`;
});

function trace(payload) {
  const refId = typeof payload === "string" ? payload : payload?.refId;
  currentId.value = refId || "";
  currentRef.value = typeof payload === "string" ? null : payload?.refData || null;
  activeTab.value = "sourceTrace";
}

function fieldText(paths) {
  return formatValue(paths.map((path) => pathValue(path)).find(Boolean));
}

function pathValue(path) {
  return path.split(".").reduce((value, key) => value?.[key], props.snapshot);
}

function formatValue(value) {
  if (Array.isArray(value)) return value.join("\n");
  if (value && typeof value === "object") return value.summary || "";
  return value || "";
}

function pageNumber(ref) {
  if (ref.page_number) return ref.page_number;
  const match = String(ref.source_pages || "").match(/\d+/);
  return match ? match[0] : "1";
}

defineExpose({ trace });
</script>

<template>
  <aside class="panel">
    <div class="tabs" role="tablist">
      <button :class="{ active: activeTab === 'processDraft' }" type="button" @click="activeTab = 'processDraft'">
        {{ runtime.t("panel.draft") }}
      </button>
      <button :class="{ active: activeTab === 'sourceTrace' }" type="button" @click="activeTab = 'sourceTrace'">
        {{ runtime.t("panel.source") }}
      </button>
    </div>

    <section v-if="activeTab === 'processDraft'" class="body">
      <template v-if="hasDraft">
        <div v-for="[label, value] in draftRows" :key="label" class="draft-row">
          <span>{{ label }}</span>
          <p>{{ value || runtime.t("common.pending") }}</p>
        </div>
      </template>
      <p v-else class="empty">{{ runtime.t("panel.emptyDraft") }}</p>
    </section>

    <section v-else class="body">
      <template v-if="current">
        <header class="source-head">
          <div>
            <h2>{{ current.title || current.source_doc || currentId }}</h2>
            <p class="meta">{{ citationMeta(current) }}</p>
          </div>
          <a v-if="pdfHref" :href="pdfHref" target="_blank" rel="noopener" data-testid="open-pdf">
            {{ runtime.t("panel.openPdf", { page: pageNumber(current) }) }}
          </a>
        </header>

        <PdfPageViewer
          v-if="pdfSrc"
          :key="`${pdfSrc}#${pageNumber(current)}`"
          :src="pdfSrc"
          :page="pageNumber(current)"
          :title="current.title || current.source_doc || currentId"
          data-testid="pdf-viewer"
        />

        <details v-if="current.snippet || current.bbox" class="source-extra">
          <summary>{{ runtime.t("panel.sourceExcerpt") }}</summary>
          <p v-if="current.snippet" class="snippet">{{ current.snippet }}</p>
          <p v-if="current.bbox" class="highlight">{{ runtime.t("panel.bbox") }}</p>
        </details>
      </template>
      <p v-else class="empty">{{ runtime.t("panel.emptySource") }}</p>
    </section>
  </aside>
</template>

<style scoped>
.panel {
  display: flex;
  min-width: 280px;
  flex: 0 0 auto;
  flex-direction: column;
  border-left: 1px solid var(--rd-border);
  background: var(--rd-surface);
}

.tabs {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
  padding: 10px;
  border-bottom: 1px solid var(--rd-border);
}

.tabs button {
  min-height: 34px;
  border: 1px solid var(--rd-border);
  border-radius: var(--rd-radius-control);
  background: var(--rd-surface-soft);
  color: var(--rd-text-dim);
  cursor: pointer;
}

.tabs button.active {
  border-color: #cfd7ff;
  background: var(--rd-accent-soft);
  color: var(--rd-accent);
  font-weight: 650;
}

.body {
  flex: 1;
  overflow: auto;
  padding: 16px;
}

.draft-row {
  padding: 12px 0;
  border-bottom: 1px solid var(--rd-border);
}

.draft-row span,
.meta {
  color: var(--rd-text-dim);
  font-size: 12px;
}

.draft-row p {
  margin: 6px 0 0;
  white-space: pre-wrap;
}

h2 {
  margin: 0 0 8px;
  color: var(--rd-text-strong);
  font-size: 16px;
}

.source-head {
  display: grid;
  gap: 10px;
  margin-bottom: 12px;
}

.source-head a {
  justify-self: start;
}

.source-extra {
  margin-top: 12px;
  border-top: 1px solid var(--rd-border);
  padding-top: 10px;
}

.source-extra summary {
  cursor: pointer;
  color: var(--rd-text-dim);
  font-size: 12px;
  font-weight: 650;
}

.snippet,
.highlight {
  padding: 10px;
  border-radius: var(--rd-radius-control);
  background: var(--rd-surface-soft);
  line-height: 1.6;
}

a {
  color: var(--rd-accent);
  font-weight: 650;
}

.empty {
  margin: 0;
  color: var(--rd-text-dim);
}
</style>
