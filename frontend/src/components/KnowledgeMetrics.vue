<script setup>
import { computed } from "vue";
import { useRuntimeStore } from "../stores/runtime.js";
import { useKnowledgeStore } from "../stores/knowledge.js";

const runtime = useRuntimeStore();
const store = useKnowledgeStore();

const metrics = computed(() => [
  {
    label: runtime.t("knowledge.rawPdf"),
    value: store.documents.length,
    note: runtime.t("knowledge.rawPdfNote"),
  },
  {
    label: runtime.t("knowledge.indexedDocs"),
    value: indexedCount(),
    note: runtime.t("knowledge.indexedDocsNote"),
  },
  {
    label: runtime.t("knowledge.vectorChunks"),
    value: store.summary?.qdrant?.points_count || 0,
    note: runtime.t("knowledge.vectorChunksNote"),
  },
  {
    label: runtime.t("knowledge.indexStatus"),
    value: store.summary?.qdrant?.status || "—",
    note: store.summary?.collection || "—",
  },
]);

function indexedCount() {
  return store.documents.filter((doc) => doc.status === "indexed").length;
}
</script>

<template>
  <div class="knowledge-metrics" data-testid="knowledge-metrics">
    <article v-for="metric in metrics" :key="metric.label" class="metric">
      <span>{{ metric.label }}</span>
      <strong>{{ metric.value }}</strong>
      <small>{{ metric.note }}</small>
    </article>
  </div>
</template>

<style scoped>
.knowledge-metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(150px, 1fr));
  gap: 12px;
}

.metric {
  min-height: 84px;
  padding: 14px 16px;
  border: 1px solid var(--rd-border);
  border-radius: var(--rd-radius-control);
  background: var(--rd-surface);
}

.metric span,
.metric small {
  display: block;
  color: var(--rd-text-dim);
  font-size: 12px;
}

.metric strong {
  display: block;
  margin-top: 8px;
  color: var(--rd-text-strong);
  font-size: 24px;
  line-height: 1;
}

.metric small {
  margin-top: 8px;
}

@media (max-width: 1100px) {
  .knowledge-metrics {
    grid-template-columns: repeat(2, minmax(150px, 1fr));
  }
}
</style>
