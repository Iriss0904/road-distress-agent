<script setup>
import { computed } from "vue";
import { knowledgeDocUrl } from "../api/knowledge.js";
import { useRuntimeStore } from "../stores/runtime.js";
import { useKnowledgeStore } from "../stores/knowledge.js";

const FIRST_NUMBER_RE = /\d+/;

const runtime = useRuntimeStore();
const store = useKnowledgeStore();

const snippet = computed(() => store.preview?.snippet || null);
const headingPath = computed(() => joinList(snippet.value?.heading_path, " / "));
const semanticRole = computed(() => joinList(snippet.value?.semantic_role, " · "));
const sourceUrl = computed(() => {
  const docId = store.activeDocId;
  if (!docId) return "";
  const page = firstPage(snippet.value?.source_pages);
  return page ? `${knowledgeDocUrl(docId)}#page=${page}` : knowledgeDocUrl(docId);
});

function joinList(value, separator) {
  return Array.isArray(value) ? value.join(separator) : value || "—";
}

function firstPage(value) {
  const match = String(value || "").match(FIRST_NUMBER_RE);
  return match ? match[0] : "";
}
</script>

<template>
  <aside class="knowledge-preview" data-testid="knowledge-preview">
    <header>
      <h2>{{ runtime.t("knowledge.previewTitle") }}</h2>
      <p>{{ store.activeDocument?.title || runtime.t("knowledge.noSelection") }}</p>
    </header>
    <div class="body">
      <p v-if="store.previewLoading" class="muted">{{ runtime.t("knowledge.previewLoading") }}</p>
      <p v-else-if="store.previewError" class="error">{{ store.previewError }}</p>
      <p v-else-if="store.activeDocument?.status === 'raw_only'" class="muted">
        {{ runtime.t("knowledge.rawOnlyNotice") }}
      </p>
      <p v-else-if="!snippet" class="muted">{{ runtime.t("knowledge.noPreview") }}</p>
      <template v-else>
        <div class="trace">
          <span>source_doc_id</span>
          <span>source_pages</span>
          <span>heading_path</span>
        </div>
        <h3>{{ snippet.clause_id || runtime.t("knowledge.previewExample") }}</h3>
        <p class="excerpt">{{ snippet.rawtext }}</p>
        <dl>
          <dt>{{ runtime.t("knowledge.contextPrefix") }}</dt>
          <dd>{{ snippet.context_prefix || "—" }}</dd>
          <dt>{{ runtime.t("knowledge.headingPath") }}</dt>
          <dd>{{ headingPath }}</dd>
          <dt>{{ runtime.t("knowledge.pages") }}</dt>
          <dd>{{ snippet.source_pages || "—" }}</dd>
          <dt>{{ runtime.t("knowledge.semanticRole") }}</dt>
          <dd>{{ semanticRole }}</dd>
        </dl>
        <a class="primary" :href="sourceUrl" target="_blank" rel="noreferrer">
          {{ runtime.t("knowledge.openSource") }}
        </a>
      </template>
    </div>
  </aside>
</template>

<style scoped>
.knowledge-preview {
  display: grid;
  min-width: 0;
  grid-template-rows: auto minmax(0, 1fr);
  overflow: hidden;
  border: 1px solid var(--rd-border);
  border-radius: var(--rd-radius-control);
  background: var(--rd-surface);
}

header {
  padding: 15px 16px;
  border-bottom: 1px solid var(--rd-border);
}

h2,
h3,
p {
  margin: 0;
}

h2 {
  color: var(--rd-text-strong);
  font-size: 16px;
}

header p,
.muted,
dt {
  color: var(--rd-text-dim);
  font-size: 12px;
}

header p {
  margin-top: 6px;
}

.body {
  overflow: auto;
  padding: 16px;
}

.trace {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 14px;
}

.trace span {
  padding: 3px 8px;
  border-radius: 999px;
  background: var(--rd-surface-soft);
  color: var(--rd-text-dim);
  font-size: 12px;
}

h3 {
  color: var(--rd-text-strong);
  font-size: 14px;
}

.excerpt {
  margin-top: 10px;
  padding: 14px;
  border-left: 3px solid var(--rd-accent);
  border-radius: 0 8px 8px 0;
  background: #f7f8ff;
  color: var(--rd-text);
  font-size: 14px;
  line-height: 1.72;
}

dl {
  display: grid;
  grid-template-columns: 88px 1fr;
  gap: 10px 12px;
  margin: 14px 0;
  padding-top: 12px;
  border-top: 1px solid var(--rd-border);
}

dd {
  margin: 0;
  color: var(--rd-text);
  font-size: 13px;
}

.primary {
  display: inline-flex;
  min-height: 32px;
  align-items: center;
  padding: 0 11px;
  border-radius: var(--rd-radius-control);
  background: var(--rd-accent);
  color: #fff;
  font-size: 12px;
  font-weight: 650;
  text-decoration: none;
}

.error {
  color: var(--rd-danger);
  font-size: 13px;
}
</style>
