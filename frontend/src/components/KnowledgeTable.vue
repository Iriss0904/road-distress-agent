<script setup>
import { useRuntimeStore } from "../stores/runtime.js";
import { useKnowledgeStore } from "../stores/knowledge.js";

const BYTES_PER_KIB = 1024;
const KIB_PER_MIB = 1024;
const MIB = BYTES_PER_KIB * KIB_PER_MIB;

defineProps({
  rows: { type: Array, required: true },
});

const emit = defineEmits(["open"]);
const runtime = useRuntimeStore();
const store = useKnowledgeStore();

function statusLabel(status) {
  return status === "indexed" ? runtime.t("knowledge.indexed") : runtime.t("knowledge.rawOnly");
}

function sizeLabel(bytes) {
  if (!bytes) return "—";
  if (bytes >= MIB) return `${(bytes / MIB).toFixed(1)} MiB`;
  return `${(bytes / BYTES_PER_KIB).toFixed(0)} KiB`;
}
</script>

<template>
  <section class="knowledge-table" data-testid="knowledge-table">
    <table>
      <thead>
        <tr>
          <th>{{ runtime.t("knowledge.document") }}</th>
          <th>{{ runtime.t("knowledge.lane") }}</th>
          <th>{{ runtime.t("knowledge.status") }}</th>
          <th>{{ runtime.t("knowledge.chunks") }}</th>
          <th>{{ runtime.t("knowledge.size") }}</th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="doc in rows"
          :key="doc.doc_id"
          :class="{ active: store.activeDocId === doc.doc_id }"
        >
          <td>
            <button type="button" class="doc-button" @click="emit('open', doc)">
              <span class="doc-icon">PDF</span>
              <span class="doc-copy">
                <strong>{{ doc.title }}</strong>
                <small>{{ doc.filename }}</small>
              </span>
            </button>
          </td>
          <td><span class="tag lane">{{ doc.lane }}</span></td>
          <td><span :class="['tag', doc.status]">{{ statusLabel(doc.status) }}</span></td>
          <td class="mono">{{ doc.chunk_count || "—" }}</td>
          <td>{{ sizeLabel(doc.size_bytes) }}</td>
        </tr>
      </tbody>
    </table>
    <p v-if="!rows.length" class="empty">{{ runtime.t("knowledge.empty") }}</p>
  </section>
</template>

<style scoped>
.knowledge-table {
  overflow: auto;
}

table {
  width: 100%;
  min-width: 760px;
  border-collapse: collapse;
}

th,
td {
  padding: 13px 14px;
  border-bottom: 1px solid var(--rd-border);
  font-size: 13px;
  text-align: left;
}

th {
  color: var(--rd-text-dim);
  background: #fbfcff;
  font-size: 12px;
  font-weight: 650;
}

tr.active td {
  background: #f7f8ff;
}

.doc-button {
  display: flex;
  width: 100%;
  min-width: 0;
  align-items: center;
  gap: 10px;
  border: 0;
  background: transparent;
  color: inherit;
  cursor: pointer;
  text-align: left;
}

.doc-button:focus-visible {
  border-radius: var(--rd-radius-control);
  outline: 2px solid var(--rd-accent);
  outline-offset: 2px;
}

.doc-icon {
  display: grid;
  width: 34px;
  height: 42px;
  flex: none;
  place-items: center;
  border: 1px solid #dfe5ff;
  border-radius: 6px;
  background: var(--rd-accent-soft);
  color: var(--rd-accent);
  font-size: 11px;
  font-weight: 750;
}

.doc-copy {
  min-width: 0;
}

.doc-copy strong,
.doc-copy small {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.doc-copy strong {
  color: var(--rd-text-strong);
  font-weight: 650;
}

.doc-copy small {
  margin-top: 4px;
  color: var(--rd-text-dim);
  font-size: 12px;
}

.tag {
  display: inline-flex;
  padding: 3px 8px;
  border-radius: 999px;
  font-size: 12px;
  white-space: nowrap;
}

.indexed {
  background: #eaf7f0;
  color: var(--rd-success);
}

.raw_only {
  background: #fff5dd;
  color: var(--rd-warning);
}

.lane {
  background: #e6f6f8;
  color: #087e8b;
}

.mono {
  font-variant-numeric: tabular-nums;
}

.empty {
  margin: 0;
  padding: 18px;
  color: var(--rd-text-dim);
  font-size: 13px;
}
</style>
