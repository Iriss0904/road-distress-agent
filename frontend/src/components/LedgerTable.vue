<script setup>
import { computed, ref } from "vue";
import { useRouter } from "vue-router";
import { UNFILED_PROJECT_ID, useTasksStore } from "../stores/tasks.js";
import { useRuntimeStore } from "../stores/runtime.js";
import { useThreadsStore } from "../stores/threads.js";

const runtime = useRuntimeStore();
const store = useTasksStore();
const threads = useThreadsStore();
const router = useRouter();
const actionError = ref("");

const selectedCount = computed(() => store.selected.size);
const rows = computed(() => store.activeLedger.rows || []);
const project = computed(() => store.activeLedger.project);
const canPromote = computed(() => realProjectId.value && threads.activeId);
const realProjectId = computed(() => {
  const id = project.value?.project_id;
  return id && id !== UNFILED_PROJECT_ID ? id : "";
});

const evidenceLabels = {
  complete: "evidence.complete",
  gap: "evidence.gap",
  missing: "evidence.missing",
};

const reviewLabels = {
  active: "review.active",
  reviewed: "review.reviewed",
  delivered: "review.delivered",
  archived: "review.archived",
};

async function promoteActiveThread() {
  actionError.value = "";
  try {
    await store.promoteThread(realProjectId.value, threads.activeId);
  } catch (err) {
    actionError.value = err.message;
  }
}

async function viewThread(threadId) {
  if (!threadId) return;
  actionError.value = "";
  try {
    await threads.activate(threadId);
    await router.push({ name: "diagnose" });
  } catch (err) {
    actionError.value = err.message;
  }
}

function evidenceLabel(value) {
  return evidenceLabels[value] ? runtime.t(evidenceLabels[value]) : value || runtime.t("common.none");
}

function reviewLabel(value) {
  return reviewLabels[value] ? runtime.t(reviewLabels[value]) : value || runtime.t("common.none");
}
</script>

<template>
  <section class="ledger">
    <header class="toolbar">
      <div>
        <h2>{{ project?.name || runtime.t("ledger.choose") }}</h2>
        <p>{{ runtime.t("ledger.records", { count: rows.length }) }}</p>
      </div>
      <div class="actions">
        <button type="button" :disabled="!canPromote" @click="promoteActiveThread">
          {{ runtime.t("ledger.promote") }}
        </button>
        <button type="button" disabled :title="runtime.t('ledger.reviewTitle')">
          {{ runtime.t("ledger.review") }}
        </button>
        <span>{{ runtime.t("ledger.selected", { count: selectedCount }) }}</span>
        <button type="button" disabled :title="runtime.t('ledger.reviewTitle')">
          {{ runtime.t("ledger.generate", { count: selectedCount }) }}
        </button>
      </div>
    </header>
    <p v-if="actionError" class="error">{{ actionError }}</p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th class="select-col">{{ runtime.t("ledger.select") }}</th>
            <th>{{ runtime.t("ledger.distress") }}</th>
            <th>{{ runtime.t("ledger.method") }}</th>
            <th>{{ runtime.t("ledger.location") }}</th>
            <th>{{ runtime.t("ledger.evidence") }}</th>
            <th>{{ runtime.t("ledger.reviewCol") }}</th>
            <th>{{ runtime.t("ledger.viewThread") }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-if="!rows.length">
            <td colspan="7" class="empty">{{ runtime.t("ledger.empty") }}</td>
          </tr>
          <tr v-for="row in rows" :key="row.record_id || row.source_thread_id">
            <td class="select-col">
              <input
                type="checkbox"
                :disabled="!row.record_id"
                :checked="row.record_id ? store.selected.has(row.record_id) : false"
                @change="store.toggleSelect(row.record_id)"
              />
            </td>
            <td>{{ row.defect_category || runtime.t("common.none") }}</td>
            <td>{{ row.chosen_method || runtime.t("common.pending") }}</td>
            <td>{{ row.location || runtime.t("common.none") }}</td>
            <td>
              <span :class="['evidence', row.evidence]">{{ evidenceLabel(row.evidence) }}</span>
            </td>
            <td>{{ reviewLabel(row.review) }}</td>
            <td>
              <button type="button" class="link" @click="viewThread(row.source_thread_id)">
                {{ runtime.t("ledger.viewThread") }}
              </button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>
</template>

<style scoped>
.ledger {
  display: flex;
  min-width: 0;
  flex: 1;
  flex-direction: column;
  overflow: hidden;
  background: var(--rd-bg);
}

.toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: center;
  justify-content: space-between;
  padding: 14px 16px;
  border-bottom: 1px solid var(--rd-border);
  background: var(--rd-surface);
}

h2,
p {
  margin: 0;
}

h2 {
  color: var(--rd-text-strong);
  font-size: 17px;
}

p,
.actions span,
.empty,
.error {
  color: var(--rd-text-dim);
  font-size: 12px;
}

.actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}

button {
  min-height: 32px;
  border: 1px solid var(--rd-border);
  border-radius: var(--rd-radius-control);
  background: var(--rd-surface-soft);
  color: var(--rd-text);
  cursor: pointer;
}

button:not(:disabled):hover {
  border-color: var(--rd-border-strong);
  color: var(--rd-accent);
}

button:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}

.table-wrap {
  overflow: auto;
  padding: 12px 16px 20px;
}

table {
  width: 100%;
  min-width: 860px;
  border-collapse: collapse;
  background: var(--rd-surface);
}

th,
td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--rd-border);
  text-align: left;
  vertical-align: middle;
}

th {
  color: var(--rd-text-dim);
  font-size: 12px;
  font-weight: 650;
}

td {
  color: var(--rd-text);
  font-size: 13px;
}

.select-col {
  width: 58px;
}

.evidence {
  display: inline-flex;
  padding: 2px 7px;
  border-radius: 999px;
  color: #fff;
  font-size: 12px;
}

.evidence.complete { background: var(--rd-success); }
.evidence.gap { background: var(--rd-warning); }
.evidence.missing { background: var(--rd-danger); }

.link {
  border: 0;
  background: transparent;
  color: var(--rd-accent);
  font-weight: 650;
}

.error {
  padding: 8px 16px 0;
  color: var(--rd-danger);
}
</style>
