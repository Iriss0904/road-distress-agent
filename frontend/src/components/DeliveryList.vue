<script setup>
import { computed } from "vue";
import { useRuntimeStore } from "../stores/runtime.js";
import { useDeliveriesStore } from "../stores/deliveries.js";

const runtime = useRuntimeStore();
const store = useDeliveriesStore();

const rows = computed(() => store.items);

function open(row) {
  store.open(row.delivery_id);
}

function statusLabel(status) {
  return { draft: runtime.t("delivery.draft"), final: runtime.t("delivery.final") }[status]
    || status || runtime.t("common.none");
}

function versionCount(row) {
  return runtime.t("delivery.versionCount", { count: row.latest_version_no || 0 });
}

function shortDate(value) {
  if (!value) return "";
  return value.slice(0, 10);
}
</script>

<template>
  <aside class="delivery-list">
    <header>
      <h2>{{ runtime.t("delivery.title") }}</h2>
      <span>{{ runtime.t("delivery.count", { count: rows.length }) }}</span>
    </header>
    <p v-if="store.error" class="error">{{ store.error }}</p>
    <p v-if="store.loading" class="empty">{{ runtime.t("delivery.loading") }}</p>
    <button
      v-for="row in rows"
      :key="row.delivery_id"
      type="button"
      class="delivery-row"
      :class="{ active: store.activeId === row.delivery_id }"
      @click="open(row)"
    >
      <span class="row-head">
        <span class="title">{{ row.title || row.delivery_id }}</span>
        <span :class="['status', row.status]">{{ statusLabel(row.status) }}</span>
      </span>
      <span class="meta">
        <span>{{ row.project_id }}</span>
        <span>{{ versionCount(row) }}</span>
      </span>
      <span class="id-line">
        <span>{{ row.delivery_id }}</span>
        <span>{{ shortDate(row.created_at) }}</span>
      </span>
    </button>
  </aside>
</template>

<style scoped>
.delivery-list {
  display: flex;
  width: 300px;
  min-width: 280px;
  flex-direction: column;
  gap: 8px;
  overflow: auto;
  padding: 14px;
  border-right: 1px solid var(--rd-border);
  background: var(--rd-surface);
}

header,
.row-head,
.meta,
.id-line {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

header {
  padding: 2px 2px 8px;
}

h2 {
  margin: 0;
  color: var(--rd-text-strong);
  font-size: 15px;
}

header span,
.meta,
.id-line,
.empty,
.error {
  color: var(--rd-text-dim);
  font-size: 12px;
}

.delivery-row {
  display: grid;
  gap: 7px;
  padding: 10px;
  border: 1px solid transparent;
  border-radius: var(--rd-radius-control);
  background: transparent;
  color: var(--rd-text);
  cursor: pointer;
  text-align: left;
}

.delivery-row:hover,
.delivery-row.active,
.delivery-row:focus-visible {
  border-color: var(--rd-border);
  background: var(--rd-surface-soft);
  outline: none;
}

.title {
  min-width: 0;
  overflow: hidden;
  color: var(--rd-text-strong);
  font-weight: 650;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.status {
  flex: 0 0 auto;
  padding: 2px 7px;
  border-radius: 999px;
  color: #fff;
  font-size: 12px;
}

.status.draft {
  background: var(--rd-st-draft);
}

.status.final {
  background: var(--rd-st-deliv);
}

.error {
  color: var(--rd-danger);
}
</style>
