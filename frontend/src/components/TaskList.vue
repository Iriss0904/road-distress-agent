<script setup>
import { computed } from "vue";
import { useRuntimeStore } from "../stores/runtime.js";
import { UNFILED_PROJECT_ID, useTasksStore } from "../stores/tasks.js";

const runtime = useRuntimeStore();
const store = useTasksStore();

const taskRows = computed(() => [
  {
    project_id: UNFILED_PROJECT_ID,
    name: runtime.t("tasks.unfiled"),
    rows: [],
    unfiledCount: store.unfiled.length,
    synthetic: true,
  },
  ...store.projects,
]);

function progress(project) {
  if (project.synthetic) return { reviewed: 0, total: project.unfiledCount || 0 };
  const rows = project.rows || [];
  const reviewed = rows.filter((row) => row.review && row.review !== "active").length;
  return { reviewed, total: rows.length };
}

function open(project) {
  if (project.synthetic) {
    store.openUnfiled();
    return;
  }
  store.openLedger(project.project_id);
}
</script>

<template>
  <aside class="task-list">
    <header>
      <h2>{{ runtime.t("tasks.title") }}</h2>
      <span>{{ runtime.t("tasks.count", { count: store.projects.length }) }}</span>
    </header>
    <p v-if="store.error" class="error">{{ store.error }}</p>
    <p v-if="store.loading" class="empty">{{ runtime.t("tasks.loading") }}</p>
    <button
      v-for="project in taskRows"
      :key="project.project_id"
      type="button"
      class="task-row"
      :class="{ active: store.activeProjectId === project.project_id }"
      @click="open(project)"
    >
      <span class="title">{{ project.name || project.project_id }}</span>
      <span class="meta">
        <span>{{ runtime.t("tasks.progress", { done: progress(project).reviewed, total: progress(project).total }) }}</span>
        <span>{{ runtime.t("tasks.distressCount", { count: progress(project).total }) }}</span>
      </span>
    </button>
  </aside>
</template>

<style scoped>
.task-list {
  display: flex;
  width: 280px;
  min-width: 260px;
  flex-direction: column;
  gap: 8px;
  overflow: auto;
  padding: 14px;
  border-right: 1px solid var(--rd-border);
  background: var(--rd-surface);
}

header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 2px 2px 8px;
}

h2 {
  margin: 0;
  color: var(--rd-text-strong);
  font-size: 15px;
}

header span,
.meta,
.empty,
.error {
  color: var(--rd-text-dim);
  font-size: 12px;
}

.task-row {
  display: grid;
  gap: 6px;
  padding: 10px;
  border: 1px solid transparent;
  border-radius: var(--rd-radius-control);
  background: transparent;
  color: var(--rd-text);
  cursor: pointer;
  text-align: left;
}

.task-row:hover,
.task-row.active,
.task-row:focus-visible {
  border-color: var(--rd-border);
  background: var(--rd-surface-soft);
  outline: none;
}

.title {
  overflow: hidden;
  color: var(--rd-text-strong);
  font-weight: 650;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.meta {
  display: flex;
  justify-content: space-between;
  gap: 8px;
}

.error {
  color: var(--rd-danger);
}
</style>
