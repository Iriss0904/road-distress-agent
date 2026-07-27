<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import { useRuntimeStore } from "../stores/runtime.js";
import { useThreadsStore } from "../stores/threads.js";

const props = defineProps({
  userId: { type: String, default: "web_user_001" },
});
const emit = defineEmits(["new-thread"]);
const runtime = useRuntimeStore();
const store = useThreadsStore();
const q = ref("");
const filter = ref("all");
const editingId = ref("");
const draftTitle = ref("");
const actionError = ref("");
let timer = null;

const statusKeys = ["draft", "ready", "promoted", "delivered"];
const labels = computed(() =>
  Object.fromEntries(statusKeys.map((status) => [status, runtime.t(`status.${status}`)])),
);
const filters = computed(() => [
  { status: "all", label: runtime.t("status.all") },
  ...statusKeys.map(statusFilter),
]);
const visibleGroups = computed(() => {
  if (filter.value === "all") return store.groups;
  return store.groups.filter((group) => group.status === filter.value);
});

onMounted(() => reload());
onUnmounted(() => clearTimeout(timer));
watch(() => props.userId, () => reload());

function statusFilter(status) {
  return { status, label: labels.value[status] };
}

function delayedReload() {
  clearTimeout(timer);
  timer = setTimeout(reload, 220);
}

async function reload() {
  actionError.value = "";
  try {
    await store.load(props.userId, q.value.trim());
  } catch (err) {
    actionError.value = err.message;
  }
}

function newThread() {
  store.startNew();
  emit("new-thread");
}

function beginRename(thread) {
  editingId.value = thread.thread_id;
  draftTitle.value = thread.title || thread.thread_id;
}

async function finishRename(thread) {
  const next = draftTitle.value.trim();
  editingId.value = "";
  if (!next || next === thread.title) return;
  try {
    await store.rename(thread.thread_id, next);
  } catch (err) {
    actionError.value = err.message;
  }
}

async function removeThread(thread) {
  const title = thread.title || thread.thread_id;
  if (!window.confirm(runtime.t("chat.delete", { title }))) return;
  try {
    await store.remove(thread.thread_id);
  } catch (err) {
    actionError.value = err.message;
  }
}
</script>

<template>
  <aside class="chat-list">
    <div class="bar">
      <button type="button" class="new" @click="newThread">{{ runtime.t("chat.new") }}</button>
      <input
        v-model="q"
        type="search"
        :placeholder="runtime.t('chat.search')"
        @input="delayedReload"
      />
    </div>
    <div class="filters" :aria-label="runtime.t('chat.filter')">
      <button
        v-for="item in filters"
        :key="item.status"
        type="button"
        :class="{ active: filter === item.status }"
        @click="filter = item.status"
      >
        {{ item.label }}
      </button>
    </div>
    <p v-if="store.error || actionError" class="error">{{ actionError || store.error }}</p>
    <p v-if="!visibleGroups.length && !store.loading" class="empty">{{ runtime.t("chat.empty") }}</p>
    <div v-for="group in visibleGroups" :key="group.status" class="group">
      <h2>{{ labels[group.status] }} <span>{{ group.items.length }}</span></h2>
      <div
        v-for="thread in group.items"
        :key="thread.thread_id"
        class="row"
        :class="{ active: store.activeId === thread.thread_id }"
        role="button"
        tabindex="0"
        @click="store.activate(thread.thread_id)"
        @keydown.enter.prevent="store.activate(thread.thread_id)"
        @keydown.space.prevent="store.activate(thread.thread_id)"
      >
        <span class="main">
          <input
            v-if="editingId === thread.thread_id"
            v-model="draftTitle"
            class="rename"
            @click.stop
            @keydown.enter.prevent="finishRename(thread)"
            @keydown.esc.prevent="editingId = ''"
            @blur="finishRename(thread)"
          />
          <span v-else class="title" @dblclick.stop="beginRename(thread)">
            {{ thread.title || thread.thread_id }}
          </span>
          <span class="meta">{{ thread.updated_at || thread.thread_id }}</span>
        </span>
        <span :class="['badge', group.status]">{{ labels[group.status] }}</span>
        <span class="actions">
          <button type="button" :title="runtime.t('chat.rename')" @click.stop="beginRename(thread)">✎</button>
          <button type="button" :title="runtime.t('chat.remove')" @click.stop="removeThread(thread)">×</button>
        </span>
      </div>
    </div>
  </aside>
</template>

<style scoped>
.chat-list {
  display: flex;
  width: 292px;
  min-width: 292px;
  flex-direction: column;
  gap: 12px;
  overflow: auto;
  padding: 12px;
  border-right: 1px solid var(--rd-border);
  background: var(--rd-surface);
}

.bar {
  display: grid;
  gap: 8px;
}

.new,
.filters button,
.actions button {
  border: 1px solid var(--rd-border);
  border-radius: var(--rd-radius-control);
  background: var(--rd-surface-soft);
  color: var(--rd-text);
  cursor: pointer;
}

.new {
  min-height: 36px;
  color: var(--rd-accent);
  font-weight: 650;
}

input {
  width: 100%;
  min-height: 34px;
  padding: 0 10px;
  border: 1px solid var(--rd-border);
  border-radius: var(--rd-radius-control);
  background: var(--rd-bg);
}

.filters {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.filters button {
  padding: 5px 8px;
  font-size: 12px;
}

.filters button.active {
  border-color: #cfd7ff;
  background: var(--rd-accent-soft);
  color: var(--rd-accent);
}

.group h2 {
  margin: 12px 0 6px;
  color: var(--rd-text-dim);
  font-size: 12px;
  font-weight: 650;
}

.row {
  position: relative;
  display: grid;
  width: 100%;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 8px;
  padding: 10px;
  border: 1px solid transparent;
  border-radius: var(--rd-radius-control);
  background: transparent;
  cursor: pointer;
  text-align: left;
}

.row:hover,
.row.active,
.row:focus-visible {
  border-color: var(--rd-border);
  background: var(--rd-surface-soft);
  outline: none;
}

.main,
.title,
.meta {
  min-width: 0;
}

.title,
.meta {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.title {
  color: var(--rd-text-strong);
  font-weight: 650;
}

.meta,
.empty,
.error {
  color: var(--rd-text-dim);
  font-size: 12px;
}

.badge {
  align-self: start;
  padding: 2px 6px;
  border-radius: 999px;
  color: #fff;
  font-size: 11px;
}

.badge.draft { background: var(--rd-st-draft); }
.badge.ready { background: var(--rd-st-todo); }
.badge.promoted { background: var(--rd-st-in); }
.badge.delivered { background: var(--rd-st-deliv); }

.actions {
  position: absolute;
  right: 8px;
  bottom: 6px;
  display: none;
  gap: 4px;
}

.row:hover .actions {
  display: flex;
}

.actions button {
  width: 24px;
  height: 24px;
}

.rename {
  min-height: 28px;
}

.error {
  color: var(--rd-danger);
}
</style>
