<script setup>
import { computed, onMounted } from "vue";
import { useRoute } from "vue-router";
import { useRuntimeStore } from "../stores/runtime.js";
import { useRuntimeTraceStore } from "../stores/runtimeTrace.js";
import { useThreadsStore } from "../stores/threads.js";

const route = useRoute();
const runtime = useRuntimeStore();
const trace = useRuntimeTraceStore();
const threads = useThreadsStore();
const title = computed(() => runtime.t(route.meta?.labelKey ?? ""));

onMounted(() => runtime.load());

function changeRole(event) {
  runtime.setUserId(event.target.value);
  threads.startNew();
}

function toggleLocale() {
  runtime.toggleLocale();
  threads.startNew();
}
</script>

<template>
  <header class="top">
    <span class="title" data-testid="view-title">{{ title }}</span>
    <span class="spacer" />
    <label class="role-select">
      <span>{{ runtime.t("role.label") }}</span>
      <select :value="runtime.userId" data-testid="role-select" @change="changeRole">
        <option v-for="role in runtime.roles" :key="role.id" :value="role.id">
          {{ role.label }}
        </option>
      </select>
    </label>
    <span class="pill" data-testid="runtime">
      <i class="dot" aria-hidden="true" />{{ runtime.label }}
    </span>
    <button type="button" class="pill action" data-testid="runtime-log" @click="runtime.toggleLog">
      {{ runtime.t("runtime.log") }} · {{ trace.eventCount }}
    </button>
    <button type="button" class="pill action" data-testid="locale-toggle" @click="toggleLocale">
      {{ runtime.t("runtime.locale") }}
    </button>
  </header>
</template>

<style scoped>
.top {
  display: flex;
  height: 48px;
  flex: none;
  align-items: center;
  gap: 12px;
  padding: 0 18px;
  border-bottom: 1px solid var(--rd-border);
  background: rgba(255, 255, 255, 0.86);
  backdrop-filter: blur(10px);
}

.title {
  color: var(--rd-text);
  font-weight: 650;
}

.spacer {
  flex: 1;
}

.role-select,
.pill {
  border-radius: 999px;
  font-size: 11px;
  line-height: 1.3;
  white-space: nowrap;
}

.role-select {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 3px 8px;
  border: 1px solid #dfe5ff;
  background: var(--rd-accent-soft);
  color: var(--rd-accent);
}

.role-select select {
  max-width: 190px;
  border: 0;
  background: transparent;
  color: inherit;
  cursor: pointer;
  font: inherit;
  font-weight: 650;
}

.pill {
  padding: 3px 9px;
  border: 1px solid var(--rd-border);
  background: var(--rd-surface);
  color: var(--rd-text-dim);
}

.action {
  cursor: pointer;
}

.action:hover,
.action:focus-visible {
  border-color: var(--rd-border-strong);
  color: var(--rd-accent);
  outline: none;
}

.dot {
  display: inline-block;
  width: 7px;
  height: 7px;
  margin-right: 5px;
  border-radius: 50%;
  background: var(--rd-success);
}
</style>
