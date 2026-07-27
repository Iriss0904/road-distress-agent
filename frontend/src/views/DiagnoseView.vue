<script setup>
import { computed, ref } from "vue";
import ChatList from "../components/ChatList.vue";
import ConversationPane from "../components/ConversationPane.vue";
import SmartPanel from "../components/SmartPanel.vue";
import { useRuntimeStore } from "../stores/runtime.js";
import { useThreadsStore } from "../stores/threads.js";

const MIN_SOURCE_WIDTH = 320;
const MAX_SOURCE_WIDTH = 820;
const MIN_CHAT_WIDTH = 420;

const runtime = useRuntimeStore();
const store = useThreadsStore();
const smartPanel = ref(null);
const root = ref(null);
const listOpen = ref(true);
const sourceWidth = ref(520);
const snapshot = computed(() => store.activeThread?.snapshot || {});

function trace(payload) {
  smartPanel.value?.trace(payload);
}

function startResize(event) {
  event.preventDefault();
  window.addEventListener("pointermove", resizeSource);
  window.addEventListener("pointerup", stopResize, { once: true });
}

function resizeSource(event) {
  if (!root.value) return;
  const rect = root.value.getBoundingClientRect();
  sourceWidth.value = clampSourceWidth(rect.right - event.clientX, rect.width);
}

function stopResize() {
  window.removeEventListener("pointermove", resizeSource);
}

function clampSourceWidth(width, totalWidth) {
  const max = Math.max(MIN_SOURCE_WIDTH, Math.min(MAX_SOURCE_WIDTH, totalWidth - MIN_CHAT_WIDTH));
  return Math.max(MIN_SOURCE_WIDTH, Math.min(max, width));
}
</script>

<template>
  <section ref="root" class="diagnose">
    <button
      type="button"
      class="collapse"
      :aria-expanded="listOpen"
      @click="listOpen = !listOpen"
    >
      {{ listOpen ? runtime.t("diagnose.collapse") : runtime.t("diagnose.expand") }}
    </button>
    <ChatList v-if="listOpen" :user-id="runtime.userId" />
    <ConversationPane :user-id="runtime.userId" @trace="trace" />
    <button
      type="button"
      class="source-resizer"
      :aria-label="runtime.t('panel.resize')"
      @pointerdown="startResize"
    />
    <SmartPanel
      ref="smartPanel"
      :snapshot="snapshot"
      :style="{ width: `${sourceWidth}px` }"
    />
  </section>
</template>

<style scoped>
.diagnose {
  position: relative;
  display: flex;
  height: calc(100vh - 48px);
  min-height: 560px;
  overflow: hidden;
}

.collapse {
  position: absolute;
  z-index: 4;
  top: 10px;
  left: 10px;
  min-height: 30px;
  padding: 0 9px;
  border: 1px solid var(--rd-border);
  border-radius: var(--rd-radius-control);
  background: rgba(255, 255, 255, 0.9);
  color: var(--rd-text-dim);
  cursor: pointer;
  font-size: 12px;
}

.collapse + :deep(.chat-list) {
  padding-top: 48px;
}

.source-resizer {
  width: 8px;
  flex: 0 0 8px;
  border: 0;
  border-right: 1px solid var(--rd-border);
  border-left: 1px solid transparent;
  background: transparent;
  cursor: col-resize;
}

.source-resizer:hover,
.source-resizer:focus-visible {
  border-left-color: #cfd7ff;
  background: var(--rd-accent-soft);
  outline: none;
}
</style>
