<script setup>
import { computed } from "vue";
import { citationMeta } from "../lib/citations.js";
import { useRuntimeStore } from "../stores/runtime.js";

const props = defineProps({
  refId: { type: String, required: true },
  refData: { type: Object, default: null },
});
const emit = defineEmits(["trace"]);
const runtime = useRuntimeStore();
const meta = computed(() => (props.refData ? citationMeta(props.refData) : ""));
</script>

<template>
  <span class="wrap">
    <button type="button" class="chip" @click="emit('trace', { refId, refData })">
      {{ refId }}
    </button>
    <span v-if="refData" class="popover">
      <strong>{{ refData.title || runtime.t("citation.untitled") }}</strong>
      <span v-if="meta">{{ meta }}</span>
      <span v-if="refData.snippet">{{ refData.snippet }}</span>
    </span>
  </span>
</template>

<style scoped>
.wrap {
  position: relative;
  display: inline-flex;
}

.chip {
  display: inline-flex;
  align-items: center;
  min-height: 22px;
  margin: 0 2px;
  padding: 0 7px;
  border: 1px solid #cfd7ff;
  border-radius: 999px;
  background: var(--rd-accent-soft);
  color: var(--rd-accent);
  cursor: pointer;
  font-size: 12px;
  font-weight: 650;
}

.chip:hover,
.chip:focus-visible {
  border-color: var(--rd-accent);
  outline: none;
}

.popover {
  position: absolute;
  z-index: 5;
  bottom: calc(100% + 8px);
  left: 0;
  display: none;
  width: min(320px, 80vw);
  padding: 10px;
  border: 1px solid var(--rd-border);
  border-radius: var(--rd-radius-control);
  background: var(--rd-surface);
  box-shadow: var(--rd-shadow-soft);
  color: var(--rd-text);
  font-size: 12px;
  line-height: 1.5;
}

.popover span,
.popover strong {
  display: block;
}

.wrap:hover .popover,
.chip:focus-visible + .popover {
  display: block;
}
</style>
