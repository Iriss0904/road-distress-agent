<script setup>
import { computed } from "vue";
import CitationChip from "./CitationChip.vue";
import { buildContentBlocks, splitRefTokens } from "../lib/citations.js";

const props = defineProps({
  message: { type: Object, default: null },
  text: { type: String, default: "" },
  role: { type: String, default: "" },
  refMap: { type: Object, default: () => ({}) },
});
const emit = defineEmits(["trace"]);

const displayRole = computed(() => props.role || props.message?.role || "assistant");
const blocks = computed(() => prepareBlocks(props.message?.text ?? props.text));

function prepareBlocks(text) {
  const shown = new Set();
  return buildContentBlocks(text).map((block) => {
    const parts = block.type === "table" ? [] : inlineParts(block.text, shown);
    const rows = block.type === "table" ? tableRows(block.rows, shown) : [];
    const refs = (block.refs || []).filter((refId) => markRef(refId, shown));
    return { ...block, parts, rows, refs };
  });
}

function tableRows(rows, shown) {
  return rows.map((row) => row.map((cell) => inlineParts(cell, shown)));
}

function inlineParts(text, shown) {
  return splitRefTokens(text).flatMap((part) => {
    if (part.refId) return markRef(part.refId, shown) ? [{ type: "ref", value: part.refId }] : [];
    return boldParts(part.text);
  });
}

function boldParts(text) {
  return String(text || "")
    .split(/(\*\*[^*]+?\*\*)/g)
    .filter(Boolean)
    .map((part) => {
      const isStrong = part.startsWith("**") && part.endsWith("**");
      return { type: isStrong ? "strong" : "text", value: cleanBold(part) };
    });
}

function cleanBold(text) {
  return String(text || "").replace(/^\*\*|\*\*$/g, "").replace(/\*\*/g, "");
}

function markRef(refId, shown) {
  if (!refId || shown.has(refId)) return false;
  shown.add(refId);
  return true;
}
</script>

<template>
  <article class="bubble" :class="displayRole">
    <div v-for="(block, index) in blocks" :key="index" :class="['block', block.type]">
      <table v-if="block.type === 'table'">
        <tr v-for="(row, rowIndex) in block.rows" :key="rowIndex">
          <component :is="rowIndex === 0 ? 'th' : 'td'" v-for="(cell, cellIndex) in row" :key="cellIndex">
            <template v-for="(part, partIndex) in cell" :key="partIndex">
              <CitationChip
                v-if="part.type === 'ref'"
                :ref-id="part.value"
                :ref-data="refMap[part.value]"
                @trace="emit('trace', $event)"
              />
              <strong v-else-if="part.type === 'strong'">{{ part.value }}</strong>
              <template v-else>{{ part.value }}</template>
            </template>
          </component>
        </tr>
      </table>
      <template v-else>
        <template v-for="(part, partIndex) in block.parts" :key="partIndex">
          <CitationChip
            v-if="part.type === 'ref'"
            :ref-id="part.value"
            :ref-data="refMap[part.value]"
            @trace="emit('trace', $event)"
          />
          <strong v-else-if="part.type === 'strong'">{{ part.value }}</strong>
          <template v-else>{{ part.value }}</template>
        </template>
      </template>
      <CitationChip
        v-for="refId in block.refs"
        :key="refId"
        :ref-id="refId"
        :ref-data="refMap[refId]"
        @trace="emit('trace', $event)"
      />
    </div>
  </article>
</template>

<style scoped>
.bubble {
  width: fit-content;
  max-width: min(760px, 86%);
  padding: 12px 14px;
  border: 1px solid var(--rd-border);
  border-radius: var(--rd-radius);
  background: var(--rd-surface);
  box-shadow: 0 4px 16px rgba(31, 38, 70, 0.04);
  line-height: 1.68;
  white-space: pre-wrap;
}

.bubble.user {
  align-self: flex-end;
  border-color: #dfe5ff;
  background: #f4f6ff;
}

.block + .block {
  margin-top: 8px;
}

.heading {
  color: var(--rd-text-strong);
  font-weight: 700;
}

table {
  width: 100%;
  border-collapse: collapse;
  white-space: normal;
}

th,
td {
  padding: 7px 8px;
  border: 1px solid var(--rd-border);
  text-align: left;
  vertical-align: top;
}

th {
  background: var(--rd-surface-soft);
  color: var(--rd-text-strong);
}
</style>
