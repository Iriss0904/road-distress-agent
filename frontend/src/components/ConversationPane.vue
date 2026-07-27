<script setup>
import { computed, nextTick, ref } from "vue";
import MessageBubble from "./MessageBubble.vue";
import { referenceMap } from "../lib/citations.js";
import { useRuntimeStore } from "../stores/runtime.js";
import { useRuntimeTraceStore } from "../stores/runtimeTrace.js";
import { useThreadsStore } from "../stores/threads.js";

const props = defineProps({
  userId: { type: String, default: "web_user_001" },
});
const emit = defineEmits(["trace"]);
const runtime = useRuntimeStore();
const store = useThreadsStore();
const runtimeTrace = useRuntimeTraceStore();
const text = ref("");
const streaming = ref(false);
const error = ref("");
const scrollEl = ref(null);
const streamedAnswer = ref(null);
const nextAnswerChunk = ref(0);
const safetyCompleted = ref(false);

const snapshot = computed(() => store.activeThread?.snapshot || {});
const transcript = computed(() => {
  const rows = visibleTranscript(store.activeThread?.transcript || [], snapshot.value);
  return streamedAnswer.value ? [...rows, streamedAnswer.value] : rows;
});

async function send() {
  const body = text.value.trim();
  if (!body || streaming.value) return;
  text.value = "";
  error.value = "";
  store.appendTranscript({ role: "user", text: body });
  await nextTick();
  scrollBottom();
  await postTurn(body);
}

async function postTurn(body) {
  resetProgressiveDelivery();
  streaming.value = true;
  try {
    const res = await fetch("/api/turn", { method: "POST", body: turnForm(body) });
    if (!res.ok) throw new Error(await responseError(res));
    if (!res.body) throw new Error("/api/turn did not return an SSE body");
    await consumeSSE(res.body);
  } catch (err) {
    error.value = err.message;
    store.appendTranscript({
      role: "assistant",
      text: runtime.t("conversation.errorPrefix", { message: err.message }),
    });
  } finally {
    streaming.value = false;
    await nextTick();
    scrollBottom();
  }
}

function turnForm(body) {
  const form = new FormData();
  form.append("text", body);
  form.append("user_id", props.userId);
  form.append("locale", runtime.locale);
  if (store.activeId) form.append("thread_id", store.activeId);
  return form;
}

async function responseError(res) {
  try {
    const payload = await res.json();
    return payload.detail || `${res.status} ${res.statusText}`;
  } catch {
    return `${res.status} ${res.statusText}`;
  }
}

async function consumeSSE(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    buffer = await drainSSEBuffer(buffer);
  }
}

async function drainSSEBuffer(buffer) {
  let next = buffer.indexOf("\n\n");
  while (next !== -1) {
    await handleSSEBlock(buffer.slice(0, next));
    buffer = buffer.slice(next + 2);
    next = buffer.indexOf("\n\n");
  }
  return buffer;
}

async function handleSSEBlock(raw) {
  const parsed = parseSSEBlock(raw);
  if (!parsed) return;
  runtimeTrace.observe(parsed.event, parsed.payload);
  if (parsed.event === "turn_start") store.activeId = parsed.payload.thread_id;
  if (parsed.event === "stage_complete") markStageComplete(parsed.payload);
  if (parsed.event === "answer_chunk") await appendAnswerChunk(parsed.payload);
  if (parsed.event === "snapshot") applySnapshot(parsed.payload);
  if (parsed.event === "error") throw new Error(errorPayloadText(parsed.payload));
  if (parsed.event === "soft_error") {
    store.appendTranscript({
      role: "assistant",
      text: runtime.t("conversation.warningPrefix", { message: errorPayloadText(parsed.payload) }),
    });
  }
}

function parseSSEBlock(raw) {
  let event = "message";
  let data = "";
  raw.split("\n").forEach((line) => {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) data += line.slice(5).trim();
  });
  if (!data) return null;
  return { event, payload: JSON.parse(data) };
}

function applySnapshot(payload) {
  reconcileStreamedAnswer(payload);
  store.applyTurnSnapshot(payload);
  snapshotMessages(payload).forEach(store.appendTranscript);
}

function markStageComplete(payload) {
  if (payload?.node === "safety_critic") safetyCompleted.value = true;
}

async function appendAnswerChunk(payload) {
  if (!safetyCompleted.value) {
    throw new Error("answer_chunk arrived before safety_critic completed");
  }
  if (typeof payload?.text !== "string") throw new TypeError("answer_chunk.text must be a string");
  if (payload.index !== nextAnswerChunk.value) {
    throw new Error(`answer_chunk index ${payload.index} arrived out of order`);
  }
  const text = `${streamedAnswer.value?.text || ""}${payload.text}`;
  streamedAnswer.value = { role: "assistant", text };
  nextAnswerChunk.value += 1;
  await nextTick();
  scrollBottom();
}

function reconcileStreamedAnswer(payload) {
  if (!streamedAnswer.value) return;
  if (payload.final_answer_message !== streamedAnswer.value.text) {
    throw new Error("streamed answer does not match authoritative snapshot");
  }
  streamedAnswer.value = null;
  nextAnswerChunk.value = 0;
}

function resetProgressiveDelivery() {
  streamedAnswer.value = null;
  nextAnswerChunk.value = 0;
  safetyCompleted.value = false;
}

function snapshotMessages(payload) {
  const messages = [];
  if (payload.scene_description) {
    messages.push({ role: "assistant", text: runtime.t("conversation.imageAnalysis", { message: payload.scene_description }) });
  }
  if (payload.direct_message) messages.push(withRefs(payload.direct_message, payload.reference_index));
  if (payload.final_answer_message) {
    messages.push(withRefs(payload.final_answer_message, payload.reference_index));
  }
  if (payload.awaiting_message) messages.push(withRefs(payload.awaiting_message, payload.candidate_reference_index));
  if (!messages.length && payload.interrupt?.prompt) messages.push({ role: "assistant", text: payload.interrupt.prompt });
  return messages;
}

function visibleTranscript(rows, snap) {
  const message = snap?.awaiting_message;
  if (!message || rows.some((row) => row.text === message)) return rows;
  return [...rows, withRefs(message, snap?.candidate_reference_index || [])];
}

function withRefs(message, refs = []) {
  const tokens = refs
    .map((ref) => `[[${ref.ref_id}]]`)
    .filter((token) => !String(message).includes(token));
  const text = tokens.length ? `${message} ${tokens.join(" ")}` : message;
  return { role: "assistant", text, references: refs };
}

function rowRefMap(row) {
  return referenceMap(row?.references || []);
}

function errorPayloadText(payload) {
  return payload?.message || payload?.detail || JSON.stringify(payload);
}

function scrollBottom() {
  if (scrollEl.value) scrollEl.value.scrollTop = scrollEl.value.scrollHeight;
}
</script>

<template>
  <section class="conversation">
    <div ref="scrollEl" class="messages">
      <p v-if="!transcript.length" class="empty">{{ runtime.t("conversation.empty") }}</p>
      <MessageBubble
        v-for="(entry, index) in transcript"
        :key="index"
        :message="entry"
        :ref-map="rowRefMap(entry)"
        @trace="emit('trace', $event)"
      />
    </div>
    <p v-if="error" class="error">{{ error }}</p>
    <form class="composer" @submit.prevent="send">
      <textarea
        v-model="text"
        :disabled="streaming"
        rows="2"
        :placeholder="runtime.t('conversation.placeholder')"
      />
      <button type="submit" :disabled="streaming || !text.trim()">
        {{ streaming ? runtime.t("conversation.sending") : runtime.t("conversation.send") }}
      </button>
    </form>
  </section>
</template>

<style scoped>
.conversation {
  display: flex;
  min-width: 0;
  flex: 1;
  flex-direction: column;
  background: var(--rd-bg);
}

.messages {
  display: flex;
  flex: 1;
  flex-direction: column;
  gap: 12px;
  overflow: auto;
  padding: 18px;
}

.empty,
.error {
  margin: 0;
  color: var(--rd-text-dim);
  font-size: 13px;
}

.error {
  padding: 0 18px;
  color: var(--rd-danger);
}

.composer {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 10px;
  padding: 12px;
  border-top: 1px solid var(--rd-border);
  background: var(--rd-surface);
}

textarea {
  min-height: 48px;
  max-height: 140px;
  resize: vertical;
  padding: 10px;
  border: 1px solid var(--rd-border);
  border-radius: var(--rd-radius-control);
}

button {
  min-width: 72px;
  border: 0;
  border-radius: var(--rd-radius-control);
  background: var(--rd-accent);
  color: #fff;
  cursor: pointer;
  font-weight: 650;
}

button:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}
</style>
