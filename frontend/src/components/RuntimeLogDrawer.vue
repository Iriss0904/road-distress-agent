<script setup>
import { computed } from "vue";
import { useRuntimeStore } from "../stores/runtime.js";
import { useRuntimeTraceStore } from "../stores/runtimeTrace.js";
import "./runtime-log.css";

const TOKEN_CHARS = 4;
const WARN_DURATION_MS = 1_000;
const SLOW_DURATION_MS = 3_000;
const KIND_KEYS = {
  error: "kindError",
  llm_call: "kindLlm",
  rerank: "kindRerank",
  retrieval: "kindRetrieval",
  user_input: "kindUserInput",
  user_selection: "kindSelection",
  weather_location_request: "kindWeatherLocation",
};
const ROLE_KEYS = {
  ai: "roleAssistant",
  assistant: "roleAssistant",
  human: "roleUser",
  system: "roleSystem",
  tool: "roleTool",
  user: "roleUser",
};

const runtime = useRuntimeStore();
const trace = useRuntimeTraceStore();
const turns = computed(() => trace.turns);

function close() {
  runtime.logOpen = false;
}

function kindLabel(kind) {
  const key = KIND_KEYS[kind];
  return key ? runtime.t(key) : kind || "trace";
}

function roleLabel(role) {
  return runtime.t(ROLE_KEYS[String(role || "").toLowerCase()] || "roleMessage");
}

function durationClass(durationMs) {
  if (durationMs === null || durationMs === undefined) return "";
  if (durationMs >= SLOW_DURATION_MS) return "slow";
  return durationMs >= WARN_DURATION_MS ? "warn" : "ok";
}

function durationText(durationMs) {
  if (durationMs === null || durationMs === undefined) return runtime.t("traceDurationPending");
  return `${durationMs} ms`;
}

function sections(event) {
  return [
    jsonSection(runtime.t("input"), event.input),
    promptSection(event.prompt),
    jsonSection(runtime.t("retrieval"), event.retrieval),
    jsonSection(runtime.t("output"), event.output),
    jsonSection(runtime.t("stateDelta"), event.state_delta),
    jsonSection(runtime.t("metadata"), event.metadata),
  ].filter(Boolean);
}

function jsonSection(title, value) {
  if (value === undefined || value === null) return null;
  return { title, type: "json", text: JSON.stringify(value, null, 2) };
}

function promptSection(messages) {
  if (!Array.isArray(messages) || !messages.length) return null;
  return { title: "Prompt", type: "prompt", messages };
}

function contentText(content) {
  return typeof content === "string" ? content : JSON.stringify(content, null, 2);
}

function tokenText(content) {
  return `${runtime.t("traceEstimatedTokens")}: ${Math.ceil(contentText(content).length / TOKEN_CHARS)}`;
}

function timeText(timestamp, locale) {
  if (!timestamp) return "";
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleTimeString(locale || runtime.locale, { hour12: false });
}
</script>

<template>
  <aside
    v-if="runtime.logOpen"
    class="debug-trace-panel open"
    :aria-label="runtime.t('runtime.log')"
  >
    <div class="debug-trace-head">
      <strong>{{ runtime.t("runtimeChain") }}</strong>
      <span class="debug-trace-count">{{ trace.eventCount }}</span>
      <div class="debug-trace-actions">
        <button type="button" @click="trace.clear">{{ runtime.t("clear") }}</button>
        <button type="button" @click="close">{{ runtime.t("close") }}</button>
      </div>
    </div>

    <div v-if="!turns.length" class="debug-trace-empty">{{ runtime.t("noTrace") }}</div>
    <div class="debug-trace-body">
      <details v-for="turn in turns" :key="turn.id" open class="debug-trace-turn">
        <summary>
          <span class="debug-trace-turn-title">{{ runtime.t("runtimeChain") }}</span>
          <span class="debug-trace-meta">{{ turn.threadId || "thread" }}</span>
          <span :class="['debug-trace-duration', durationClass(turn.summary.durationMs)]">
            {{ durationText(turn.summary.durationMs) }}
          </span>
          <span v-if="turn.errors.length" class="debug-trace-error-chip">
            {{ turn.errors.length }} {{ runtime.t("kindError") }}
          </span>
        </summary>

        <div class="debug-trace-errors">
          <details
            v-for="error in turn.errors"
            :key="`error-${error.seq}`"
            open
            class="debug-trace-detail kind-error"
          >
            <summary>
              <span class="debug-trace-kind kind-error">{{ runtime.t("kindError") }}</span>
              <span class="debug-trace-detail-title">{{ error.title }}</span>
              <span class="debug-trace-time">{{ timeText(error.timestamp, turn.locale) }}</span>
            </summary>
          </details>
        </div>

        <div class="debug-trace-layers">
          <details v-for="layer in turn.layers" :key="layer.key" open class="debug-trace-layer">
            <summary>
              <span class="debug-trace-layer-title">{{ runtime.t(layer.labelKey) }}</span>
              <span class="debug-trace-meta">
                {{ layer.nodes.length }} {{ runtime.t("traceNodes") }}
              </span>
            </summary>

            <details v-for="run in layer.nodes" :key="run.id" class="debug-trace-node">
              <summary>
                <span class="debug-trace-node-label">{{ run.label || run.node }}</span>
                <code class="debug-trace-node-name">{{ run.node }}</code>
                <span class="debug-trace-kinds">
                  <span
                    v-for="kind in Object.keys(run.kinds)"
                    :key="kind"
                    :class="['debug-trace-kind', `kind-${kind}`]"
                  >
                    {{ kindLabel(kind) }}
                  </span>
                </span>
                <span :class="['debug-trace-duration', durationClass(run.durationMs)]">
                  {{ durationText(run.durationMs) }}
                </span>
                <span class="debug-trace-meta">
                  {{ run.details.length }} {{ runtime.t("traceDetails") }}
                </span>
              </summary>

              <div class="debug-trace-details">
                <p v-if="!run.details.length" class="debug-trace-muted">
                  {{ runtime.t("traceNoDetails") }}
                </p>
                <details
                  v-for="event in run.details"
                  :key="event.seq"
                  open
                  :class="['debug-trace-detail', `kind-${event.kind || 'trace'}`]"
                >
                  <summary>
                    <span :class="['debug-trace-kind', `kind-${event.kind || 'trace'}`]">
                      {{ kindLabel(event.kind) }}
                    </span>
                    <span class="debug-trace-detail-title">
                      {{ event.title || event.node || "trace" }}
                    </span>
                    <span class="debug-trace-time">{{ timeText(event.timestamp, turn.locale) }}</span>
                  </summary>

                  <div class="debug-trace-section-list">
                    <section
                      v-for="section in sections(event)"
                      :key="section.title"
                      class="debug-trace-section"
                    >
                      <h4>{{ section.title }}</h4>
                      <template v-if="section.type === 'prompt'">
                        <details
                          v-for="(message, index) in section.messages"
                          :key="index"
                          :open="String(message.role || '').toLowerCase() !== 'system'"
                          :class="['debug-trace-message', `role-${String(message.role || 'message').toLowerCase()}`]"
                        >
                          <summary>
                            <span :class="['debug-trace-role', `role-${String(message.role || 'message').toLowerCase()}`]">
                              {{ roleLabel(message.role) }}
                            </span>
                            <span class="debug-trace-message-size">
                              {{ tokenText(message.content) }}
                            </span>
                          </summary>
                          <pre>{{ contentText(message.content) }}</pre>
                        </details>
                      </template>
                      <pre v-else>{{ section.text }}</pre>
                    </section>
                  </div>
                </details>
              </div>
            </details>
          </details>
        </div>
      </details>
    </div>
  </aside>
</template>
