import { defineStore } from "pinia";
import { computed, ref } from "vue";
import { layerForNode } from "../lib/runtimeTraceLayers.js";

const NODE_TIMING_KIND = "node_timing";
const TURN_TIMING_KIND = "turn_timing";

export const useRuntimeTraceStore = defineStore("runtimeTrace", () => {
  const turns = ref([]);
  const currentTurn = ref(null);
  const lastSnapshotTraceCount = ref(0);
  const nextSeq = ref(1);
  const eventCount = computed(() => turns.value.reduce((total, turn) => total + turnDetails(turn), 0));

  function observe(event, payload) {
    if (event === "turn_start") return startTurn(payload);
    if (event === "stage") return recordStage(payload);
    if (event === "stage_complete") return completeStage(payload);
    if (event === "trace") return recordTrace(payload);
    if (event === "snapshot") return rebuildFromSnapshot(payload);
    if (event === "error" || event === "soft_error") return recordError(event, payload);
  }

  function clear() {
    turns.value = [];
    currentTurn.value = null;
    lastSnapshotTraceCount.value = 0;
    nextSeq.value = 1;
  }

  function startTurn(payload = {}) {
    const turn = newTurn(payload);
    turn.snapshotOffset = lastSnapshotTraceCount.value;
    turns.value.push(turn);
    currentTurn.value = turn;
  }

  function recordStage(payload = {}) {
    const turn = ensureTurn(payload);
    const node = payload.node || "runtime";
    turn.labelByNode[node] = payload.label || node;
    createNodeRun(turn, {
      implicit: false,
      label: turn.labelByNode[node],
      node,
    });
  }

  function completeStage(payload = {}) {
    const turn = ensureTurn(payload);
    const node = payload.node || "runtime";
    const run = latestRun(turn, node) || createImplicitRun(turn, node);
    if (payload.label) {
      turn.labelByNode[node] = payload.label;
      run.label = payload.label;
    }
    run.closed = true;
  }

  function recordTrace(trace = {}) {
    const event = { seq: nextSeq.value++, ...trace };
    const turn = ensureTurn({ thread_id: event.thread_id });
    if (event.kind === NODE_TIMING_KIND) return attachDuration(turn, event);
    if (event.kind === TURN_TIMING_KIND) return recordTurnDuration(turn, event);
    appendDetail(turn, event);
  }

  function rebuildFromSnapshot(payload = {}) {
    const source = Array.isArray(payload.trace) ? payload.trace : [];
    const previous = ensureTurn(payload);
    const traces = source.slice(previous.snapshotOffset || 0);
    const rebuilt = newTurn({
      locale: previous.locale,
      thread_id: payload.thread_id || previous.threadId,
    });
    rebuilt.startedAt = previous.startedAt;
    rebuilt.snapshotOffset = source.length;
    rebuilt.labelByNode = { ...previous.labelByNode };
    traces.forEach((trace) => recordSnapshotTrace(rebuilt, trace));
    replaceCurrentTurn(rebuilt);
    lastSnapshotTraceCount.value = source.length;
  }

  function recordSnapshotTrace(turn, trace) {
    const event = { seq: trace.sequence || nextSeq.value++, ...trace };
    if (event.kind === NODE_TIMING_KIND) return attachDuration(turn, event);
    if (event.kind === TURN_TIMING_KIND) return recordTurnDuration(turn, event);
    appendDetail(turn, event);
  }

  function recordError(_event, payload = {}) {
    const turn = ensureTurn({});
    const node = payload.likely_next_node || payload.node || payload.last_completed_node || "server";
    const event = {
      seq: nextSeq.value++,
      kind: "error",
      node,
      output: payload,
      title: payload.title || `Stream error: ${payload.message || "unknown"}`,
    };
    const label = payload.likely_next_label || payload.last_completed_label;
    if (label) turn.labelByNode[node] = label;
    turn.errors.push(event);
    appendDetail(turn, event);
  }

  function appendDetail(turn, event) {
    const node = event.node || "runtime";
    const run = writableRun(turn, node);
    run.details.push(event);
    run.kinds[event.kind || "trace"] = true;
  }

  function attachDuration(turn, event) {
    const node = event.node || "runtime";
    const run = latestRun(turn, node) || createImplicitRun(turn, node);
    run.durationMs = event.metadata?.duration_ms ?? null;
    run.closed = true;
  }

  function recordTurnDuration(turn, event) {
    turn.summary.durationMs = event.metadata?.duration_ms ?? null;
  }

  function writableRun(turn, node) {
    const current = latestRun(turn, node);
    return current && !current.closed ? current : createImplicitRun(turn, node);
  }

  function createImplicitRun(turn, node) {
    return createNodeRun(turn, {
      implicit: true,
      label: turn.labelByNode[node] || node,
      node,
    });
  }

  function createNodeRun(turn, config) {
    const layer = layerForNode(config.node);
    const group = ensureLayer(turn, layer);
    const run = {
      closed: false,
      details: [],
      durationMs: null,
      id: `${config.node}-${turn.nextRunId}`,
      implicit: config.implicit,
      kinds: {},
      label: config.label,
      layerKey: layer.key,
      node: config.node,
      seq: nextSeq.value++,
    };
    turn.nextRunId += 1;
    group.nodes.push(run);
    turn.latestByNode[config.node] = run;
    return run;
  }

  function ensureLayer(turn, layer) {
    if (turn.layerMap[layer.key]) return turn.layerMap[layer.key];
    const group = { key: layer.key, labelKey: layer.labelKey, nodes: [], order: layer.order };
    turn.layerMap[layer.key] = group;
    turn.layers.push(group);
    turn.layers.sort((left, right) => left.order - right.order);
    return group;
  }

  function latestRun(turn, node) {
    return turn.latestByNode[node] || null;
  }

  function ensureTurn(payload = {}) {
    if (!currentTurn.value) startTurn(payload);
    return currentTurn.value;
  }

  function replaceCurrentTurn(turn) {
    if (turns.value.length === 0) turns.value.push(turn);
    else turns.value[turns.value.length - 1] = turn;
    currentTurn.value = turn;
  }

  function newTurn(payload = {}) {
    return {
      errors: [],
      id: `turn-${turns.value.length + 1}`,
      labelByNode: {},
      latestByNode: {},
      layerMap: {},
      layers: [],
      locale: payload.locale || null,
      nextRunId: 1,
      snapshotOffset: 0,
      startedAt: new Date().toISOString(),
      summary: { durationMs: null },
      threadId: payload.thread_id || null,
    };
  }

  return { turns, eventCount, observe, clear };
});

function turnDetails(turn) {
  return turn.layers.reduce(
    (total, layer) => total + layer.nodes.reduce((sum, run) => sum + run.details.length, 0),
    0,
  );
}
