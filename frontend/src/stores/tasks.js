import { defineStore } from "pinia";
import { computed, ref } from "vue";
import { getLedger, listProjects, listUnfiled, promote } from "../api/tasks.js";

export const UNFILED_PROJECT_ID = "unfiled";

export const useTasksStore = defineStore("tasks", () => {
  const projects = ref([]);
  const unfiled = ref([]);
  const activeLedger = ref(emptyLedger());
  const selected = ref(new Set());
  const loading = ref(false);
  const error = ref("");
  const currentUserId = ref("");
  const activeProjectId = computed(() => activeLedger.value.project?.project_id || "");

  async function loadProjects(userId) {
    loading.value = true;
    error.value = "";
    const userChanged = currentUserId.value && currentUserId.value !== userId;
    currentUserId.value = userId;
    try {
      const [projectRows, unfiledRows] = await Promise.all([
        listProjects(userId),
        listUnfiled(userId),
      ]);
      projects.value = asArray(projectRows, "/api/projects").map(withEmptyRows);
      unfiled.value = asArray(unfiledRows, "/api/ledger/unfiled");
      if (userChanged) resetActive();
    } catch (err) {
      error.value = err.message;
      throw err;
    } finally {
      loading.value = false;
    }
  }

  async function openLedger(id) {
    error.value = "";
    try {
      const ledger = await getLedger(id);
      activeLedger.value = normalizeLedger(ledger);
      selected.value = new Set();
      syncProjectRows(activeLedger.value);
    } catch (err) {
      error.value = err.message;
      throw err;
    }
  }

  function openUnfiled() {
    activeLedger.value = {
      project: { project_id: UNFILED_PROJECT_ID, name: "未归类（散诊断）", synthetic: true },
      rows: unfiled.value.map(unfiledRow),
    };
    selected.value = new Set();
  }

  function toggleSelect(id) {
    const next = new Set(selected.value);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    selected.value = next;
  }

  async function promoteThread(projectId, threadId) {
    if (!projectId || projectId === UNFILED_PROJECT_ID) throw new Error("请选择一个真实巡查任务。");
    if (!threadId) throw new Error("请选择一个可纳入的诊断会话。");
    const record = await promote(projectId, { thread_id: threadId });
    await refreshAfterPromote(projectId);
    return record;
  }

  async function refreshAfterPromote(projectId) {
    await openLedger(projectId);
    if (!currentUserId.value) return;
    unfiled.value = asArray(await listUnfiled(currentUserId.value), "/api/ledger/unfiled");
  }

  function syncProjectRows(ledger) {
    const projectId = ledger.project?.project_id;
    projects.value = projects.value.map((project) =>
      project.project_id === projectId ? { ...project, rows: ledger.rows } : project,
    );
  }

  function resetActive() {
    activeLedger.value = emptyLedger();
    selected.value = new Set();
  }

  return {
    projects,
    unfiled,
    activeLedger,
    selected,
    loading,
    error,
    activeProjectId,
    loadProjects,
    openLedger,
    openUnfiled,
    toggleSelect,
    promoteThread,
  };
});

function emptyLedger() {
  return { project: null, rows: [] };
}

function normalizeLedger(ledger) {
  if (!ledger || typeof ledger !== "object") throw new Error("/api/ledger returned an invalid payload");
  if (!ledger.project) throw new Error("/api/ledger returned a payload without project");
  return {
    project: ledger.project,
    rows: asArray(ledger.rows, "/api/ledger/{project_id}"),
  };
}

function asArray(value, endpoint) {
  if (Array.isArray(value)) return value;
  throw new Error(`${endpoint} returned a non-array payload`);
}

function withEmptyRows(project) {
  return { ...project, rows: project.rows || [] };
}

function unfiledRow(thread) {
  return {
    record_id: "",
    defect_category: thread.title || thread.thread_id,
    chosen_method: "待纳入任务",
    location: "",
    evidence: "missing",
    review: "未纳入",
    source_thread_id: thread.thread_id,
    synthetic: true,
  };
}
