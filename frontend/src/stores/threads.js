import { defineStore } from "pinia";
import { computed, ref } from "vue";
import { deleteThread, getThread, listThreads, renameThread } from "../api/threads.js";

export const STATUS_ORDER = ["delivered", "promoted", "ready", "draft"];

export const useThreadsStore = defineStore("threads", () => {
  const items = ref([]);
  const activeId = ref(null);
  const activeThread = ref(null);
  const loading = ref(false);
  const error = ref("");

  const groups = computed(() =>
    STATUS_ORDER.map((status) => ({
      status,
      items: items.value.filter((thread) => thread.status === status),
    })).filter((group) => group.items.length),
  );

  async function load(userId, q = "") {
    loading.value = true;
    error.value = "";
    try {
      const rows = await listThreads(userId, q);
      if (!Array.isArray(rows)) throw new Error("/api/threads returned a non-array payload");
      items.value = rows;
    } catch (err) {
      error.value = err.message;
      throw err;
    } finally {
      loading.value = false;
    }
  }

  async function activate(id) {
    try {
      activeId.value = id;
      activeThread.value = await getThread(id);
    } catch (err) {
      error.value = err.message;
      throw err;
    }
  }

  function startNew() {
    activeId.value = null;
    activeThread.value = null;
  }

  async function rename(id, title) {
    const updated = await renameThread(id, title);
    const nextTitle = updated?.title || title;
    items.value = items.value.map((thread) =>
      thread.thread_id === id ? { ...thread, title: nextTitle } : thread,
    );
    if (activeThread.value?.thread_id === id) {
      activeThread.value = { ...activeThread.value, title: nextTitle };
    }
  }

  async function remove(id) {
    await deleteThread(id);
    items.value = items.value.filter((thread) => thread.thread_id !== id);
    if (activeId.value === id) startNew();
  }

  function applyTurnSnapshot(payload) {
    const threadId = payload?.thread_id;
    if (!threadId) return;
    activeId.value = threadId;
    activeThread.value = {
      ...(activeThread.value || {}),
      thread_id: threadId,
      snapshot: payload,
    };
  }

  function appendTranscript(entry) {
    const current = activeThread.value || { thread_id: activeId.value, transcript: [] };
    activeThread.value = { ...current, transcript: [...(current.transcript || []), entry] };
  }

  return {
    items,
    activeId,
    activeThread,
    loading,
    error,
    groups,
    load,
    activate,
    startNew,
    rename,
    remove,
    applyTurnSnapshot,
    appendTranscript,
  };
});
