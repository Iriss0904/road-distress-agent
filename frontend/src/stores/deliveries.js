import { defineStore } from "pinia";
import { computed, ref } from "vue";
import {
  getDelivery,
  listDeliveries,
  regenerate as regenerateDelivery,
} from "../api/deliveries.js";

export const useDeliveriesStore = defineStore("deliveries", () => {
  const items = ref([]);
  const active = ref(null);
  const loading = ref(false);
  const regenerating = ref(false);
  const error = ref("");
  const currentUserId = ref("");
  const activeId = computed(() => active.value?.delivery?.delivery_id || "");

  async function loadList(userId) {
    loading.value = true;
    error.value = "";
    currentUserId.value = userId;
    try {
      const rows = await listDeliveries(userId);
      items.value = asArray(rows, "/api/deliveries");
    } catch (err) {
      error.value = err.message;
      throw err;
    } finally {
      loading.value = false;
    }
  }

  async function open(id) {
    loading.value = true;
    error.value = "";
    try {
      active.value = normalizeDetail(await getDelivery(id));
    } catch (err) {
      error.value = err.message;
      throw err;
    } finally {
      loading.value = false;
    }
  }

  async function regenerateActive() {
    const id = activeId.value;
    if (!id) {
      error.value = "请选择一个交付归档。";
      throw new Error(error.value);
    }
    regenerating.value = true;
    error.value = "";
    try {
      await regenerateDelivery(id);
      if (currentUserId.value) await loadList(currentUserId.value);
      await open(id);
    } catch (err) {
      error.value = err.message;
      throw err;
    } finally {
      regenerating.value = false;
    }
  }

  return {
    items,
    active,
    loading,
    regenerating,
    error,
    activeId,
    loadList,
    open,
    regenerateActive,
  };
});

function normalizeDetail(payload) {
  if (!payload || typeof payload !== "object") {
    throw new Error("/api/deliveries/{id} returned an invalid payload");
  }
  if (!payload.delivery) throw new Error("/api/deliveries/{id} returned no delivery");
  return {
    delivery: payload.delivery,
    versions: asArray(payload.versions, "/api/deliveries/{id}.versions"),
    provenance: asArray(payload.provenance, "/api/deliveries/{id}.provenance"),
  };
}

function asArray(value, endpoint) {
  if (Array.isArray(value)) return value;
  throw new Error(`${endpoint} returned a non-array payload`);
}
