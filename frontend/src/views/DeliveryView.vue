<script setup>
import { onMounted, watch } from "vue";
import DeliveryDetail from "../components/DeliveryDetail.vue";
import DeliveryList from "../components/DeliveryList.vue";
import { useRuntimeStore } from "../stores/runtime.js";
import { useDeliveriesStore } from "../stores/deliveries.js";

const runtime = useRuntimeStore();
const store = useDeliveriesStore();

onMounted(load);
watch(() => runtime.userId, load);

async function load() {
  try {
    await store.loadList(runtime.userId);
    if (store.items[0]) await store.open(store.items[0].delivery_id);
    else store.active = null;
  } catch (err) {
    store.error = err.message;
  }
}
</script>

<template>
  <section class="delivery-view">
    <DeliveryList />
    <DeliveryDetail />
  </section>
</template>

<style scoped>
.delivery-view {
  display: flex;
  min-height: 100%;
}
</style>
