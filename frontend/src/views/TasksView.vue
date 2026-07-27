<script setup>
import { onMounted, watch } from "vue";
import LedgerTable from "../components/LedgerTable.vue";
import TaskList from "../components/TaskList.vue";
import { useRuntimeStore } from "../stores/runtime.js";
import { useTasksStore } from "../stores/tasks.js";

const runtime = useRuntimeStore();
const store = useTasksStore();

onMounted(load);
watch(() => runtime.userId, load);

function load() {
  store.loadProjects(runtime.userId);
}
</script>

<template>
  <section class="tasks-view">
    <TaskList />
    <LedgerTable />
  </section>
</template>

<style scoped>
.tasks-view {
  display: flex;
  min-height: 100%;
}
</style>
