<script setup>
import NavRail from "./components/NavRail.vue";
import RuntimeLogDrawer from "./components/RuntimeLogDrawer.vue";
import TopBar from "./components/TopBar.vue";
</script>

<template>
  <div class="shell" data-testid="app-root">
    <NavRail />
    <div class="main">
      <TopBar />
      <main class="canvas">
        <RouterView />
      </main>
    </div>
    <RuntimeLogDrawer />
  </div>
</template>

<style scoped>
.shell {
  display: flex;
  min-height: 100vh;
}

.main {
  display: flex;
  flex: 1;
  min-width: 0;
  flex-direction: column;
}

.canvas {
  flex: 1;
  overflow: auto;
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.72), rgba(251, 251, 253, 0)),
    var(--rd-bg);
}
</style>
