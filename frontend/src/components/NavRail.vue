<script setup>
import { RouterLink } from "vue-router";
import { ROUTES } from "../router/index.js";
import { useRuntimeStore } from "../stores/runtime.js";

const runtime = useRuntimeStore();
</script>

<template>
  <nav class="rail" :aria-label="runtime.t('nav.diagnose')">
    <div class="logo" aria-label="Road Distress">RD</div>
    <div class="items">
      <RouterLink
        v-for="route in ROUTES"
        :key="route.name"
        :to="{ name: route.name }"
        class="item"
        active-class="active"
        data-testid="nav-item"
      >
        <span class="ic" aria-hidden="true">{{ route.meta.icon }}</span>
        <span class="tx">{{ runtime.t(route.meta.labelKey) }}</span>
      </RouterLink>
    </div>
    <div class="foot">
      <button type="button" class="item" data-testid="nav-onboarding">
        <span class="ic" aria-hidden="true">🎓</span>
        <span class="tx">{{ runtime.t("nav.onboarding") }}</span>
      </button>
      <button type="button" class="item" data-testid="nav-settings">
        <span class="ic" aria-hidden="true">⚙️</span>
        <span class="tx">{{ runtime.t("nav.settings") }}</span>
      </button>
    </div>
  </nav>
</template>

<style scoped>
.rail {
  display: flex;
  width: 76px;
  height: 100vh;
  flex: none;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  padding: 14px 0;
  border-right: 1px solid var(--rd-border);
  background: var(--rd-surface);
}

.logo {
  display: flex;
  width: 38px;
  height: 38px;
  align-items: center;
  justify-content: center;
  margin-bottom: 14px;
  border-radius: 10px;
  background: linear-gradient(135deg, var(--rd-accent), var(--rd-accent-2));
  color: #fff;
  font-weight: 700;
}

.items,
.foot {
  display: flex;
  width: 100%;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}

.items {
  flex: 1;
}

.foot {
  gap: 6px;
}

.item {
  display: flex;
  width: 60px;
  min-height: 58px;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 3px;
  border: 0;
  border-radius: 10px;
  background: none;
  color: var(--rd-text-dim);
  cursor: pointer;
  font: inherit;
  font-size: 11px;
  line-height: 1.25;
  text-align: center;
  text-decoration: none;
}

.item:hover,
.item:focus-visible {
  background: var(--rd-accent-soft);
  color: var(--rd-accent);
  outline: none;
}

.item.active {
  background: var(--rd-accent-soft);
  color: var(--rd-accent);
  font-weight: 650;
}

.ic {
  font-size: 18px;
  line-height: 1;
}
</style>
