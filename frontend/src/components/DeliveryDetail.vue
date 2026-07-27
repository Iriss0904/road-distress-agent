<script setup>
import { computed, ref } from "vue";
import { useRouter } from "vue-router";
import { downloadUrl } from "../api/deliveries.js";
import { useDeliveriesStore } from "../stores/deliveries.js";
import { useRuntimeStore } from "../stores/runtime.js";
import { useThreadsStore } from "../stores/threads.js";
import "./delivery-detail.css";

const store = useDeliveriesStore();
const runtime = useRuntimeStore();
const threads = useThreadsStore();
const router = useRouter();
const actionError = ref("");

const active = computed(() => store.active);
const delivery = computed(() => active.value?.delivery);
const versions = computed(() => active.value?.versions || []);
const provenance = computed(() => active.value?.provenance || []);
const latestVersion = computed(() => versions.value[versions.value.length - 1] || null);
const pdfVersion = computed(() => [...versions.value].reverse().find((v) => v.file_format === "pdf"));
const latestUrl = computed(() => downloadFor(latestVersion.value));
const pdfUrl = computed(() => downloadFor(pdfVersion.value));

async function regenerate() {
  actionError.value = "";
  try {
    await store.regenerateActive();
  } catch (err) {
    actionError.value = err.message;
  }
}

async function openThread(threadId) {
  if (!threadId) return;
  actionError.value = "";
  try {
    await threads.activate(threadId);
    await router.push({ name: "diagnose" });
  } catch (err) {
    actionError.value = err.message;
  }
}

function downloadFor(version) {
  if (!delivery.value || !version) return "";
  return downloadUrl(delivery.value.delivery_id, version.version_no);
}

function statusLabel(status) {
  return { draft: runtime.t("delivery.draft"), final: runtime.t("delivery.final") }[status]
    || status || runtime.t("common.none");
}

function generatedLabel(value) {
  return { auto: runtime.t("delivery.auto"), human: runtime.t("delivery.human") }[value]
    || value || runtime.t("common.none");
}

function shortDate(value) {
  if (!value) return "";
  return value.slice(0, 19).replace("T", " ");
}

function checksum(value) {
  return value ? value.slice(0, 12) : "";
}
</script>

<template>
  <section class="delivery-detail">
    <div v-if="!delivery" class="empty-state">
      <h2>{{ runtime.t("delivery.preview") }}</h2>
      <p>{{ runtime.t("delivery.empty") }}</p>
    </div>
    <template v-else>
      <header class="detail-head">
        <div>
          <p>{{ delivery.project_id }}</p>
          <h2>{{ delivery.title }}</h2>
        </div>
        <span :class="['status', delivery.status]">{{ statusLabel(delivery.status) }}</span>
      </header>
      <p v-if="store.error || actionError" class="error">{{ actionError || store.error }}</p>

      <section class="preview">
        <div class="preview-main">
          <span>{{ runtime.t("delivery.latest") }}</span>
          <strong v-if="latestVersion">v{{ latestVersion.version_no }}</strong>
          <strong v-else>{{ runtime.t("delivery.notGenerated") }}</strong>
        </div>
        <div class="preview-meta">
          <span>{{ latestVersion?.file_format?.toUpperCase() || "—" }}</span>
          <span>{{ shortDate(latestVersion?.created_at) }}</span>
          <span>{{ checksum(latestVersion?.checksum) }}</span>
        </div>
        <ul class="included">
          <li v-for="row in provenance" :key="row.record_id">
            {{ row.defect_category || row.record_id }}
          </li>
          <li v-if="!provenance.length">{{ runtime.t("delivery.noRecords") }}</li>
        </ul>
      </section>

      <div class="actions">
        <a v-if="latestUrl" class="button" :href="latestUrl">{{ runtime.t("delivery.download") }}</a>
        <button v-else type="button" disabled>{{ runtime.t("delivery.download") }}</button>
        <a v-if="pdfUrl" class="button" :href="pdfUrl">{{ runtime.t("delivery.exportPdf") }}</a>
        <button v-else type="button" disabled>{{ runtime.t("delivery.exportPdf") }}</button>
        <button type="button" :disabled="store.regenerating" @click="regenerate">
          {{ store.regenerating ? runtime.t("delivery.regenerating") : runtime.t("delivery.regenerate") }}
        </button>
      </div>

      <section class="block">
        <header>
          <h3>{{ runtime.t("delivery.provenance") }}</h3>
          <span>{{ runtime.t("delivery.records", { count: provenance.length }) }}</span>
        </header>
        <div v-if="!provenance.length" class="empty-line">
          {{ runtime.t("delivery.noProvenance") }}
        </div>
        <button
          v-for="row in provenance"
          :key="row.record_id"
          type="button"
          class="provenance-row"
          @click="openThread(row.source_thread_id)"
        >
          <span>{{ row.defect_category || row.record_id }}</span>
          <span>{{ row.source_thread_id }}</span>
        </button>
      </section>

      <section class="block">
        <header>
          <h3>{{ runtime.t("delivery.versionHistory") }}</h3>
          <span>{{ runtime.t("delivery.versionCount", { count: versions.length }) }}</span>
        </header>
        <ol class="versions">
          <li v-for="version in versions" :key="version.version_id">
            <span>v{{ version.version_no }}</span>
            <span>{{ generatedLabel(version.generated_by) }}</span>
            <span>{{ version.file_format.toUpperCase() }}</span>
            <span>{{ shortDate(version.created_at) }}</span>
          </li>
        </ol>
      </section>
    </template>
  </section>
</template>
