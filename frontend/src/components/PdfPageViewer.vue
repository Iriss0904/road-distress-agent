<script setup>
import { computed, nextTick, onBeforeUnmount, ref, shallowRef, watch } from "vue";
import pdfWorkerUrl from "pdfjs-dist/legacy/build/pdf.worker.min.mjs?url";
import { useRuntimeStore } from "../stores/runtime.js";

const DEFAULT_PAGE = 1;
const DEFAULT_ZOOM = 1;
const MIN_ZOOM = 0.6;
const MAX_ZOOM = 2.4;
const ZOOM_STEP = 0.2;
const MIN_FIT_SCALE = 0.35;
const MAX_RENDER_SCALE = 4;
const CANVAS_GUTTER = 10;
const DECIMAL_RADIX = 10;
const PERCENT_FACTOR = 100;
const SCALE_PRECISION = 2;
const DEVICE_PIXEL_RATIO_FALLBACK = 1;
const CANCELLED_RENDER = "RenderingCancelledException";

let pdfjsPromise = null;

const props = defineProps({
  src: { type: String, default: "" },
  page: { type: [Number, String], default: DEFAULT_PAGE },
  title: { type: String, default: "" },
});

const runtime = useRuntimeStore();
const shell = ref(null);
const canvas = ref(null);
const pdfDoc = shallowRef(null);
const currentPage = ref(DEFAULT_PAGE);
const pageCount = ref(0);
const zoom = ref(DEFAULT_ZOOM);
const loading = ref(false);
const rendering = ref(false);
const error = ref("");
const loadTask = shallowRef(null);
const renderTask = shallowRef(null);
const renderToken = ref(0);

const pageLabel = computed(() =>
  runtime.t("panel.pdfPage", { page: currentPage.value, total: pageCount.value || "-" }),
);
const zoomLabel = computed(() => `${Math.round(zoom.value * PERCENT_FACTOR)}%`);
const canGoBack = computed(() => currentPage.value > DEFAULT_PAGE);
const canGoNext = computed(() => currentPage.value < pageCount.value);
const canZoomOut = computed(() => zoom.value > MIN_ZOOM);
const canZoomIn = computed(() => zoom.value < MAX_ZOOM);
const busy = computed(() => loading.value || rendering.value);

watch(() => props.src, () => void loadDocument(), { immediate: true });
watch(() => props.page, syncPageFromProp);
watch([currentPage, zoom], () => {
  if (!pdfDoc.value || loading.value) return;
  void renderPage(nextToken());
});

onBeforeUnmount(() => {
  renderToken.value += 1;
  void resetDocument();
});

async function loadDocument() {
  const token = nextToken();
  try {
    await resetDocument();
    error.value = "";
    if (token !== renderToken.value || !props.src) return;
    loading.value = true;
    const { getDocument } = await loadPdfjs();
    if (token !== renderToken.value) return;
    loadTask.value = getDocument({ url: props.src });
    const document = await loadTask.value.promise;
    if (token !== renderToken.value) return;
    pdfDoc.value = document;
    pageCount.value = document.numPages;
    currentPage.value = normalizePage(props.page, document.numPages);
    loading.value = false;
    await renderPage(token);
  } catch (cause) {
    if (token !== renderToken.value) return;
    loading.value = false;
    error.value = errorText(cause);
  }
}

async function renderPage(token) {
  if (!pdfDoc.value) return;
  rendering.value = true;
  error.value = "";
  try {
    await cancelRender();
    await nextTick();
    if (token !== renderToken.value) return;
    if (!canvas.value) return;
    const page = await pdfDoc.value.getPage(currentPage.value);
    if (token !== renderToken.value) return;
    await paintPage(page);
  } catch (cause) {
    if (isCancelled(cause)) return;
    error.value = errorText(cause);
  } finally {
    if (token === renderToken.value) rendering.value = false;
  }
}

async function paintPage(page) {
  const context = canvas.value.getContext("2d");
  if (!context) throw new Error("Canvas 2D context is unavailable");
  const viewport = page.getViewport({ scale: renderScale(page) });
  const pixelRatio = window.devicePixelRatio || DEVICE_PIXEL_RATIO_FALLBACK;
  canvas.value.width = Math.floor(viewport.width * pixelRatio);
  canvas.value.height = Math.floor(viewport.height * pixelRatio);
  canvas.value.style.width = `${Math.floor(viewport.width)}px`;
  canvas.value.style.height = `${Math.floor(viewport.height)}px`;
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  const task = page.render({ canvasContext: context, viewport });
  renderTask.value = task;
  try {
    await task.promise;
  } finally {
    if (renderTask.value === task) renderTask.value = null;
  }
}

function renderScale(page) {
  const viewport = page.getViewport({ scale: DEFAULT_ZOOM });
  const width = shell.value?.clientWidth || viewport.width;
  const fitScale = Math.max(MIN_FIT_SCALE, (width - CANVAS_GUTTER) / viewport.width);
  return Math.min(MAX_RENDER_SCALE, fitScale * zoom.value);
}

async function resetDocument() {
  await cancelRender();
  if (loadTask.value) loadTask.value.destroy();
  if (pdfDoc.value) pdfDoc.value.destroy();
  loadTask.value = null;
  pdfDoc.value = null;
  pageCount.value = 0;
  loading.value = false;
  rendering.value = false;
}

async function cancelRender() {
  const task = renderTask.value;
  if (!task) return;
  renderTask.value = null;
  task.cancel();
  try {
    await task.promise;
  } catch (cause) {
    if (!isCancelled(cause)) throw cause;
  }
}

function syncPageFromProp() {
  currentPage.value = normalizePage(props.page, pageCount.value);
}

function shiftPage(delta) {
  currentPage.value = normalizePage(currentPage.value + delta, pageCount.value);
}

function shiftZoom(delta) {
  zoom.value = clampZoom(zoom.value + delta);
}

function normalizePage(value, total) {
  const parsed = Number.parseInt(value, DECIMAL_RADIX);
  const page = Number.isFinite(parsed) ? parsed : DEFAULT_PAGE;
  const upper = total || page;
  return Math.min(Math.max(page, DEFAULT_PAGE), upper);
}

function clampZoom(value) {
  const clamped = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, value));
  return Number(clamped.toFixed(SCALE_PRECISION));
}

function nextToken() {
  renderToken.value += 1;
  return renderToken.value;
}

function errorText(cause) { return cause instanceof Error ? cause.message : String(cause); }
function isCancelled(cause) { return cause?.name === CANCELLED_RENDER; }

async function loadPdfjs() {
  if (!pdfjsPromise) {
    pdfjsPromise = import("pdfjs-dist/legacy/build/pdf.min.mjs").then((module) => {
      module.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;
      return module;
    });
  }
  return pdfjsPromise;
}
</script>

<template>
  <div class="pdf-viewer">
    <div class="pdf-toolbar">
      <button type="button" :disabled="!canGoBack || busy" :aria-label="runtime.t('panel.pdfPrevious')" @click="shiftPage(-1)">
        &lt;
      </button>
      <span>{{ pageLabel }}</span>
      <button type="button" :disabled="!canGoNext || busy" :aria-label="runtime.t('panel.pdfNext')" @click="shiftPage(1)">
        &gt;
      </button>
      <i />
      <button type="button" :disabled="!canZoomOut || busy" :aria-label="runtime.t('panel.pdfZoomOut')" @click="shiftZoom(-ZOOM_STEP)">
        -
      </button>
      <span>{{ zoomLabel }}</span>
      <button type="button" :disabled="!canZoomIn || busy" :aria-label="runtime.t('panel.pdfZoomIn')" @click="shiftZoom(ZOOM_STEP)">
        +
      </button>
    </div>

    <div ref="shell" class="pdf-canvas-shell" :aria-busy="busy">
      <p v-if="loading" class="pdf-status">{{ runtime.t("panel.pdfLoading") }}</p>
      <canvas v-show="!loading && !error" ref="canvas" :aria-label="title" data-testid="pdf-canvas" />
      <p v-if="error" class="pdf-error">{{ runtime.t("panel.pdfRenderError", { message: error }) }}</p>
    </div>
  </div>
</template>

<style scoped>
.pdf-viewer {
  display: grid;
  min-height: 0;
  border: 1px solid var(--rd-border);
  border-radius: var(--rd-radius-control);
  background: var(--rd-bg);
}

.pdf-toolbar {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px;
  border-bottom: 1px solid var(--rd-border);
  background: var(--rd-surface);
  color: var(--rd-text-dim);
  font-size: 12px;
}

.pdf-toolbar button {
  width: 28px;
  height: 28px;
  border: 1px solid var(--rd-border);
  border-radius: var(--rd-radius-control);
  background: var(--rd-surface-soft);
  color: var(--rd-text-strong);
  cursor: pointer;
}

.pdf-toolbar button:disabled {
  cursor: not-allowed;
  opacity: 0.45;
}

.pdf-toolbar button:focus-visible {
  border-color: var(--rd-accent);
  outline: none;
}

.pdf-toolbar i {
  flex: 1;
}

.pdf-canvas-shell {
  position: relative;
  display: grid;
  min-height: 520px;
  max-height: calc(100vh - 230px);
  overflow: auto;
  place-items: start center;
  padding: 12px 0;
}

canvas {
  display: block;
  background: white;
  box-shadow: 0 8px 28px rgba(31, 38, 70, 0.12);
}

.pdf-status,
.pdf-error {
  margin: 18px;
  color: var(--rd-text-dim);
  font-size: 13px;
}

.pdf-error {
  color: #a33a3a;
}
</style>
