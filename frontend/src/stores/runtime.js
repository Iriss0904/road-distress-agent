import { defineStore } from "pinia";
import { computed, ref } from "vue";
import { getJSON } from "../api/client.js";
import { LOCALES, ROLE_OPTIONS, formatMessage } from "../i18n/messages.js";

const LOCALE_KEY = "road_distress_locale";
const USER_KEY = "road_distress_memory_user_id";
export const DEFAULT_USER_ID = "local_user";

export const useRuntimeStore = defineStore("runtime", () => {
  const profile = ref(null);
  const locale = ref(storedValue(LOCALE_KEY, "zh-CN"));
  const userId = ref(storedValue(USER_KEY, DEFAULT_USER_ID));
  const logOpen = ref(false);
  const label = computed(() => {
    if (!profile.value) return t("runtime.connecting");
    const { mode, llm, rag } = profile.value;
    return [mode, llm, rag].filter(Boolean).join(" · ");
  });
  const roles = computed(() =>
    ROLE_OPTIONS.map((role) => ({ ...role, label: t(role.labelKey) })),
  );
  const currentRole = computed(
    () => roles.value.find((role) => role.id === userId.value) || roles.value[0],
  );

  async function load() {
    profile.value = await getJSON("/api/profile");
  }

  function t(key, vars) {
    return formatMessage(locale.value, key, vars);
  }

  function setLocale(next) {
    if (!LOCALES.includes(next)) throw new Error(`unsupported locale: ${next}`);
    locale.value = next;
    persist(LOCALE_KEY, next);
    if (typeof document !== "undefined") {
      document.documentElement.lang = next;
      document.title = t("app.title");
    }
  }

  function toggleLocale() {
    setLocale(locale.value === "zh-CN" ? "en-US" : "zh-CN");
  }

  function setUserId(next) {
    if (!ROLE_OPTIONS.some((role) => role.id === next)) throw new Error(`unknown user: ${next}`);
    userId.value = next;
    persist(USER_KEY, next);
  }

  function toggleLog() {
    logOpen.value = !logOpen.value;
  }

  setLocale(locale.value);

  return {
    profile,
    label,
    locale,
    userId,
    logOpen,
    roles,
    currentRole,
    load,
    setLocale,
    setUserId,
    t,
    toggleLocale,
    toggleLog,
  };
});

function storedValue(key, fallback) {
  if (typeof localStorage === "undefined") return fallback;
  return localStorage.getItem(key) || fallback;
}

function persist(key, value) {
  if (typeof localStorage !== "undefined") localStorage.setItem(key, value);
}
