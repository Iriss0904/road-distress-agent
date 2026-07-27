import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

const API_TARGET = process.env.VITE_API_TARGET || "http://127.0.0.1:8010";

export default defineConfig({
  plugins: [vue()],
  build: { outDir: "dist", emptyOutDir: true },
  server: {
    proxy: {
      "/api": { target: API_TARGET, changeOrigin: true },
    },
  },
  test: { css: true, environment: "jsdom", globals: false },
});
