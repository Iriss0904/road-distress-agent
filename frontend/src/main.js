import { createApp } from "vue";
import { createPinia } from "pinia";
import "./styles/tokens.css";
import App from "./App.vue";
import { router } from "./router/index.js";

createApp(App).use(createPinia()).use(router).mount("#app");
