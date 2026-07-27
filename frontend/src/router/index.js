import { createRouter, createWebHistory } from "vue-router";
import DeliveryView from "../views/DeliveryView.vue";
import DiagnoseView from "../views/DiagnoseView.vue";
import KnowledgeView from "../views/KnowledgeView.vue";
import TasksView from "../views/TasksView.vue";

export const ROUTES = [
  {
    name: "diagnose",
    path: "/",
    component: DiagnoseView,
    meta: { icon: "💬", labelKey: "nav.diagnose" },
  },
  {
    name: "tasks",
    path: "/tasks",
    component: TasksView,
    meta: { icon: "📋", labelKey: "nav.tasks" },
  },
  {
    name: "delivery",
    path: "/delivery",
    component: DeliveryView,
    meta: { icon: "📦", labelKey: "nav.delivery" },
  },
  {
    name: "knowledge",
    path: "/knowledge",
    component: KnowledgeView,
    meta: { icon: "📚", labelKey: "nav.knowledge" },
  },
];

export const router = createRouter({
  history: createWebHistory(),
  routes: ROUTES,
});
