const OTHER_LAYER = Object.freeze({
  key: "other",
  labelKey: "traceLayerOther",
  nodes: Object.freeze([]),
  order: 99,
});

const TRACE_LAYER_DEFINITIONS = Object.freeze([
  layer({
    key: "input",
    order: 1,
    labelKey: "traceLayerInput",
    nodes: ["user_input", "parallel_context_loader"],
  }),
  layer({
    key: "routing",
    order: 2,
    labelKey: "traceLayerRouting",
    nodes: ["top_router", "intent_router", "handle_off_intent", "off_topic_refuser"],
  }),
  layer({ key: "vision", order: 3, labelKey: "traceLayerVision", nodes: ["vision_subgraph"] }),
  layer({
    key: "reconcile",
    order: 4,
    labelKey: "traceLayerReconcile",
    nodes: ["diagnosis_reconcile"],
  }),
  layer({
    key: "disease",
    order: 5,
    labelKey: "traceLayerDisease",
    nodes: [
      "disease_selection_handler",
      "disease_query_rewriter",
      "disease_retriever",
      "disease_continue_discriminator",
      "disease_discriminator",
      "disease_result_router",
    ],
  }),
  layer({
    key: "method",
    order: 6,
    labelKey: "traceLayerMethod",
    nodes: [
      "method_selection_handler",
      "method_query_rewriter",
      "method_retriever",
      "method_discriminator",
      "method_result_router",
      "detail_retriever_v2",
    ],
  }),
  layer({
    key: "kb",
    order: 7,
    labelKey: "traceLayerKb",
    nodes: [
      "kb_query_planner",
      "kb_query_rewriter",
      "kb_retriever",
      "kb_direct_meta_answer",
      "kb_answer_composer",
      "kb_hop_retriever",
      "kb_planned_answer_composer",
      "kb_clarification_composer",
    ],
  }),
  layer({
    key: "weather_safety",
    order: 8,
    labelKey: "traceLayerWeatherSafety",
    nodes: [
      "weather_location_handler",
      "safety_norm_rewriter",
      "safety_norm_retriever",
      "address_weather_loader",
      "construction_arrangement_advisor",
      "weather_advisor",
      "construction_tip_offer",
    ],
  }),
  layer({ key: "synthesis", order: 9, labelKey: "traceLayerSynthesis", nodes: ["answer_composer"] }),
  layer({
    key: "critic_memory",
    order: 10,
    labelKey: "traceLayerCriticMemory",
    nodes: ["safety_critic", "memory_writer"],
  }),
]);

const NODE_TO_LAYER = Object.freeze(buildNodeMap(TRACE_LAYER_DEFINITIONS));

export function layerForNode(nodeName) {
  return NODE_TO_LAYER[nodeName] || OTHER_LAYER;
}

function layer(config) {
  return Object.freeze({ ...config, nodes: Object.freeze(config.nodes) });
}

function buildNodeMap(layers) {
  return Object.fromEntries(layers.flatMap((entry) => entry.nodes.map((node) => [node, entry])));
}
