import { app } from "../../scripts/app.js";

const NODE_TYPE = "KiriImageLoader";
const metadataByNode = new WeakMap();
const livePreviewByNode = new WeakMap();
const executionMetadataByNode = new WeakMap();
const livePreviewSerialByNode = new WeakMap();
const persistedPreviewRestorePending = new WeakSet();
const requestSerialByNode = new WeakMap();
const attachedNodes = new WeakSet();
const patchedCanvases = new WeakSet();
const patchedDrawCanvases = new WeakSet();
const patchedPreviewWidgets = new WeakSet();
const previewSourceInstalledNodes = new WeakSet();
const nativePreviewFallbackReaders = new WeakMap();
const upstreamImageUrlCache = new Map();
const alphaMaskPreviewCache = new WeakMap();
const comparePointerCanvases = new WeakSet();
const compareDragByCanvas = new WeakMap();
const compareWidgetHandlers = new WeakMap();
const previewSurfaceHandlers = new WeakMap();
let activeFallbackMenuCleanup = null;
const previewIntegrationErrorsLogged = new WeakSet();
const MAX_IMAGE_TRACE_NODES = 256;
const MAX_COMPARISON_PREVIEW_PIXELS = 2_000_000;
const MAX_COMPARISON_PREVIEW_EDGE = 2048;
const PERSISTED_LIVE_PREVIEW_PROPERTY = "kiri_live_preview";
const MAX_PERSISTED_LIVE_PREVIEW_CHARS = 8 * 1024 * 1024;

function isKiriNode(node) {
  return node?.comfyClass === NODE_TYPE || node?.constructor?.comfyClass === NODE_TYPE || node?.type === NODE_TYPE;
}

function getImageWidget(node) {
  return node?.widgets?.find((widget) => widget?.name === "image");
}

function currentFilename(node) {
  const widget = getImageWidget(node);
  return typeof widget?.value === "string" ? widget.value : "";
}

function getMetadataState(node, sourceKey = "loaded") {
  return metadataByNode.get(node)?.[sourceKey] || null;
}

function setMetadataState(node, sourceKey, state) {
  const states = metadataByNode.get(node) || {};
  states[sourceKey] = state;
  metadataByNode.set(node, states);
}

function descriptorSignature(descriptor) {
  if (descriptor?.runtimeMetadata) {
    return `runtime\n${descriptor.sourceKey || "source"}\n${descriptor.executionSerial || 0}`;
  }
  if (!descriptor?.filename) return descriptor?.status || "unavailable";
  return [descriptor.type || "input", descriptor.subfolder || "", descriptor.filename, descriptor.pathToken || ""].join("\n");
}

function parseRuntimeMetadata(value) {
  if (value && typeof value === "object") return value;
  if (typeof value !== "string" || !value.trim()) return null;
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch {
    return null;
  }
}

function runtimeMetadataCopyItems(record) {
  const items = [];
  const add = (category, label, value) => {
    if (value === undefined || value === null || value === "") return;
    const normalized = typeof value === "string" ? value : JSON.stringify(value, null, 2);
    if (!normalized) return;
    items.push({ category, label, value: normalized });
  };
  const graph = record?.graph || {};
  const prompts = graph.prompts && typeof graph.prompts === "object" ? graph.prompts : {};
  const hasPositivePrompt = Object.prototype.hasOwnProperty.call(prompts, "positive");
  const hasNegativePrompt = Object.prototype.hasOwnProperty.call(prompts, "negative");
  if (hasPositivePrompt) add("Prompts", "Positive Prompt", prompts.positive);
  if (hasNegativePrompt) add("Prompts", "Negative Prompt", prompts.negative);
  for (const item of graph.text_encoders_and_prompts || []) {
    const branch = item?.branch === "positive" ? "Positive Prompt" : item?.branch === "negative" ? "Negative Prompt" : null;
    if (branch === "Positive Prompt" && hasPositivePrompt) continue;
    if (branch === "Negative Prompt" && hasNegativePrompt) continue;
    if (branch) {
      const value = Object.values(item.values || {}).filter(Boolean).join("\n");
      add("Prompts", branch, value);
    } else {
      for (const [key, value] of Object.entries(item?.values || {})) add("Other Prompt Inputs", `${item.class_type || "Text Encoder"} · ${key}`, value);
    }
  }
  (graph.samplers || []).forEach((sampler, index) => {
    for (const [key, value] of Object.entries(sampler?.settings || {})) {
      add("Generation", `Sampler ${index + 1} · ${key}`, value);
    }
  });
  for (const resource of graph.model_resources || []) {
    const category = resource?.type === "LoRA" ? "LoRAs" : "Models";
    const name = resource?.name || resource?.type || "Resource";
    for (const [key, value] of Object.entries(resource || {})) {
      if (["node_id", "class_type", "input", "type", "name"].includes(key)) continue;
      add(category, `${name} · ${key}`, value);
    }
    add(category, `${resource?.type || "Resource"} · Name`, name);
  }
  for (const detail of graph.kiri_loader_details || []) {
    for (const [key, value] of Object.entries(detail || {})) {
      if (key === "node_id") continue;
      add("Kiri Loader Details", key.replaceAll("_", " "), value);
    }
  }
  for (const item of graph.active_upstream_nodes || []) {
    const className = item?.title || item?.class_type || "Upstream Node";
    for (const [key, value] of Object.entries(item?.inputs || {})) {
      add("ComfyUI Nodes", `${className} · ${key}`, value);
    }
  }
  if (record?.custom_metadata?.value !== undefined) add("Custom Metadata", "Custom Metadata", record.custom_metadata.value);
  if (record?.parameters) add("Generation", "A1111 Parameters", record.parameters);
  return items;
}

function connectedSourceIdentity(node, sourceKey) {
  const inputName = sourceKey === "image_a" ? "image_in" : "mask_in";
  const source = getConnectedSource(node, inputName);
  return source ? `${String(source.node?.id ?? "unknown")}:${Number(source.slot || 0)}` : null;
}

function effectiveImageDescriptor(node, sourceKey) {
  const traced = connectedImageDescriptor(node, sourceKey);
  if (!traced) return null;
  if (traced.filename) return traced;
  const executed = executionMetadataByNode.get(node);
  const identityMatches = !!executed && executed.sourceIdentities?.[sourceKey] === connectedSourceIdentity(node, sourceKey);
  if (!identityMatches) return traced;
  if (sourceKey === "image_a" && executed?.savedOutputA) {
    return { ...executed.savedOutputA, originNode: node, originKind: "kiri-generated-output" };
  }
  if (sourceKey === "image_b" && executed?.savedOutputB) {
    return { ...executed.savedOutputB, originNode: node, originKind: "kiri-generated-output" };
  }
  const runtimeMetadata = sourceKey === "image_a" ? executed?.metadataA : executed?.metadataB;
  if (runtimeMetadata) {
    return {
      status: "runtime",
      runtimeMetadata: true,
      runtimeRecord: runtimeMetadata,
      runtimeWorkflow: runtimeMetadata.workflow || null,
      runtimeKey: sourceKey,
      sourceKey,
      executionSerial: executed.serial,
      message: "Generation metadata was collected from this connected Kiri input. The source has no independently saved file.",
    };
  }
  return traced;
}

async function readMetadataForSource(node, sourceKey, descriptor) {
  const serials = requestSerialByNode.get(node) || new Map();
  const serial = (serials.get(sourceKey) || 0) + 1;
  serials.set(sourceKey, serial);
  requestSerialByNode.set(node, serials);

  if (descriptor?.runtimeMetadata && descriptor.runtimeRecord) {
    setMetadataState(node, sourceKey, {
      ...(descriptor || {}),
      ...descriptor.runtimeRecord,
      loading: false,
      unavailable: false,
      runtime: true,
      copy_items: runtimeMetadataCopyItems(descriptor.runtimeRecord),
      embedded_workflow: descriptor.runtimeWorkflow || descriptor.runtimeRecord.embedded_workflow || null,
      sourceSignature: descriptorSignature(descriptor),
    });
    return;
  }

  if (!descriptor?.filename) {
    setMetadataState(node, sourceKey, {
      ...(descriptor || {}),
      loading: false,
      unavailable: true,
      copy_items: [],
      message: descriptor?.message || "No original image file could be traced from this connected source.",
      sourceSignature: descriptorSignature(descriptor),
    });
    return;
  }

  const source = {
    filename: descriptor.filename,
    type: descriptor.type || "input",
    subfolder: descriptor.subfolder || "",
    pathToken: descriptor.pathToken || "",
    originNode: descriptor.originNode || null,
    originWidget: descriptor.originWidget || null,
    sourceSignature: descriptorSignature(descriptor),
  };
  setMetadataState(node, sourceKey, { ...source, loading: true, copy_items: [] });

  try {
    const params = new URLSearchParams({
      filename: source.filename,
      type: source.type,
      subfolder: source.subfolder,
      path_token: source.pathToken,
    });
    const response = await fetch(`/kiri-image-loader/metadata?${params.toString()}`, {
      method: "GET",
      cache: "no-store",
    });
    const payload = await response.json();
    if (requestSerialByNode.get(node)?.get(sourceKey) !== serial) return;

    if (!response.ok) throw new Error(payload?.error || `HTTP ${response.status}`);
    setMetadataState(node, sourceKey, {
      ...payload,
      ...source,
      loading: false,
      unavailable: false,
      copy_items: Array.isArray(payload.copy_items) ? payload.copy_items : [],
    });
    node.setDirtyCanvas?.(true, true);
  } catch (error) {
    if (requestSerialByNode.get(node)?.get(sourceKey) !== serial) return;
    setMetadataState(node, sourceKey, {
      ...source,
      loading: false,
      error: error?.message || String(error),
      copy_items: [],
    });
    console.warn(`[Kiri Image Loader] ${sourceKey} metadata read failed:`, error);
  }
}

function loadedImageDescriptor(node) {
  const filename = currentFilename(node);
  return filename ? {
    filename,
    type: "input",
    subfolder: "",
    originNode: node,
    originWidget: getImageWidget(node),
    originKind: "loaded",
  } : { status: "unavailable", message: "No image is selected in Kiri Image Loader." };
}

function syncMetadataSource(node, sourceKey, descriptor) {
  const signature = descriptorSignature(descriptor);
  const current = getMetadataState(node, sourceKey);
  if (current?.sourceSignature === signature) return;
  void readMetadataForSource(node, sourceKey, descriptor);
}

function isSocketConnected(input) {
  try {
    return !!input && (
      (input.link !== undefined && input.link !== null) ||
      (Array.isArray(input.links) && input.links.length > 0)
    );
  } catch {
    return false;
  }
}

function hasConnectedImageInput(node) {
  try {
    return (node?.inputs || []).some((input) =>
      (input?.name === "image_in" || input?.name === "mask_in") && isSocketConnected(input)
    );
  } catch {
    return false;
  }
}

function isSubgraphBoundaryNode(node) {
  if (!node) return false;
  try {
    if (node.subgraph && typeof node.subgraph === "object") return true;
    if (typeof node.isSubgraphNode === "function" && node.isSubgraphNode()) return true;
    return String(node.constructor?.name || "") === "SubgraphNode";
  } catch {
    return false;
  }
}

function getOwningGraph(node) {
  try { return node?.graph || app.graph || null; }
  catch { return app.graph || null; }
}

function getGraphLink(graph, linkId) {
  if (!graph || linkId === undefined || linkId === null) return null;
  try {
    const direct = graph.getLink?.(linkId);
    if (direct) return direct;
  } catch { /* use the compatibility link maps below */ }
  try {
    const mapLink = graph.links?.get?.(linkId) || graph.links?.[linkId];
    if (mapLink) return mapLink;
  } catch { /* graph implementation may expose a protected store */ }
  return null;
}

function findInputLink(node, input) {
  const graph = getOwningGraph(node);
  const inputIndex = (() => {
    try { return (node?.inputs || []).indexOf(input); }
    catch { return -1; }
  })();

  // ComfyUI SubgraphNode overrides getInputLink(slot) to resolve the *internal
  // subgraph output* at that slot, not the host node's external input link.
  // Calling it here can return an unrelated link or index a missing output
  // slot, which is unsafe during the canvas draw loop. For subgraph boundaries
  // always resolve the parent-graph input slot's own link ID instead.
  if (!isSubgraphBoundaryNode(node) && inputIndex >= 0) {
    try {
      const link = node?.getInputLink?.(inputIndex);
      if (link) return link;
    } catch { /* fall through to the owning graph's link store */ }
  }

  const linkIds = [];
  try {
    if (input?.link !== undefined && input?.link !== null) linkIds.push(input.link);
  } catch { /* continue with compatibility link arrays */ }
  try {
    if (Array.isArray(input?.links)) linkIds.push(...input.links);
  } catch { /* ignore malformed slots during graph mutations */ }
  for (const linkId of linkIds) {
    const link = getGraphLink(graph, linkId);
    if (link) return link;
  }
  return null;
}

function getConnectedSource(node, inputName) {
  try {
    const input = node?.inputs?.find((item) => item?.name === inputName);
    if (!input || !isSocketConnected(input)) return null;
    const link = findInputLink(node, input);
    const graph = getOwningGraph(node);
    if (!link) return null;
    let sourceNode = null;
    try { sourceNode = graph?.getNodeById?.(link.origin_id) || null; }
    catch { /* fall back to the graph's node array */ }
    if (!sourceNode) {
      try {
        sourceNode = graph?._nodes?.find((candidate) => String(candidate.id) === String(link.origin_id)) || null;
      } catch { /* a graph may be in the middle of subgraph conversion */ }
    }
    if (!sourceNode) return null;
    const slot = Number(link.origin_slot ?? 0);
    return Number.isFinite(slot) ? { node: sourceNode, slot } : null;
  } catch {
    // Preview/metadata introspection must never break ComfyUI's graph renderer.
    return null;
  }
}

function resolveSubgraphOutputSource(subgraphNode, outputSlot) {
  if (!isSubgraphBoundaryNode(subgraphNode)) return null;
  const slot = Number(outputSlot);
  if (!Number.isInteger(slot) || slot < 0) return null;

  // Current ComfyUI exposes this public resolver on SubgraphNode. It follows
  // the selected output slot to the exact interior node/output pair.
  try {
    if (typeof subgraphNode.resolveSubgraphOutputLink === "function") {
      const resolved = subgraphNode.resolveSubgraphOutputLink(slot);
      if (resolved?.outputNode) {
        let sourceSlot = -1;
        try { sourceSlot = (resolved.outputNode.outputs || []).indexOf(resolved.output); }
        catch { /* use the link slot below */ }
        if (sourceSlot < 0) sourceSlot = Number(resolved.link?.origin_slot ?? 0);
        if (Number.isInteger(sourceSlot) && sourceSlot >= 0) {
          return { node: resolved.outputNode, slot: sourceSlot };
        }
      }
    }
  } catch {
    // A slot may be transiently absent while the graph is being converted.
  }

  // Compatibility fallback for frontend builds without resolveSubgraphOutputLink.
  try {
    const subgraph = subgraphNode.subgraph;
    const outputSlotObject = subgraph?.outputNode?.slots?.[slot];
    const innerLink = outputSlotObject?.getLinks?.()?.[0];
    const resolved = innerLink?.resolve?.(subgraph);
    if (resolved?.outputNode) {
      let sourceSlot = -1;
      try { sourceSlot = (resolved.outputNode.outputs || []).indexOf(resolved.output); }
      catch { /* use the link slot below */ }
      if (sourceSlot < 0) sourceSlot = Number(innerLink.origin_slot ?? 0);
      if (Number.isInteger(sourceSlot) && sourceSlot >= 0) {
        return { node: resolved.outputNode, slot: sourceSlot };
      }
    }
  } catch { /* safely report no preview/metadata source */ }
  return null;
}

function getNodePreviewLookupKeys(node) {
  const keys = [];
  const id = node?.id;
  if (id === undefined || id === null) return keys;
  try {
    const graphId = node?.graph?.id;
    if (graphId !== undefined && graphId !== null && String(graphId)) {
      // ComfyUI uses graph-scoped locator IDs for UI state because node IDs are
      // local to each graph/subgraph layer.
      keys.push(`${String(graphId)}:${String(id)}`);
    }
  } catch { /* use the legacy local ID key */ }
  keys.push(String(id));
  return [...new Set(keys)];
}

function reportPreviewIntegrationError(node, error, context = "preview") {
  if (node && typeof node === "object" && !previewIntegrationErrorsLogged.has(node)) {
    previewIntegrationErrorsLogged.add(node);
    console.warn(`[Kiri Image Loader] Isolated a ${context} integration error; ComfyUI rendering will continue.`, error);
  }
}

function isLoadImageNode(node) {
  const type = String(node?.comfyClass || node?.type || node?.constructor?.comfyClass || "");
  return type === "LoadImage" || type.endsWith(".LoadImage");
}

function getImageFilenameWidget(node) {
  return node?.widgets?.find((widget) => widget?.name === "image" && typeof widget.value === "string") ||
    node?.widgets?.find((widget) => /image/i.test(String(widget?.name || "")) && typeof widget.value === "string") ||
    null;
}

function descriptorFromOutputRecord(record, originNode = null) {
  if (!record) return null;
  let filename = typeof record === "object" ? record.filename : "";
  let type = typeof record === "object" ? String(record.type || "output").toLowerCase() : "output";
  let subfolder = typeof record === "object" ? String(record.subfolder || "") : "";
  const url = typeof record === "string" ? record : (record.url || record.preview_url || record.src || "");
  if (!filename && typeof url === "string" && url.includes("/view?")) {
    try {
      const parsed = new URL(url, window.location?.href || "http://localhost/");
      filename = parsed.searchParams.get("filename") || "";
      type = String(parsed.searchParams.get("type") || type).toLowerCase();
      subfolder = parsed.searchParams.get("subfolder") || subfolder;
    } catch { /* ignore nonstandard preview URLs */ }
  }
  const pathToken = typeof record === "object" ? String(record.path_token || "") : "";
  if (!filename || (!["input", "output", "temp"].includes(type) && !(type === "kiri_external" && pathToken))) return null;
  return {
    filename: String(filename),
    type,
    subfolder,
    pathToken,
    originNode,
    originWidget: null,
    originKind: "saved-output",
  };
}

function recordsForNode(node) {
  const id = String(node?.id ?? "");
  const candidates = [
    node?.images,
    node?.previewImages,
    app.nodeOutputs?.[id]?.images,
    app.nodeOutputs?.[id]?.output?.images,
  ];
  // A frontend node may hold decoded HTMLImageElements in `node.images` while
  // ComfyUI separately stores real filename records in `nodeOutputs`. Inspect
  // all sources rather than stopping at the first non-empty array.
  return candidates.flatMap((candidate) => Array.isArray(candidate) ? candidate : []);
}

function imageInputSources(node) {
  const result = [];
  for (const input of node?.inputs || []) {
    const type = String(input?.type || "").toUpperCase();
    const name = String(input?.name || "").toLowerCase();
    if (!(type.includes("IMAGE") || type.includes("MASK") || /^(image|images|mask|background|source_image)$/.test(name))) continue;
    if (!isSocketConnected(input)) continue;
    const source = getConnectedSource(node, input.name);
    if (source) result.push(source);
  }
  return result;
}

function uniqueDescriptors(descriptors) {
  const map = new Map();
  for (const descriptor of descriptors) {
    if (descriptor?.filename) map.set(descriptorSignature(descriptor), descriptor);
  }
  return [...map.values()];
}

function traceImageFileSource(sourceNode, sourceSlot = 0, visited = new Set(), context = { remaining: MAX_IMAGE_TRACE_NODES }) {
  if (!sourceNode) return { status: "unavailable", message: "The connected source node is unavailable." };
  if (!context || typeof context !== "object") context = { remaining: MAX_IMAGE_TRACE_NODES };
  if (!Number.isFinite(context.remaining)) context.remaining = MAX_IMAGE_TRACE_NODES;
  if (context.remaining <= 0) {
    return { status: "unavailable", message: "Image-source tracing stopped at its safety limit; the graph may be unusually large or recursive." };
  }
  context.remaining -= 1;
  // Node IDs are local to each graph layer. Use object identity, not node.id,
  // so an inner subgraph node with ID 12 is not mistaken for root node ID 12.
  if (visited.has(sourceNode)) return { status: "unavailable", message: "An image-source cycle was detected." };
  const nextVisited = new Set(visited);
  nextVisited.add(sourceNode);

  // A parent SubgraphNode is only a boundary. Resolve the requested output
  // slot to the exact interior source before walking image inputs. Do not call
  // its getInputLink() method: ComfyUI overrides that method for subgraph
  // output resolution, not normal host input lookup.
  if (isSubgraphBoundaryNode(sourceNode)) {
    const resolved = resolveSubgraphOutputSource(sourceNode, sourceSlot);
    if (!resolved) {
      return { status: "unavailable", message: "The selected subgraph output has no resolvable interior source." };
    }
    return traceImageFileSource(resolved.node, resolved.slot, nextVisited, context);
  }

  if (isKiriNode(sourceNode)) {
    // Output C is the dedicated selected-file output. It must not inherit A/B
    // graph provenance or accidentally resolve the node's saved A generation.
    if (Number(sourceSlot) === 2) {
      const loaded = loadedImageDescriptor(sourceNode);
      return loaded.filename ? loaded : {
        status: "unavailable",
        message: "This Kiri output C has no selected Loaded Image file.",
      };
    }
    // When Kiri A has already executed, prefer the exact PNG Kiri saved for
    // that output. This also preserves the opaque token used for absolute
    // save paths outside ComfyUI/output when another Kiri node is downstream.
    if (Number(sourceSlot) === 0 || Number(sourceSlot) === 1) {
      const executed = executionMetadataByNode.get(sourceNode);
      const savedOutput = Number(sourceSlot) === 0 ? executed?.savedOutputA : executed?.savedOutputB;
      if (savedOutput?.filename) return savedOutput;
    }
    const inputName = Number(sourceSlot) === 1 ? "mask_in" : "image_in";
    const input = sourceNode.inputs?.find((item) => item?.name === inputName);
    if (input && isSocketConnected(input)) {
      const upstream = getConnectedSource(sourceNode, inputName);
      if (upstream) return traceImageFileSource(upstream.node, upstream.slot, nextVisited, context);
    }
    const fallback = loadedImageDescriptor(sourceNode);
    return fallback.filename ? fallback : {
      status: "unavailable",
      message: "This Kiri output has no connected file source or selected fallback image.",
    };
  }

  if (isLoadImageNode(sourceNode)) {
    const widget = getImageFilenameWidget(sourceNode);
    const filename = typeof widget?.value === "string" ? widget.value : "";
    if (filename) {
      return {
        filename,
        type: "input",
        subfolder: "",
        originNode: sourceNode,
        originWidget: widget,
        originKind: "load-image",
      };
    }
    return { status: "unavailable", message: "The upstream Load Image node has no selected file." };
  }

  // Saved preview records identify real ComfyUI output/temp files and can be
  // inspected even when the producing node is not a Load Image node.
  const outputDescriptors = uniqueDescriptors(recordsForNode(sourceNode)
    .map((record) => descriptorFromOutputRecord(record, sourceNode))
    .filter(Boolean));
  if (outputDescriptors.length === 1) return outputDescriptors[0];
  if (outputDescriptors.length > 1) {
    return { status: "ambiguous", message: "This node exposes multiple saved images; their file origins cannot be inferred from this socket alone." };
  }

  // For pass-through/transform nodes, walk upstream image inputs. The shared
  // budget prevents a wide DAG or malformed graph from causing exponential
  // work during repeated canvas redraws.
  const inherited = [];
  for (const parent of imageInputSources(sourceNode)) {
    if (context.remaining <= 0) break;
    const descriptor = traceImageFileSource(parent.node, parent.slot, nextVisited, context);
    if (descriptor?.filename) inherited.push(descriptor);
  }
  const unique = uniqueDescriptors(inherited);
  if (unique.length === 1) {
    return { ...unique[0], originKind: "traced-upstream", tracedThrough: true };
  }
  if (unique.length > 1) {
    return { status: "ambiguous", message: "This image combines multiple upstream files, so Kiri cannot safely choose one metadata source." };
  }
  return {
    status: "unavailable",
    message: "This connected image has no recoverable source file. Its pixels are available for preview and passthrough, but embedded file metadata and file actions are unavailable.",
  };
}

function connectedImageDescriptor(node, sourceKey) {
  const inputName = sourceKey === "image_a" ? "image_in" : "mask_in";
  const input = node?.inputs?.find((item) => item?.name === inputName);
  if (!input || !isSocketConnected(input)) return null;
  const source = getConnectedSource(node, inputName);
  if (!source) return { status: "unavailable", message: "The connected input link could not be resolved." };
  return traceImageFileSource(source.node, source.slot);
}

function syncAllMetadataSources(node) {
  try {
    syncMetadataSource(node, "loaded", loadedImageDescriptor(node));
    const inputA = node.inputs?.find((input) => input?.name === "image_in");
    const inputB = node.inputs?.find((input) => input?.name === "mask_in");
    if (inputA && isSocketConnected(inputA)) syncMetadataSource(node, "image_a", effectiveImageDescriptor(node, "image_a"));
    else syncMetadataSource(node, "image_a", null);
    if (inputB && isSocketConnected(inputB)) syncMetadataSource(node, "image_b", effectiveImageDescriptor(node, "image_b"));
    else syncMetadataSource(node, "image_b", null);
  } catch (error) {
    reportPreviewIntegrationError(node, error, "metadata-source resolution");
  }
}

function imageUrlFromOutputRecord(record) {
  if (typeof record === "string") return record;
  if (!record || typeof record !== "object") return "";
  if (typeof record.url === "string") return record.url;
  if (typeof record.preview_url === "string") return record.preview_url;
  if (typeof record.src === "string") return record.src;
  if (!record.filename) return "";
  const params = new URLSearchParams({
    filename: String(record.filename),
    type: String(record.type || "output"),
  });
  if (record.subfolder) params.set("subfolder", String(record.subfolder));
  return `/view?${params.toString()}`;
}

function loadUpstreamImageUrl(url) {
  if (!url) return null;
  let cached = upstreamImageUrlCache.get(url);
  if (cached?.image?.complete && (cached.image.naturalWidth || cached.image.width)) return cached.image;
  if (cached?.loading) return null;
  const image = cached?.image || new Image();
  cached = { image, loading: true };
  upstreamImageUrlCache.set(url, cached);
  image.onload = () => {
    cached.loading = false;
    cached.failed = false;
    for (const node of app.graph?._nodes || []) {
      if (isKiriNode(node) && hasConnectedImageInput(node)) {
        syncLivePreviewFromUpstream(node);
        node.setDirtyCanvas?.(true, true);
      }
    }
  };
  image.onerror = () => {
    cached.loading = false;
    cached.failed = true;
  };
  image.src = url;
  return image.complete && (image.naturalWidth || image.width) ? image : null;
}

function isRenderableImageSource(image) {
  if (!image) return false;
  const width = Number(image.naturalWidth || image.videoWidth || image.width || 0);
  const height = Number(image.naturalHeight || image.videoHeight || image.height || 0);
  return Number.isFinite(width) && Number.isFinite(height) && width > 0 && height > 0;
}

function getSourceNodePreviewImage(sourceNode, sourceSlot = 0, visited = new Set(), depth = 0) {
  if (!sourceNode || depth > 16 || visited.has(sourceNode)) return null;
  const nextVisited = new Set(visited);
  nextVisited.add(sourceNode);

  // SubgraphNode output previews belong to the exact interior node/output
  // represented by that boundary slot. Resolve recursively so nested
  // subgraphs and chained Kiri nodes are handled without guessing by name.
  if (isSubgraphBoundaryNode(sourceNode)) {
    const resolved = resolveSubgraphOutputSource(sourceNode, sourceSlot);
    return resolved
      ? getSourceNodePreviewImage(resolved.node, resolved.slot, nextVisited, depth + 1)
      : null;
  }

  // When Kiri nodes are chained, prefer the preview for the exact output slot
  // rather than the node's combined A/B comparison image. Ignore empty or
  // stale image/canvas objects from upstream nodes that have not executed yet.
  if (isKiriNode(sourceNode)) {
    if (Number(sourceSlot) === 2) {
      // C is always the selected Loaded Image, independently of connected A/B.
      // Read the native selected-file preview, never the composite canvas.
      return getLoadedFallbackPreviewImages(sourceNode)[0] || null;
    }
    const branchInputName = Number(sourceSlot) === 0 ? "image_in" : Number(sourceSlot) === 1 ? "mask_in" : null;
    if (!branchInputName) return null;
    const branchInput = sourceNode.inputs?.find((input) => input?.name === branchInputName);
    const branchIsConnected = isSocketConnected(branchInput);
    // Kiri's backend defaults an unconnected A or B output to the selected
    // Loaded Image. Its native `imgs` is a combined display surface, so only
    // use the saved native file preview here when this exact output branch is
    // genuinely unconnected. A connected-but-not-yet-executed branch must
    // remain unavailable rather than leaking a stale/file fallback preview.
    if (!branchIsConnected) return getLoadedFallbackPreviewImages(sourceNode)[0] || null;
    const kiriState = livePreviewByNode.get(sourceNode);
    const slotImage = Number(sourceSlot) === 0 ? kiriState?.a : kiriState?.b;
    // Never substitute the node's combined canvas for an empty connected slot.
    return isRenderableImageSource(slotImage) ? slotImage : null;
  }

  // Some frontend objects expose lazy/proxy properties; one failing preview
  // getter must not abort the whole ComfyUI draw loop.
  let sourceImages = [];
  try { sourceImages = Array.isArray(sourceNode.imgs) ? sourceNode.imgs : []; }
  catch { sourceImages = []; }
  if (sourceImages.length) {
    let selectedIndex = 0;
    try { selectedIndex = Number.isInteger(sourceNode.imageIndex) ? sourceNode.imageIndex : 0; }
    catch { selectedIndex = 0; }
    const image = sourceImages[Math.min(Math.max(selectedIndex, 0), sourceImages.length - 1)];
    if (isRenderableImageSource(image)) return image;
  }

  const previewLists = [];
  try { if (Array.isArray(sourceNode.previewImages)) previewLists.push(sourceNode.previewImages); } catch {}
  try { if (Array.isArray(sourceNode.images)) previewLists.push(sourceNode.images); } catch {}
  for (const id of getNodePreviewLookupKeys(sourceNode)) {
    try { if (Array.isArray(app.nodePreviewImages?.[id])) previewLists.push(app.nodePreviewImages[id]); } catch {}
    try { if (Array.isArray(app.nodeOutputs?.[id]?.images)) previewLists.push(app.nodeOutputs[id].images); } catch {}
    try { if (Array.isArray(app.nodeOutputs?.[id]?.output?.images)) previewLists.push(app.nodeOutputs[id].output.images); } catch {}
  }
  for (const list of previewLists) {
    if (!list.length) continue;
    const record = list[list.length - 1];
    const url = imageUrlFromOutputRecord(record);
    if (!url) continue;
    const image = loadUpstreamImageUrl(url);
    if (image) return image;
  }
  return null;
}

function makeLoadImageMaskPreview(image) {
  if (!image) return null;
  const cached = alphaMaskPreviewCache.get(image);
  if (cached) return cached;
  const width = Math.max(1, Number(image.naturalWidth || image.width));
  const height = Math.max(1, Number(image.naturalHeight || image.height));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return image;
  try {
    ctx.drawImage(image, 0, 0, width, height);
    const frame = ctx.getImageData(0, 0, width, height);
    const pixels = frame.data;
    for (let i = 0; i < pixels.length; i += 4) {
      // ComfyUI LoadImage emits an inverted alpha mask (1 - alpha).
      const value = 255 - pixels[i + 3];
      pixels[i] = value;
      pixels[i + 1] = value;
      pixels[i + 2] = value;
      pixels[i + 3] = 255;
    }
    ctx.putImageData(frame, 0, 0);
    Object.defineProperty(canvas, "naturalWidth", { configurable: true, get: () => canvas.width });
    Object.defineProperty(canvas, "naturalHeight", { configurable: true, get: () => canvas.height });
    alphaMaskPreviewCache.set(image, canvas);
    return canvas;
  } catch (error) {
    console.warn("[Kiri Image Loader] Could not derive upstream LoadImage mask preview:", error);
    return image;
  }
}

function resolveConnectedPreview(node, inputName) {
  const source = getConnectedSource(node, inputName);
  if (!source) return null;
  let image = getSourceNodePreviewImage(source.node, source.slot);
  if (!isRenderableImageSource(image)) return null;

  const output = source.node.outputs?.[source.slot];
  const outputType = String(output?.type || "").toUpperCase();
  const outputName = String(output?.name || "").toLowerCase();
  const isLoadImage = source.node.comfyClass === "LoadImage" ||
    source.node.type === "LoadImage" || source.node.constructor?.comfyClass === "LoadImage";
  if (isLoadImage && source.slot === 1 && (outputType === "MASK" || outputName === "mask")) {
    image = makeLoadImageMaskPreview(image);
  }
  return image;
}

function syncLivePreviewFromUpstream(node) {
  try {
    if (!hasConnectedImageInput(node)) return false;
    const inputA = node.inputs?.find((input) => input?.name === "image_in");
    const inputB = node.inputs?.find((input) => input?.name === "mask_in");
    const imageA = inputA && isSocketConnected(inputA) ? resolveConnectedPreview(node, "image_in") : null;
    const imageB = inputB && isSocketConnected(inputB) ? resolveConnectedPreview(node, "mask_in") : null;
    const previous = livePreviewByNode.get(node);
    if (!imageA && !imageB) {
      // A connected branch may legitimately have no preview before first
      // execution. Remove stale graph pixels rather than leaving an old image
      // visible or passing it into the comparison renderer.
      if (previous?.source === "graph") {
        livePreviewByNode.delete(node);
        node.setDirtyCanvas?.(true, true);
      }
      return false;
    }

    if (previous?.source === "graph" && previous.a === imageA && previous.b === imageB) return false;
    const state = {
      source: "graph",
      mode: imageA && imageB ? "compare" : "single",
      a: imageA,
      b: imageB,
      slider: previous?.slider ?? 0.5,
    };
    state.renderedImage = imageA && imageB
      ? createComparisonCanvas(imageA, imageB, state.slider, getComparisonSidesSwapped(node))
      : (imageA || imageB);
    livePreviewByNode.set(node, state);
    return true;
  } catch (error) {
    reportPreviewIntegrationError(node, error, "upstream preview");
    return false;
  }
}

function findPropertyDescriptor(object, property) {
  let current = object;
  while (current) {
    const descriptor = Object.getOwnPropertyDescriptor(current, property);
    if (descriptor) return descriptor;
    current = Object.getPrototypeOf(current);
  }
  return null;
}

function installNativePreviewSource(node) {
  if (!node || previewSourceInstalledNodes.has(node)) return;
  previewSourceInstalledNodes.add(node);

  const ownDescriptor = Object.getOwnPropertyDescriptor(node, "imgs");
  const inheritedDescriptor = ownDescriptor ? null : findPropertyDescriptor(
    Object.getPrototypeOf(node),
    "imgs"
  );
  const sourceDescriptor = ownDescriptor || inheritedDescriptor;
  let fallbackImages = ownDescriptor && "value" in ownDescriptor
    ? ownDescriptor.value
    : undefined;

  const readFallback = () => {
    if (sourceDescriptor?.get) {
      try {
        const value = sourceDescriptor.get.call(node);
        if (value !== undefined) fallbackImages = value;
      } catch { /* keep the last assigned fallback */ }
    }
    return Array.isArray(fallbackImages) ? fallbackImages : [];
  };
  nativePreviewFallbackReaders.set(node, readFallback);

  try {
    Object.defineProperty(node, "imgs", {
      configurable: true,
      enumerable: sourceDescriptor?.enumerable ?? true,
      get() {
        // While either comparison input is connected, route the native preview
        // through the explicit A/B/Loaded display policy. Auto mode uses the
        // selected file only when neither connected branch has a usable preview.
        const mode = getVisiblePreviewMode(node);
        if (hasConnectedImageInput(node) || mode !== "auto") {
          return getVisiblePreviewImages(node);
        }
        return readFallback();
      },
      set(value) {
        // ComfyUI can asynchronously replace its selected-file preview after a
        // user changes the selector or uploads a new image. Store that normally
        // for the Loaded Image-only choice and Auto mode's final fallback.
        if (sourceDescriptor?.set) {
          sourceDescriptor.set.call(node, value);
        } else {
          fallbackImages = value;
        }
      },
    });
  } catch (error) {
    previewSourceInstalledNodes.delete(node);
    console.warn("[Kiri Image Loader] Could not install native preview source:", error);
  }
}

function patchNativePreviewWidgets(node) {
  if (!node?.widgets) return;
  for (const widget of node.widgets) {
    if (!widget || typeof widget.drawWidget !== "function" || patchedPreviewWidgets.has(widget)) continue;

    const constructorName = String(widget.constructor?.name || "");
    const widgetName = String(widget.name || "");
    const isImagePreview = constructorName.includes("ImagePreviewWidget") ||
      /image.?preview/i.test(widgetName) || widget.type === "image_preview";
    if (!isImagePreview) continue;

    const originalDrawWidget = widget.drawWidget;
    widget.drawWidget = function (ctx, options = {}, ...rest) {
      const mode = getVisiblePreviewMode(node);
      if (!hasConnectedImageInput(node) && mode === "auto") {
        return originalDrawWidget.call(this, ctx, options, ...rest);
      }

      const images = getVisiblePreviewImages(node);
      // The modern Vue/LiteGraph preview renderer prefers options.previewImages
      // over node.imgs. Override that explicit argument as well as node.imgs so
      // the selected-file preview cannot win the draw-order race.
      const liveOptions = options && typeof options === "object"
        ? { ...options, previewImages: images }
        : { previewImages: images };
      return originalDrawWidget.call(this, ctx, liveOptions, ...rest);
    };
    patchedPreviewWidgets.add(widget);
  }
}

function previewAssetForPersistence(record) {
  if (!record) return null;
  if (typeof record === "string") {
    if (!record) return null;
    const mime = record.startsWith("data:")
      ? (record.match(/^data:([^;,]+)/)?.[1] || "image/jpeg")
      : "image/jpeg";
    return { data: record, mime };
  }
  if (typeof record !== "object") return null;
  const data = record.data || record.base64 || record.image || "";
  if (typeof data !== "string" || !data) return null;
  const mime = record.mime || (data.startsWith("data:")
    ? (data.match(/^data:([^;,]+)/)?.[1] || "image/jpeg")
    : "image/jpeg");
  return { data, mime };
}

function currentLivePreviewSourceIdentities(node) {
  return {
    image_a: connectedSourceIdentity(node, "image_a"),
    image_b: connectedSourceIdentity(node, "image_b"),
  };
}

function persistLivePreview(node, payload, state) {
  try {
    if (!node || typeof node !== "object") return false;
    const a = previewAssetForPersistence(payload?.a);
    const b = previewAssetForPersistence(payload?.b);
    if (!a && !b) {
      clearPersistedLivePreview(node);
      return false;
    }
    const size = (a?.data.length || 0) + (b?.data.length || 0);
    if (size > MAX_PERSISTED_LIVE_PREVIEW_CHARS) {
      // Avoid turning a normal workflow into an unexpectedly huge JSON file.
      clearPersistedLivePreview(node);
      console.info("[Kiri Image Loader] This comparison preview is too large to embed in the workflow; it will remain available until the workflow tab is rebuilt.");
      return false;
    }
    if (!node.properties || typeof node.properties !== "object") node.properties = {};
    node.properties[PERSISTED_LIVE_PREVIEW_PROPERTY] = {
      version: 1,
      mode: state?.mode === "compare" && a && b ? "compare" : "single",
      a,
      b,
      slider: Math.max(0, Math.min(1, Number(state?.slider ?? 0.5))),
      sourceIdentities: currentLivePreviewSourceIdentities(node),
    };
    return true;
  } catch (error) {
    console.warn("[Kiri Image Loader] Could not persist the live comparison preview in the workflow:", error);
    return false;
  }
}

function clearPersistedLivePreview(node) {
  try {
    const properties = node?.properties;
    if (properties && Object.prototype.hasOwnProperty.call(properties, PERSISTED_LIVE_PREVIEW_PROPERTY)) {
      delete properties[PERSISTED_LIVE_PREVIEW_PROPERTY];
    }
  } catch { /* never let optional preview cleanup affect the host graph */ }
}

function persistedPreviewConnectionStatus(node, saved) {
  const expected = saved?.sourceIdentities;
  if (!expected || typeof expected !== "object") return "match";
  const current = currentLivePreviewSourceIdentities(node);
  for (const [sourceKey, inputName] of [["image_a", "image_in"], ["image_b", "mask_in"]]) {
    if ((expected[sourceKey] ?? null) === (current[sourceKey] ?? null)) continue;
    const input = node?.inputs?.find((item) => item?.name === inputName);
    // During workflow reconstruction a link can exist before its source node
    // is resolvable. Wait for a later canvas pass rather than deleting a valid
    // snapshot merely because ComfyUI is midway through restoring the graph.
    if (isSocketConnected(input) && current[sourceKey] === null) return "pending";
    return "mismatch";
  }
  return "match";
}

function persistComparisonSlider(node, state) {
  try {
    const saved = node?.properties?.[PERSISTED_LIVE_PREVIEW_PROPERTY];
    if (saved && state) saved.slider = Math.max(0, Math.min(1, Number(state.slider ?? 0.5)));
  } catch { /* preview interaction must remain independent of host properties */ }
}

async function restorePersistedLivePreview(node) {
  if (!node || persistedPreviewRestorePending.has(node)) return false;
  const saved = node.properties?.[PERSISTED_LIVE_PREVIEW_PROPERTY];
  if (!saved || saved.version !== 1 || (!saved.a && !saved.b)) return false;
  const connectionStatus = persistedPreviewConnectionStatus(node, saved);
  if (connectionStatus === "mismatch") {
    clearPersistedLivePreview(node);
    return false;
  }
  if (connectionStatus === "pending") return false;

  persistedPreviewRestorePending.add(node);
  const serial = (livePreviewSerialByNode.get(node) || 0) + 1;
  livePreviewSerialByNode.set(node, serial);
  const state = {
    source: "execution",
    restoring: true,
    mode: saved.mode === "compare" ? "compare" : "single",
    a: null,
    b: null,
    slider: Math.max(0, Math.min(1, Number(saved.slider ?? 0.5))),
  };
  livePreviewByNode.set(node, state);
  try {
    const [imageA, imageB] = await Promise.all([
      loadPreviewAsset(saved.a),
      loadPreviewAsset(saved.b),
    ]);
    const restoredConnectionStatus = persistedPreviewConnectionStatus(node, saved);
    if (livePreviewSerialByNode.get(node) !== serial ||
        livePreviewByNode.get(node) !== state ||
        node.properties?.[PERSISTED_LIVE_PREVIEW_PROPERTY] !== saved ||
        restoredConnectionStatus !== "match") {
      if (livePreviewByNode.get(node) === state) livePreviewByNode.delete(node);
      if (restoredConnectionStatus === "mismatch") clearPersistedLivePreview(node);
      return false;
    }
    state.a = imageA;
    state.b = imageB;
    state.restoring = false;
    if (!state.a && !state.b) {
      livePreviewByNode.delete(node);
      clearPersistedLivePreview(node);
      refreshNativeLivePreview(node, null);
      return false;
    }
    state.mode = state.a && state.b ? "compare" : "single";
    refreshNativeLivePreview(node, state);
    syncAllMetadataSources(node);
    return true;
  } catch (error) {
    if (livePreviewByNode.get(node) === state) livePreviewByNode.delete(node);
    console.warn("[Kiri Image Loader] Could not restore the saved comparison preview:", error);
    return false;
  } finally {
    persistedPreviewRestorePending.delete(node);
  }
}

function attachAutomaticReader(node, retry = 0) {
  if (!isKiriNode(node) || attachedNodes.has(node)) return;

  // Keep legacy socket IDs so older Kiri workflows retain their connections,
  // while presenting clearer A/B labels on the canvas.
  for (const [socketName, label] of [["image_in", "A"], ["mask_in", "B"], ["custom_metadata", "CUSTOM META"]]) {
    const slot = node.inputs?.find((input) => input?.name === socketName);
    if (slot) slot.label = label;
  }
  hideMetadataSettingsWidgets(node);
  for (const [name, label] of [["save_path", "Save Path"], ["filename_template", "File Name"]]) {
    const setting = node?.widgets?.find((item) => item?.name === name);
    if (setting) setting.label = label;
  }

  const widget = getImageWidget(node);
  if (!widget) {
    // Some ComfyUI frontend builds finish creating widgets just after nodeCreated.
    if (retry < 8) setTimeout(() => attachAutomaticReader(node, retry + 1), 80);
    return;
  }
  attachedNodes.add(node);
  installNativePreviewSource(node);
  patchNativePreviewWidgets(node);
  installPreviewInteractionSurface(node);

  const originalCallback = widget.callback;
  widget.callback = function (...args) {
    const result = originalCallback?.apply(this, args);
    queueMicrotask(() => {
      syncMetadataSource(node, "loaded", loadedImageDescriptor(node));
      syncAllMetadataSources(node);
    });
    return result;
  };

  const originalOnWidgetChanged = node.onWidgetChanged;
  node.onWidgetChanged = function (name, value, oldValue, widgetArg) {
    const result = originalOnWidgetChanged?.call(this, name, value, oldValue, widgetArg);
    if (name === "image") {
      queueMicrotask(() => {
        syncMetadataSource(node, "loaded", loadedImageDescriptor(node));
        syncAllMetadataSources(node);
      });
    }
    return result;
  };

  const originalOnConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = originalOnConfigure?.apply(this, args);
    queueMicrotask(() => {
      hideMetadataSettingsWidgets(node);
      void restorePersistedLivePreview(node);
      syncAllMetadataSources(node);
      node.setDirtyCanvas?.(true, true);
    });
    return result;
  };

  const originalOnConnectionsChange = node.onConnectionsChange;
  node.onConnectionsChange = function (...args) {
    const result = originalOnConnectionsChange?.apply(this, args);
    const slotIndex = Number(args[1]);
    const slot = this.inputs?.[slotIndex];
    if (slot?.name === "image_in" || slot?.name === "mask_in") {
      // A newly connected source must not leave the previous file preview
      // visible while waiting for its first evaluated tensor. Disconnecting
      // restores the selected image's own preview immediately.
      livePreviewByNode.delete(this);
      livePreviewSerialByNode.set(this, (livePreviewSerialByNode.get(this) || 0) + 1);
      this.setDirtyCanvas?.(true, true);
      queueMicrotask(() => {
        syncLivePreviewFromUpstream(this);
        syncAllMetadataSources(this);
        this.setDirtyCanvas?.(true, true);
      });
    }
    return result;
  };

  // Image upload widgets can update the combo programmatically after upload.
  const originalOnExecuted = node.onExecuted;
  node.onExecuted = function (...args) {
    const result = originalOnExecuted?.apply(this, args);
    void updateLivePreview(node, args[0] || {}).catch((error) => {
      console.warn("[Kiri Image Loader] Could not process execution metadata/preview:", error);
      syncAllMetadataSources(node);
    });
    return result;
  };

  queueMicrotask(() => syncAllMetadataSources(node));
}

async function copyText(value) {
  const text = String(value ?? "");
  if (!text) return;

  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return;
    }
  } catch (error) {
    console.debug("[Kiri Image Loader] Clipboard API unavailable; using fallback.", error);
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  try {
    document.execCommand("copy");
  } finally {
    textarea.remove();
  }
}

function imageDimensions(source) {
  if (!isRenderableImageSource(source)) return null;
  return {
    width: Math.round(Number(source.naturalWidth || source.videoWidth || source.width)),
    height: Math.round(Number(source.naturalHeight || source.videoHeight || source.height)),
  };
}

function canCopyImageSource(node, sourceKey) {
  if (sourceKey === "loaded") return !!loadedImageDescriptor(node)?.filename;
  const inputName = sourceKey === "image_a" ? "image_in" : "mask_in";
  const input = node?.inputs?.find((item) => item?.name === inputName);
  if (!isSocketConnected(input)) return false;
  const descriptor = effectiveImageDescriptor(node, sourceKey);
  const isActualSavedOutput = descriptor?.originKind === "kiri-generated-output" ||
    (descriptor?.originKind === "saved-output" && !descriptor?.tracedThrough);
  if (descriptor?.filename && (!descriptor.tracedThrough || isActualSavedOutput)) return true;
  const state = livePreviewByNode.get(node);
  const preview = sourceKey === "image_a" ? state?.a : state?.b;
  return isRenderableImageSource(preview) || isRenderableImageSource(resolveConnectedPreview(node, inputName));
}

function imageDescriptorUrl(descriptor) {
  if (!descriptor?.filename) return "";
  const params = new URLSearchParams({
    filename: descriptor.filename,
    type: descriptor.type || "input",
    subfolder: descriptor.subfolder || "",
  });
  if (descriptor.pathToken) {
    params.set("path_token", descriptor.pathToken);
    return `/kiri-image-loader/image?${params.toString()}`;
  }
  return `/view?${params.toString()}`;
}

async function fetchImageDescriptorBlob(descriptor) {
  const url = imageDescriptorUrl(descriptor);
  if (!url) throw new Error("This source has no retrievable image file.");
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`Could not read the source image (HTTP ${response.status}).`);
  const blob = await response.blob();
  if (!blob || !blob.size || !String(blob.type || "").startsWith("image/")) {
    throw new Error("The source did not return a valid image.");
  }
  return blob;
}

async function decodeImageBlob(blob) {
  if (typeof createImageBitmap === "function") {
    return { image: await createImageBitmap(blob), closeAfter: true };
  }
  return new Promise((resolve, reject) => {
    const objectUrl = URL.createObjectURL(blob);
    const image = new Image();
    image.onload = () => resolve({ image, closeAfter: false, objectUrl });
    image.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      reject(new Error("The clipboard image could not be decoded."));
    };
    image.src = objectUrl;
  });
}

async function renderImageBlobAsPng(blob) {
  const decoded = await decodeImageBlob(blob);
  const image = decoded.image;
  try {
    const dimensions = imageDimensions(image);
    if (!dimensions) throw new Error("The selected source contains no usable image pixels.");
    if (dimensions.width * dimensions.height > 64 * 1024 * 1024) {
      throw new Error("This image is too large to place on the clipboard safely (over 64 megapixels).");
    }
    const canvas = document.createElement("canvas");
    canvas.width = dimensions.width;
    canvas.height = dimensions.height;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("The browser could not create an image canvas.");
    context.drawImage(image, 0, 0, dimensions.width, dimensions.height);
    const png = await new Promise((resolve, reject) => {
      canvas.toBlob((result) => result ? resolve(result) : reject(new Error("The browser could not encode the image for the clipboard.")), "image/png");
    });
    if (!png?.size) throw new Error("The browser produced an empty clipboard image.");
    return png;
  } finally {
    if (decoded.closeAfter && typeof image.close === "function") image.close();
    if (decoded.objectUrl) URL.revokeObjectURL(decoded.objectUrl);
  }
}

async function sourceImageBlobForClipboard(node, sourceKey) {
  if (sourceKey === "loaded") return fetchImageDescriptorBlob(loadedImageDescriptor(node));

  const inputName = sourceKey === "image_a" ? "image_in" : "mask_in";
  const descriptor = effectiveImageDescriptor(node, sourceKey);
  const state = livePreviewByNode.get(node);
  const preview = sourceKey === "image_a" ? state?.a : state?.b;
  const isActualSavedOutput = descriptor?.originKind === "kiri-generated-output" ||
    (descriptor?.originKind === "saved-output" && !descriptor?.tracedThrough);

  // Prefer exact saved output files. For transformed nodes, prefer the live
  // output preview over an inherited source file because those pixels differ.
  if (descriptor?.filename && (!descriptor.tracedThrough || isActualSavedOutput)) {
    try { return await fetchImageDescriptorBlob(descriptor); }
    catch (error) { if (!isRenderableImageSource(preview)) throw error; }
  }

  const image = isRenderableImageSource(preview) ? preview : resolveConnectedPreview(node, inputName);
  if (isRenderableImageSource(image)) {
    const dimensions = imageDimensions(image);
    if (dimensions.width * dimensions.height > 64 * 1024 * 1024) {
      throw new Error("This image is too large to place on the clipboard safely (over 64 megapixels).");
    }
    const canvas = document.createElement("canvas");
    canvas.width = dimensions.width;
    canvas.height = dimensions.height;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("The browser could not create an image canvas.");
    context.drawImage(image, 0, 0, dimensions.width, dimensions.height);
    return new Promise((resolve, reject) => {
      canvas.toBlob((result) => result ? resolve(result) : reject(new Error("Could not encode the source preview.")), "image/png");
    });
  }
  throw new Error("This source does not currently expose its own image pixels to copy. Run the upstream node once, then try again; Kiri will not substitute an ancestor image.");
}

function copyImageSourceToClipboard(node, sourceKey, label) {
  const clipboard = navigator.clipboard;
  const ClipboardItemCtor = globalThis.ClipboardItem || window.ClipboardItem;
  if (!clipboard?.write || typeof ClipboardItemCtor !== "function") {
    window.alert("Kiri Image Loader: This browser does not support copying image pixels directly to the clipboard. Try a Chromium-based browser with clipboard access enabled.");
    return;
  }
  // Promise-backed ClipboardItems allow image loading to continue while the
  // clipboard write is initiated inside the menu's user activation.
  const pngPromise = sourceImageBlobForClipboard(node, sourceKey).then(renderImageBlobAsPng);
  try {
    const item = new ClipboardItemCtor({ "image/png": pngPromise });
    clipboard.write([item]).then(() => {
      console.info(`[Kiri Image Loader] Copied ${label} pixels to the clipboard.`);
    }).catch((error) => {
      console.warn("[Kiri Image Loader] Image clipboard copy failed:", error);
      window.alert(`Kiri Image Loader: Could not copy ${label}. ${error?.message || String(error)}`);
    });
  } catch (error) {
    console.warn("[Kiri Image Loader] Could not initialize image clipboard copy:", error);
    window.alert(`Kiri Image Loader: Could not copy ${label}. ${error?.message || String(error)}`);
  }
}

async function uploadClipboardImageToLoadedImage(node, blob) {
  if (!blob || !blob.size || !String(blob.type || "").startsWith("image/")) throw new Error("The clipboard does not contain a supported image.");
  const widget = getImageWidget(node);
  if (!widget) throw new Error("The Loaded Image selector is unavailable on this node.");
  // Normalize clipboard formats (including browser-specific JPEG/WebP variants)
  // to a real PNG before handing them to ComfyUI's LoadImage selector.
  const pngBlob = await renderImageBlobAsPng(blob);
  const file = new File([pngBlob], `Kiri Clipboard ${Date.now()}.png`, { type: "image/png" });
  const form = new FormData();
  form.append("image", file);
  form.append("type", "input");
  const response = await fetch("/upload/image", { method: "POST", body: form });
  let payload = {};
  try { payload = await response.json(); } catch { /* reported below */ }
  if (!response.ok || !payload?.name) throw new Error(payload?.error || `ComfyUI rejected the image upload (HTTP ${response.status}).`);
  const selectedPath = [String(payload.subfolder || "").replace(/^\/+|\/+$/g, ""), String(payload.name)].filter(Boolean).join("/");
  await refreshManagedImageWidget(node, selectedPath);
  widget.options = widget.options || {};
  const values = Array.isArray(widget.options.values) ? [...widget.options.values] : [];
  if (!values.includes(selectedPath)) values.unshift(selectedPath);
  widget.options.values = values;
  if (widget.value !== selectedPath) {
    widget.value = selectedPath;
    widget.callback?.(selectedPath, app.canvas, node);
  }
  syncAllMetadataSources(node);
  node.setDirtyCanvas?.(true, true);
  console.info(`[Kiri Image Loader] Pasted clipboard image into Loaded Image: ${selectedPath}`);
  return selectedPath;
}

function showClipboardPasteFallback(node) {
  const overlay = document.createElement("div");
  overlay.style.cssText = "position:fixed;inset:0;z-index:10020;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;padding:20px;";
  const panel = document.createElement("div");
  panel.style.cssText = "width:min(440px,92vw);background:#202329;color:#eee;border:1px solid #4a4f58;border-radius:10px;padding:18px;box-shadow:0 12px 36px rgba(0,0,0,.45);font:13px sans-serif;";
  const title = document.createElement("div");
  title.textContent = "Paste an image into Loaded Image";
  title.style.cssText = "font-size:15px;font-weight:600;margin-bottom:8px;";
  const note = document.createElement("div");
  note.textContent = "Click the area below, then press Ctrl+V (or Cmd+V). Kiri will upload the clipboard image into ComfyUI's input folder.";
  note.style.cssText = "line-height:1.45;color:#c9cdd4;margin-bottom:12px;";
  const target = document.createElement("div");
  target.contentEditable = "true";
  target.tabIndex = 0;
  target.textContent = "Click here, then paste your image…";
  target.style.cssText = "min-height:72px;border:1px dashed #77808d;border-radius:7px;padding:14px;outline:none;background:#171a1f;color:#aeb5c0;";
  const close = document.createElement("button");
  close.textContent = "Cancel";
  close.style.cssText = "margin-top:12px;padding:7px 12px;border:1px solid #555d68;border-radius:6px;background:#303640;color:#fff;cursor:pointer;";
  const remove = () => overlay.remove();
  close.addEventListener("click", remove);
  target.addEventListener("paste", (event) => {
    const items = Array.from(event.clipboardData?.items || []);
    const imageItem = items.find((item) => item?.kind === "file" && String(item.type || "").startsWith("image/"));
    const file = imageItem?.getAsFile?.();
    if (!file) { window.alert("Kiri Image Loader: No image was found on the clipboard."); return; }
    event.preventDefault();
    remove();
    void uploadClipboardImageToLoadedImage(node, file).catch((error) => {
      console.warn("[Kiri Image Loader] Clipboard image upload failed:", error);
      window.alert(`Kiri Image Loader: Could not paste the image. ${error?.message || String(error)}`);
    });
  });
  panel.append(title, note, target, close);
  overlay.appendChild(panel);
  overlay.addEventListener("pointerdown", (event) => { if (event.target === overlay) remove(); });
  document.body.appendChild(overlay);
  target.focus?.();
}

function pasteImageIntoLoadedImage(node) {
  if (!navigator.clipboard?.read) {
    showClipboardPasteFallback(node);
    return;
  }
  navigator.clipboard.read().then(async (items) => {
    for (const item of items || []) {
      const mime = (item?.types || []).find((type) => String(type).startsWith("image/"));
      if (!mime) continue;
      const blob = await item.getType(mime);
      await uploadClipboardImageToLoadedImage(node, blob);
      return;
    }
    throw new Error("No image was found on the clipboard.");
  }).catch((error) => {
    if (["NotAllowedError", "SecurityError", "NotReadableError"].includes(error?.name)) {
      console.debug("[Kiri Image Loader] Clipboard read permission unavailable; showing paste target.", error);
      showClipboardPasteFallback(node);
      return;
    }
    console.warn("[Kiri Image Loader] Could not paste the clipboard image:", error);
    window.alert(`Kiri Image Loader: Could not paste the image. ${error?.message || String(error)}`);
  });
}

async function refreshManagedImageWidget(targetNode, preferredNext = null) {
  if (!targetNode) return;
  const widget = getImageFilenameWidget(targetNode) || getImageWidget(targetNode);
  if (!widget) return;
  try {
    const listResponse = await fetch("/kiri-image-loader/files", { cache: "no-store" });
    const listPayload = await listResponse.json();
    if (!listResponse.ok || !listPayload?.ok) return;
    const files = Array.isArray(listPayload.files) ? listPayload.files : [];
    if (widget.options && Array.isArray(widget.options.values)) widget.options.values = files;
    const current = typeof widget.value === "string" ? widget.value : "";
    const next = preferredNext && files.includes(preferredNext)
      ? preferredNext
      : (files.includes(current) ? current : (files[0] || ""));
    if (next !== current) {
      widget.value = next;
      widget.callback?.(next);
      targetNode.setDirtyCanvas?.(true, true);
    }
  } catch (error) {
    console.warn("[Kiri Image Loader] Could not refresh an upstream image selector:", error);
  }
}

function isOpenableEmbeddedWorkflow(workflow) {
  return !!workflow && typeof workflow === "object" &&
    Array.isArray(workflow.nodes) && Array.isArray(workflow.links) &&
    (workflow.nodes.length === 0 || workflow.nodes.some((item) =>
      item && typeof item === "object" && item.id !== undefined &&
      (typeof item.type === "string" || typeof item.class_type === "string")
    ));
}

function embeddedWorkflowTabName(data) {
  const raw = String(data?.filename || "Embedded Image");
  const basename = raw.split(/[\\/]/).pop() || "Embedded Image";
  const withoutExtension = basename.replace(/\.[^.]+$/, "") || "Embedded Image";
  const safe = withoutExtension.replace(/[<>:"/\\|?*\x00-\x1f]/g, "_").trim().slice(0, 100);
  return `${safe || "Embedded Image"} - Embedded Workflow.json`;
}

async function openEmbeddedWorkflow(node, sourceKey) {
  const data = getMetadataState(node, sourceKey);
  const workflowData = data?.embedded_workflow;
  if (!isOpenableEmbeddedWorkflow(workflowData)) {
    window.alert("Kiri Image Loader: This image does not contain a supported embedded ComfyUI workflow.");
    return false;
  }

  // Use ComfyUI's workflow-tab store. Creating a new temporary workflow keeps
  // the current canvas and its unsaved edits intact while opening the recovered
  // graph in a separate tab.
  const workflowStore = app.extensionManager?.workflow;
  if (typeof workflowStore?.createNewTemporary !== "function" ||
      typeof workflowStore?.openWorkflow !== "function") {
    window.alert(
      "Kiri Image Loader: This ComfyUI frontend does not expose the workflow-tab API required to open an embedded workflow without replacing the current canvas. Your current workflow was left unchanged."
    );
    return false;
  }

  try {
    // Detach the data from the metadata cache before handing it to ComfyUI's
    // workflow store, which adds its own workflow identity and tracks edits.
    const safeWorkflow = JSON.parse(JSON.stringify(workflowData));
    const tabName = embeddedWorkflowTabName(data);
    const workflow = workflowStore.createNewTemporary(tabName, safeWorkflow);
    if (!workflow) throw new Error("ComfyUI could not create a new workflow tab.");
    await workflowStore.openWorkflow(workflow);
    console.info(`[Kiri Image Loader] Opened embedded workflow from ${data?.filename || sourceKey}.`);
    return true;
  } catch (error) {
    console.error("[Kiri Image Loader] Could not open embedded workflow:", error);
    window.alert(`Kiri Image Loader: Could not open embedded workflow. ${error?.message || String(error)}`);
    return false;
  }
}

async function performFileAction(node, sourceKey, action) {
  const descriptor = sourceKey === "loaded"
    ? loadedImageDescriptor(node)
    : effectiveImageDescriptor(node, sourceKey);
  if (!descriptor?.filename) {
    window.alert(`Kiri Image Loader: ${descriptor?.message || "This image has no accessible source file."}`);
    return;
  }

  try {
    const response = await fetch("/kiri-image-loader/file-action", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: descriptor.filename,
        type: descriptor.type || "input",
        subfolder: descriptor.subfolder || "",
        path_token: descriptor.pathToken || "",
        action,
      }),
      cache: "no-store",
    });
    const payload = await response.json();
    if (!response.ok || !payload?.ok) throw new Error(payload?.error || `HTTP ${response.status}`);

    if (action === "delete") {
      // If the source was a standard Load Image node, move its selector to the
      // next available image so the graph no longer points at a removed file.
      if (descriptor.type === "input" && descriptor.originNode && descriptor.originNode !== node) {
        await refreshManagedImageWidget(descriptor.originNode);
      }
      if (sourceKey === "loaded" || (descriptor.type === "input" && descriptor.filename === currentFilename(node))) {
        await refreshManagedImageWidget(node);
      }
      syncAllMetadataSources(node);
      // Saved output records (and other references to the same file) can stay
      // in the frontend after deletion. Mark every matching source as deleted
      // so the menu does not repeat file actions against a stale path.
      const sourceDescriptors = {
        loaded: loadedImageDescriptor(node),
        image_a: connectedImageDescriptor(node, "image_a"),
        image_b: connectedImageDescriptor(node, "image_b"),
      };
      const deletedSignature = descriptorSignature(descriptor);
      for (const [key, currentDescriptor] of Object.entries(sourceDescriptors)) {
        if (descriptorSignature(currentDescriptor) !== deletedSignature) continue;
        const refreshedState = getMetadataState(node, key) || {};
        setMetadataState(node, key, {
          ...refreshedState,
          deleted: true,
          unavailable: true,
          loading: false,
          message: "This source file has been moved to the Recycle Bin / Trash.",
          copy_items: [],
          sourceSignature: deletedSignature,
        });
      }
      node.setDirtyCanvas?.(true, true);
    }
    console.info(`[Kiri Image Loader] ${payload.message || "File action completed."}`);
  } catch (error) {
    window.alert(`Kiri Image Loader: ${error?.message || String(error)}`);
  }
}

function metadataEssentials(data) {
  const items = Array.isArray(data?.copy_items) ? data.copy_items : [];
  const byLabel = (pattern) => items.find((item) => pattern.test(String(item?.label || "")));
  const positive = byLabel(/^(positive prompt|prompt)$/i);
  const negative = byLabel(/^(negative prompt|negative)$/i);
  const seedItems = items.filter((item) => /(^|\b)(seed|noise seed)(\b|$)/i.test(String(item?.label || "")));
  const samplerSettings = items.filter((item) => {
    const label = String(item?.label || "").trim();
    return /sampler\s*\d+\s*[·:]\s*(steps|cfg|cfg_scale|sampler|sampler_name|scheduler|denoise|add_noise|start_at_step|end_at_step|force_full_denoise|return_with_leftover_noise|variation_strength)/i.test(label) ||
      /^(steps|cfg|cfg scale|cfg_scale|sampler|sampler name|sampler_name|scheduler|denoise|add noise|add_noise|start at step|start_at_step|end at step|end_at_step|force full denoise|force_full_denoise|return with leftover noise|return_with_leftover_noise|variation strength|variation_strength)(\s*\(ksampler[^)]*\))?$/i.test(label);
  });
  const runtime = data?.runtime_metadata || data?.graph ? runtimeMetadataCopyItems(data.runtime_metadata || data) : [];
  const runtimeByLabel = (pattern) => runtime.find((item) => pattern.test(String(item?.label || "")));
  const positiveItem = positive || runtimeByLabel(/^positive prompt$/i);
  const negativeItem = negative || runtimeByLabel(/^negative prompt$/i);
  const seeds = seedItems.length ? seedItems : runtime.filter((item) => /sampler\s*\d+\s*·\s*(noise_)?seed/i.test(String(item?.label || "")));
  const settings = samplerSettings.length ? samplerSettings : runtime.filter((item) => /sampler\s*\d+\s*·\s*(steps|cfg|cfg_scale|sampler|sampler_name|scheduler|denoise|add_noise|start_at_step|end_at_step|force_full_denoise|return_with_leftover_noise|variation_strength)/i.test(String(item?.label || "")));
  return { positiveItem, negativeItem, seeds, settings };
}

function metadataCopyItems(data) {
  if (!data) return [{ content: "Reading image metadata…", disabled: true }];
  if (data.loading) return [{ content: "Reading image metadata…", disabled: true }];
  if (data.deleted) return [{ content: data.message || "Source image was moved to Trash", disabled: true }];
  if (data.unavailable) return [{ content: data.message || "Original file metadata is unavailable", disabled: true }];
  if (data.error) return [{ content: `Metadata unavailable: ${data.error}`, disabled: true }];

  const essentials = metadataEssentials(data);
  const result = [{ content: "PROMPTS", disabled: true }];
  result.push(essentials.positiveItem
    ? { content: "Copy Positive Prompt", callback: () => copyText(essentials.positiveItem.value) }
    : { content: "Copy Positive Prompt", disabled: true });
  result.push(essentials.negativeItem
    ? { content: "Copy Negative Prompt", callback: () => copyText(essentials.negativeItem.value) }
    : { content: "Copy Negative Prompt", disabled: true });
  result.push(null, { content: "GENERATION", disabled: true });
  if (essentials.seeds.length) {
    const seedText = essentials.seeds.map((item) => `${item.label}: ${item.value}`).join("\n");
    result.push({ content: "Copy Seed(s)", callback: () => copyText(seedText) });
  }
  if (essentials.settings.length) {
    const samplerText = essentials.settings.map((item) => `${item.label}: ${item.value}`).join("\n");
    result.push({ content: "Copy KSampler Settings", callback: () => copyText(samplerText) });
  }
  if (!essentials.seeds.length && !essentials.settings.length) result.push({ content: "No seed or KSampler settings found", disabled: true });
  const total = Array.isArray(data.copy_items) ? data.copy_items.length : runtimeMetadataCopyItems(data.runtime_metadata || data).length;
  if (total) result.push(null, { content: "View All Metadata…", callback: () => showFullMetadataDialog(data) });
  else result.push(null, { content: "No metadata detected", disabled: true });
  return result;
}

function showFullMetadataDialog(data) {
  if (typeof document === "undefined" || !document.body) return;
  const overlay = document.createElement("div");
  Object.assign(overlay.style, {
    position: "fixed", inset: "0", zIndex: "100000", display: "flex",
    alignItems: "center", justifyContent: "center", padding: "20px",
    background: "rgba(0,0,0,.68)",
  });
  const dialog = document.createElement("div");
  Object.assign(dialog.style, {
    width: "min(900px, 94vw)", maxHeight: "86vh", display: "flex",
    flexDirection: "column", gap: "10px", padding: "16px", boxSizing: "border-box",
    color: "var(--input-text, #eee)", background: "var(--comfy-menu-bg, #222)",
    border: "1px solid var(--border-color, #555)", borderRadius: "9px",
    boxShadow: "0 14px 48px rgba(0,0,0,.55)",
  });
  const header = document.createElement("div");
  Object.assign(header.style, { display: "flex", justifyContent: "space-between", alignItems: "center", gap: "12px" });
  const title = document.createElement("strong");
  title.textContent = "Image Metadata · Read Only";
  const close = document.createElement("button");
  close.type = "button";
  close.textContent = "Close";
  const remove = () => overlay.remove();
  close.addEventListener("click", remove);
  header.append(title, close);
  const body = document.createElement("div");
  Object.assign(body.style, { overflow: "auto", minHeight: "80px", minWidth: "0", width: "100%", boxSizing: "border-box", paddingRight: "4px" });
  const sourceItems = Array.isArray(data.copy_items) ? data.copy_items : runtimeMetadataCopyItems(data.runtime_metadata || data);
  const groups = new Map();
  for (const item of sourceItems) {
    const category = item.category || "Other Metadata";
    if (!groups.has(category)) groups.set(category, []);
    groups.get(category).push(item);
  }
  if (groups.size === 0) {
    const empty = document.createElement("div");
    empty.textContent = data.message || "No metadata found.";
    body.appendChild(empty);
  }
  for (const [category, fields] of groups) {
    const section = document.createElement("section");
    section.style.marginBottom = "14px";
    const heading = document.createElement("div");
    heading.textContent = category;
    Object.assign(heading.style, { fontWeight: "700", margin: "4px 0 7px", opacity: ".9" });
    section.appendChild(heading);
    for (const item of fields) {
      const row = document.createElement("div");
      Object.assign(row.style, {
        display: "grid",
        // Keep both tracks shrinkable. Metadata keys often contain long
        // dotted/underscored platform paths that otherwise overflow the
        // label cell and visually collide with the value column.
        gridTemplateColumns: "minmax(0, 34%) minmax(0, 1fr) auto",
        gap: "12px", alignItems: "start", width: "100%", minWidth: "0",
        boxSizing: "border-box", padding: "5px 8px",
        borderTop: "1px solid var(--border-color, #444)",
      });
      const label = document.createElement("div");
      label.textContent = String(item.label || "Metadata");
      Object.assign(label.style, {
        fontWeight: "600", minWidth: "0", maxWidth: "100%",
        whiteSpace: "normal", overflowWrap: "anywhere", wordBreak: "break-word",
        boxSizing: "border-box", lineHeight: "1.4",
      });
      const value = document.createElement("pre");
      value.textContent = String(item.value ?? "");
      Object.assign(value.style, {
        margin: "0", minWidth: "0", maxWidth: "100%", boxSizing: "border-box",
        whiteSpace: "pre-wrap", overflowWrap: "anywhere", wordBreak: "break-word",
        font: "12px/1.45 monospace", opacity: ".9",
      });
      row.append(label, value);
      // Platform handoff objects are intended to be pasted into their app/API
      // parameter forms. Keep the viewer readable while adding a direct copy
      // action for structured payloads instead of forcing manual text selection.
      if (/json|handoff|interop/i.test(String(item.label || ""))) {
        const copyButton = document.createElement("button");
        copyButton.type = "button";
        copyButton.textContent = "Copy JSON";
        copyButton.setAttribute("aria-label", `Copy ${String(item.label || "metadata")} JSON`);
        Object.assign(copyButton.style, {
          alignSelf: "start", whiteSpace: "nowrap", padding: "4px 7px",
          borderRadius: "5px", border: "1px solid var(--border-color, #555)",
          color: "inherit", background: "transparent", font: "11px sans-serif",
          cursor: "pointer",
        });
        copyButton.addEventListener("click", () => { void copyText(item.value); });
        row.appendChild(copyButton);
      }
      section.appendChild(row);
    }
    body.appendChild(section);
  }
  dialog.append(header, body);
  overlay.appendChild(dialog);
  overlay.addEventListener("pointerdown", (event) => { if (event.target === overlay) remove(); });
  overlay.addEventListener("keydown", (event) => { if (event.key === "Escape") remove(); });
  document.body.appendChild(overlay);
  close.focus?.();
}

function parseEditorJsonEntry(entries, keyword) {
  const entry = entries.find((item) => String(item?.keyword || "").toLowerCase() === keyword);
  if (!entry || typeof entry.value !== "string") return null;
  try {
    const value = JSON.parse(entry.value);
    return value && typeof value === "object" ? value : null;
  } catch {
    return null;
  }
}

function editorParameterLabel(key) {
  const aliases = {
    seed: "Seed", noise_seed: "Seed", steps: "Steps", cfg: "CFG", cfg_scale: "CFG",
    sampler_name: "Sampler", sampler: "Sampler", scheduler: "Scheduler",
    denoise: "Denoise", width: "Width", height: "Height",
  };
  const normalized = String(key || "").trim().toLowerCase();
  return aliases[normalized] || String(key || "Parameter").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function buildMetadataEditorModel(data) {
  const entries = Array.isArray(data?.editable_metadata) ? data.editable_metadata : [];
  const kiri = parseEditorJsonEntry(entries, "kiri_metadata") || {};
  const graph = kiri.graph && typeof kiri.graph === "object" ? kiri.graph : {};
  const graphPrompts = graph.prompts && typeof graph.prompts === "object" ? graph.prompts : {};
  const copyItems = Array.isArray(data?.copy_items) ? data.copy_items : [];
  const copyValue = (labels) => {
    const wanted = new Set(labels.map((label) => label.toLowerCase()));
    const found = copyItems.find((item) => wanted.has(String(item?.label || "").toLowerCase()));
    return found?.value == null ? "" : String(found.value);
  };

  const positive = String(graphPrompts.positive ?? copyValue(["Positive Prompt", "Prompt"]));
  const negative = String(graphPrompts.negative ?? copyValue(["Negative Prompt", "Negative"]));
  const parameters = [];
  const samplers = Array.isArray(graph.samplers) ? graph.samplers : [];
  samplers.forEach((sampler, samplerIndex) => {
    const settings = sampler?.settings && typeof sampler.settings === "object" ? sampler.settings : {};
    for (const [key, value] of Object.entries(settings)) {
      const label = editorParameterLabel(key);
      parameters.push({
        kind: "sampler", sampler_index: samplerIndex, key,
        label: samplers.length > 1 ? `Sampler ${samplerIndex + 1} · ${label}` : label,
        value: value == null ? "" : String(value),
      });
    }
  });

  // For PNGs without Kiri graph metadata, reuse the reader's already
  // categorized interpretation instead of showing raw JSON chunks as fields.
  if (!parameters.length) {
    const skip = new Set(["filename", "format", "color mode", "width", "height"]);
    for (const item of copyItems) {
      const category = String(item?.category || "").toLowerCase();
      const label = String(item?.label || "");
      if (!(["generation", "image"].includes(category)) || skip.has(label.toLowerCase())) continue;
      parameters.push({ kind: "generic", key: label, label, value: String(item?.value ?? "") });
    }
  }

  const dimensions = kiri.image_dimensions && typeof kiri.image_dimensions === "object" ? kiri.image_dimensions : {};
  const width = dimensions.width ?? data?.width ?? "";
  const height = dimensions.height ?? data?.height ?? "";
  parameters.push({ kind: "dimension", key: "width", label: "Width", value: width == null ? "" : String(width) });
  parameters.push({ kind: "dimension", key: "height", label: "Height", value: height == null ? "" : String(height) });

  const coreKeys = new Set(["prompt", "kiri_metadata", "parameters", "kiri nodes used", "kiri metadata mode"]);
  const additional = entries.filter((entry) => !coreKeys.has(String(entry?.keyword || "").toLowerCase()));
  return { positive, negative, parameters, additional, originalEntries: entries, hasKiriMetadata: !!Object.keys(kiri).length };
}

function showMetadataEditor(node, sourceKey, descriptor, data) {
  if (typeof document === "undefined" || !document.body) return;
  if (!data?.metadata_editable || !Array.isArray(data.editable_metadata)) {
    window.alert("Kiri Image Loader: Metadata editing is currently available for PNG images only. Refresh the metadata menu and try again.");
    return;
  }

  const model = buildMetadataEditorModel(data);
  const overlay = document.createElement("div");
  Object.assign(overlay.style, {
    position: "fixed", inset: "0", zIndex: "100001", display: "flex",
    alignItems: "center", justifyContent: "center", padding: "20px",
    background: "rgba(0,0,0,.72)",
  });
  const dialog = document.createElement("div");
  Object.assign(dialog.style, {
    width: "min(980px, 96vw)", maxHeight: "90vh", display: "flex",
    flexDirection: "column", gap: "10px", padding: "16px", boxSizing: "border-box",
    color: "var(--input-text, #eee)", background: "var(--comfy-menu-bg, #222)",
    border: "1px solid var(--border-color, #555)", borderRadius: "9px",
    boxShadow: "0 14px 48px rgba(0,0,0,.55)",
  });
  const header = document.createElement("div");
  Object.assign(header.style, { display: "flex", justifyContent: "space-between", alignItems: "center", gap: "12px" });
  const title = document.createElement("strong");
  title.textContent = `Edit PNG Metadata · ${descriptor.filename}`;
  const close = document.createElement("button");
  close.type = "button";
  close.textContent = "Close";
  const remove = () => overlay.remove();
  close.addEventListener("click", remove);
  header.append(title, close);

  const note = document.createElement("div");
  const isWorkflowOffImage = model.originalEntries.some((entry) =>
    String(entry?.keyword || "").toLowerCase() === "kiri metadata mode" &&
    /workflow.{0,20}\boff\b/i.test(String(entry?.value || ""))
  );
  note.textContent = isWorkflowOffImage
    ? "This image was saved with Embed Complete Workflow OFF. It has no ComfyUI-loadable node graph. Saving edits will not add JSON workflow data, node IDs, graph inputs, links, or ComfyUI workflow records. Parameter-only metadata for other apps may still be present when enabled; it contains settings and resource references, not ComfyUI nodes. Width and Height edit recorded metadata only; they do not resize the image."
    : "Edit the positive prompt, negative prompt, and generation parameters. Kiri's internal metadata and ComfyUI graph JSON are kept behind the scenes. Width and Height edit the recorded metadata only; they do not resize the image.";
  Object.assign(note.style, { fontSize: "12px", lineHeight: "1.5", opacity: ".82" });

  const body = document.createElement("div");
  Object.assign(body.style, { overflow: "auto", minHeight: "100px", paddingRight: "4px" });
  const structuredFields = [];
  const additionalRows = [];
  const controlStyle = {
    boxSizing: "border-box", width: "100%", color: "var(--input-text, #eee)",
    background: "var(--comfy-input-bg, #333)", border: "1px solid var(--border-color, #555)",
    borderRadius: "5px", padding: "7px", font: "12px/1.45 monospace",
  };

  function addSection(name) {
    const section = document.createElement("section");
    section._kiriSection = name;
    Object.assign(section.style, { marginBottom: "14px", padding: "8px 10px", border: "1px solid var(--border-color, #444)", borderRadius: "6px" });
    const heading = document.createElement("div");
    heading.textContent = name;
    Object.assign(heading.style, { fontWeight: "700", margin: "0 0 8px", opacity: ".96" });
    section.appendChild(heading);
    body.appendChild(section);
    return section;
  }

  function addPromptField(section, label, value, kind) {
    const wrapper = document.createElement("div");
    Object.assign(wrapper.style, { display: "flex", flexDirection: "column", gap: "5px", marginBottom: "7px" });
    const caption = document.createElement("label");
    caption.textContent = label;
    caption.style.fontWeight = "600";
    const editor = document.createElement("textarea");
    editor.value = String(value ?? "");
    editor.rows = Math.max(3, Math.min(10, editor.value.split("\n").length + 1));
    editor.placeholder = label;
    Object.assign(editor.style, controlStyle, { resize: "vertical", minHeight: "76px" });
    wrapper.append(caption, editor);
    section.appendChild(wrapper);
    structuredFields.push({ kind, editor });
  }

  function addParameterField(section, field) {
    const row = document.createElement("div");
    Object.assign(row.style, { display: "grid", gridTemplateColumns: "minmax(160px, 28%) minmax(0, 1fr)", gap: "10px", alignItems: "center", padding: "5px 0", borderTop: "1px solid var(--border-color, #444)" });
    const label = document.createElement("label");
    label.textContent = field.label;
    label.style.fontWeight = "600";
    const value = document.createElement("input");
    value.type = "text";
    value.value = String(field.value ?? "");
    Object.assign(value.style, controlStyle);
    row.append(label, value);
    section.appendChild(row);
    structuredFields.push({ ...field, editor: value });
  }

  const promptSection = addSection("Prompts");
  addPromptField(promptSection, "Positive Prompt", model.positive, "positive_prompt");
  addPromptField(promptSection, "Negative Prompt", model.negative, "negative_prompt");
  const parameterSection = addSection("Parameters");
  for (const field of model.parameters) addParameterField(parameterSection, field);

  let additionalSection = model.additional.length ? addSection("Additional Metadata") : null;
  function addAdditionalField(entry = {}) {
    const row = document.createElement("div");
    Object.assign(row.style, { display: "grid", gridTemplateColumns: "minmax(150px, 24%) minmax(0, 1fr) auto", gap: "8px", alignItems: "start", padding: "8px 0", borderTop: "1px solid var(--border-color, #444)" });
    const key = document.createElement("input");
    key.type = "text";
    key.value = String(entry.keyword ?? "");
    key.placeholder = "Metadata key";
    Object.assign(key.style, controlStyle);
    const value = document.createElement("textarea");
    value.value = String(entry.value ?? "");
    value.rows = Math.max(2, Math.min(8, value.value.split("\n").length + 1));
    value.placeholder = "Metadata value";
    Object.assign(value.style, controlStyle, { resize: "vertical", minHeight: "52px" });
    const del = document.createElement("button");
    del.type = "button";
    del.textContent = "Remove";
    del.addEventListener("click", () => { row.remove(); additionalRows.splice(additionalRows.indexOf(row), 1); });
    row.append(key, value, del);
    (additionalSection || body).appendChild(row);
    additionalRows.push(row);
    row._kiriRead = () => ({ keyword: key.value, value: value.value });
  }
  for (const entry of model.additional) addAdditionalField(entry);

  const footer = document.createElement("div");
  Object.assign(footer.style, { display: "flex", justifyContent: "space-between", alignItems: "center", gap: "10px", flexWrap: "wrap" });
  const add = document.createElement("button");
  add.type = "button";
  add.textContent = "Add Metadata Field";
  add.addEventListener("click", () => {
    if (!additionalSection) {
      additionalSection = addSection("Additional Metadata");
    }
    addAdditionalField();
    body.scrollTop = body.scrollHeight;
  });
  const actions = document.createElement("div");
  Object.assign(actions.style, { display: "flex", gap: "8px" });
  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.textContent = "Cancel";
  cancel.addEventListener("click", remove);
  const save = document.createElement("button");
  save.type = "button";
  save.textContent = "Save Metadata";
  save.addEventListener("click", async () => {
    const editorValues = {
      positive_prompt: structuredFields.find((field) => field.kind === "positive_prompt")?.editor.value ?? "",
      negative_prompt: structuredFields.find((field) => field.kind === "negative_prompt")?.editor.value ?? "",
      parameters: structuredFields.filter((field) => field.kind === "sampler" || field.kind === "generic").map((field) => ({
        kind: field.kind, sampler_index: field.sampler_index ?? 0, key: field.key, label: field.label, value: field.editor.value,
      })),
      image_width: structuredFields.find((field) => field.kind === "dimension" && field.key === "width")?.editor.value ?? "",
      image_height: structuredFields.find((field) => field.kind === "dimension" && field.key === "height")?.editor.value ?? "",
    };
    const additional = additionalRows.map((row) => row._kiriRead()).filter((entry) => entry.keyword.trim());
    const emptyKey = additionalRows.some((row) => !row._kiriRead().keyword.trim() && row._kiriRead().value.trim());
    if (emptyKey) {
      window.alert("Every non-empty metadata value needs a key.");
      return;
    }
    // Preserve all original text chunks as the backend's source document;
    // semantic sections are applied to the structured Kiri/A1111 chunks there.
    const metadataEntries = model.originalEntries.filter((entry) => [
      "prompt", "kiri_metadata", "parameters", "kiri nodes used", "kiri metadata mode",
    ].includes(String(entry?.keyword || "").toLowerCase()));
    metadataEntries.push(...additional);
    save.disabled = true;
    save.textContent = "Saving…";
    try {
      const response = await fetch("/kiri-image-loader/file-action", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filename: descriptor.filename,
          type: descriptor.type || "input",
          subfolder: descriptor.subfolder || "",
          path_token: descriptor.pathToken || "",
          action: "edit_metadata",
          metadata_entries: metadataEntries,
          editor_values: editorValues,
        }),
        cache: "no-store",
      });
      const payload = await response.json();
      if (!response.ok || !payload?.ok) throw new Error(payload?.error || `HTTP ${response.status}`);
      await readMetadataForSource(node, sourceKey, descriptor);
      node.setDirtyCanvas?.(true, true);
      remove();
      window.alert(`Kiri Image Loader: ${payload.message || "PNG metadata updated."}`);
    } catch (error) {
      save.disabled = false;
      save.textContent = "Save Metadata";
      window.alert(`Kiri Image Loader: ${error?.message || String(error)}`);
    }
  });
  actions.append(cancel, save);
  footer.append(add, actions);
  dialog.append(header, note, body, footer);
  overlay.appendChild(dialog);
  overlay.addEventListener("pointerdown", (event) => { if (event.target === overlay) remove(); });
  overlay.addEventListener("keydown", (event) => { if (event.key === "Escape") remove(); });
  document.body.appendChild(overlay);
  close.focus?.();
}

function generationDataForSource(data) {
  if (data?.generation_data && Array.isArray(data.generation_data.nodes)) return data.generation_data;
  const record = data?.runtime_metadata || data;
  const graph = record?.graph && typeof record.graph === "object" ? record.graph : {};
  if (Array.isArray(graph.active_upstream_nodes)) {
    return {
      schema_version: 1,
      source: "Kiri structured graph",
      workflow_id: record?.source_workflow_id || null,
      nodes: graph.active_upstream_nodes,
    };
  }
  return null;
}

function nodeClassType(node) {
  return String(node?.comfyClass || node?.constructor?.comfyClass || node?.type || "").trim();
}

function nodeDisplayTitle(node) {
  try {
    const title = typeof node?.getTitle === "function" ? node.getTitle() : null;
    return String(node?.title || node?.properties?.title || title || "").trim();
  } catch {
    return String(node?.title || node?.properties?.title || "").trim();
  }
}

function nodeIdText(node) {
  return String(node?.id ?? "?");
}

function sourceNodeDisplayTitle(source) {
  const title = String(source?.title || "").trim();
  const type = String(source?.class_type || "Unknown");
  const id = String(source?.id ?? "?");
  return `${title ? `${title} · ` : ""}${type} · Source ID ${id}`;
}

function targetNodeDisplayTitle(target) {
  const type = nodeClassType(target) || "Unknown Node";
  const title = nodeDisplayTitle(target);
  const graphId = target?.graph?.id;
  const location = graphId ? ` · Graph ${String(graphId).slice(0, 8)}` : "";
  return `${title ? `${title} · ` : ""}${type} · Node ID ${nodeIdText(target)}${location}`;
}

function currentGraphNodes(referenceNode) {
  const graph = referenceNode?.graph || app.graph || app.canvas?.graph;
  const rows = Array.isArray(graph?._nodes) ? graph._nodes : (Array.isArray(graph?.nodes) ? graph.nodes : []);
  return [...new Set(rows.filter(Boolean))];
}

function isImportableScalar(value) {
  if (typeof value === "string" || typeof value === "boolean") return true;
  return typeof value === "number" && Number.isFinite(value);
}

function targetWidgetForInput(target, inputName) {
  const destinationInput = (target?.inputs || []).find((input) => input?.name === inputName);
  // A connected socket owns its value. Importing a serialized widget value
  // must never disconnect or override an existing connection.
  if (destinationInput && isSocketConnected(destinationInput)) return null;
  const widgetName = destinationInput?.widget?.name || inputName;
  const widget = (target?.widgets || []).find((candidate) => candidate?.name === widgetName) ||
    (target?.widgets || []).find((candidate) => candidate?.name === inputName);
  if (!widget || typeof widget !== "object") return null;
  const widgetType = String(widget.type || "").toLowerCase();
  if (widgetType === "button" || widget.options?.serialize === false || widget.serialize === false) return null;
  return widget;
}

function compatibleWidgetFields(sourceNode, targetNode) {
  const inputs = sourceNode?.inputs && typeof sourceNode.inputs === "object" ? sourceNode.inputs : {};
  const fields = [];
  for (const [name, value] of Object.entries(inputs)) {
    // Prompt links, model/conditioning objects and nested structured values are
    // deliberately excluded. The importer only transfers serialized scalar
    // widget values into an existing widget with the same input name.
    if (!isImportableScalar(value)) continue;
    const widget = targetWidgetForInput(targetNode, name);
    if (!widget) continue;
    fields.push({
      name,
      sourceValue: value,
      targetValue: widget.value,
      widgetName: widget.name || name,
    });
  }
  return fields;
}

function sourceNodeCandidates(data, referenceNode) {
  const generationData = generationDataForSource(data);
  if (!generationData || !Array.isArray(generationData.nodes)) return [];
  const targets = currentGraphNodes(referenceNode);
  const result = [];
  for (const source of generationData.nodes) {
    if (!source || !source.class_type || !source.inputs || typeof source.inputs !== "object") continue;
    const matches = targets.filter((target) => nodeClassType(target) === String(source.class_type));
    const compatibleTargets = matches.filter((target) => compatibleWidgetFields(source, target).length > 0);
    if (compatibleTargets.length) result.push({ source, targets: compatibleTargets });
  }
  return result;
}

async function applyNodeWidgetValues(targetNode, fields) {
  const applied = [];
  const skipped = [];
  for (const field of fields || []) {
    const widget = targetWidgetForInput(targetNode, field.name);
    if (!widget) {
      skipped.push(field.name);
      continue;
    }
    const value = field.sourceValue;
    try {
      // Native combo widgets need the imported value present in their option
      // list so it remains visible even when the current list is still loading.
      if (String(widget.type || "").toLowerCase() === "combo") {
        widget.options = widget.options || {};
        const values = Array.isArray(widget.options.values) ? [...widget.options.values] : [];
        if (!values.includes(value)) values.push(value);
        widget.options.values = values;
      }
      if (typeof widget.setValue === "function") {
        await widget.setValue(value, { node: targetNode, canvas: app.canvas });
      } else {
        widget.value = value;
        if (typeof widget.callback === "function") {
          await widget.callback(value, app.canvas, targetNode, [0, 0]);
        } else if (typeof targetNode.onWidgetChanged === "function") {
          targetNode.onWidgetChanged(widget.name || field.name, value, widget);
        }
      }
      applied.push(field.name);
    } catch (error) {
      skipped.push(field.name);
      console.warn(`[Kiri Image Loader] Could not apply ${field.name} to ${nodeClassType(targetNode)}:`, error);
    }
  }

  // Kiri LoRA Loader keeps its dynamic visible rows synchronized from the
  // canonical hidden loras_json widget. Its own onConfigure hook rebuilds those
  // rows and restores the trigger-word selector without replacing connections.
  if (/keryulapowerloraloader/i.test(nodeClassType(targetNode)) && applied.length) {
    try {
      await targetNode._kLorasPromise;
      if (typeof targetNode.onConfigure === "function") targetNode.onConfigure({});
    } catch (error) {
      console.warn("[Kiri Image Loader] Kiri LoRA rows could not be refreshed after import:", error);
    }
  }

  try { targetNode.graph?.change?.(); } catch { /* optional graph API */ }
  try { app.graph?.change?.(); } catch { /* optional graph API */ }
  targetNode.setDirtyCanvas?.(true, true);
  app.canvas?.setDirty?.(true, true);
  return { applied, skipped };
}

function showApplyGenerationDataDialog(node, sourceKey, sourceLabel, metadata) {
  if (typeof document === "undefined" || !document.body) return;
  const candidates = sourceNodeCandidates(metadata, node);
  if (!candidates.length) {
    window.alert("Kiri Image Loader: This image does not contain node settings that match compatible widgets in the current workflow.");
    return;
  }

  const overlay = document.createElement("div");
  Object.assign(overlay.style, {
    position: "fixed", inset: "0", zIndex: "100002", display: "flex",
    alignItems: "center", justifyContent: "center", padding: "20px",
    background: "rgba(0,0,0,.72)",
  });
  const dialog = document.createElement("div");
  Object.assign(dialog.style, {
    width: "min(820px, 96vw)", maxHeight: "90vh", display: "flex", flexDirection: "column",
    gap: "10px", padding: "16px", boxSizing: "border-box",
    color: "var(--input-text, #eee)", background: "var(--comfy-menu-bg, #222)",
    border: "1px solid var(--border-color, #555)", borderRadius: "9px",
    boxShadow: "0 14px 48px rgba(0,0,0,.55)", font: "13px/1.4 sans-serif",
  });
  const header = document.createElement("div");
  Object.assign(header.style, { display: "flex", justifyContent: "space-between", alignItems: "center", gap: "12px" });
  const title = document.createElement("strong");
  title.textContent = `Apply Generation Data · ${sourceLabel}`;
  const close = document.createElement("button");
  close.type = "button";
  close.textContent = "Close";
  const remove = () => overlay.remove();
  close.addEventListener("click", remove);
  header.append(title, close);

  const note = document.createElement("div");
  note.textContent = "Choose the original node whose settings you want, then explicitly choose which matching node in your current workflow should receive them. Matching uses the exact node type and compatible widget names; node IDs are shown for reference only and are never used to select a destination automatically.";
  Object.assign(note.style, { fontSize: "12px", opacity: ".82", lineHeight: "1.5" });

  const form = document.createElement("div");
  Object.assign(form.style, { display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" });
  const makeSelectField = (labelText) => {
    const wrapper = document.createElement("label");
    Object.assign(wrapper.style, { display: "flex", flexDirection: "column", gap: "5px", minWidth: "0", fontWeight: "600" });
    const caption = document.createElement("span");
    caption.textContent = labelText;
    const select = document.createElement("select");
    Object.assign(select.style, {
      width: "100%", minWidth: "0", boxSizing: "border-box", padding: "8px",
      borderRadius: "5px", border: "1px solid var(--border-color, #555)",
      background: "var(--comfy-input-bg, #333)", color: "var(--input-text, #eee)", font: "12px sans-serif",
    });
    wrapper.append(caption, select);
    return { wrapper, select };
  };
  const sourceField = makeSelectField("Source node in the image's generation graph");
  const targetField = makeSelectField("Destination node in your current workflow");
  form.append(sourceField.wrapper, targetField.wrapper);

  const fieldsHeading = document.createElement("strong");
  fieldsHeading.textContent = "Compatible settings to transfer";
  const fieldsPanel = document.createElement("div");
  Object.assign(fieldsPanel.style, {
    overflow: "auto", minHeight: "100px", maxHeight: "42vh", padding: "8px",
    border: "1px solid var(--border-color, #444)", borderRadius: "6px",
  });
  const status = document.createElement("div");
  status.textContent = "Select a source node and destination node to preview compatible settings.";
  status.style.opacity = ".75";
  status.style.fontSize = "12px";
  fieldsPanel.appendChild(status);

  const footer = document.createElement("div");
  Object.assign(footer.style, { display: "flex", justifyContent: "space-between", alignItems: "center", gap: "10px" });
  const safety = document.createElement("span");
  safety.textContent = "Existing links and workflow layout are preserved.";
  safety.style.opacity = ".75";
  safety.style.fontSize = "11px";
  const actions = document.createElement("div");
  Object.assign(actions.style, { display: "flex", gap: "8px" });
  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.textContent = "Cancel";
  const apply = document.createElement("button");
  apply.type = "button";
  apply.textContent = "Apply Selected Settings";
  apply.disabled = true;
  actions.append(cancel, apply);
  footer.append(safety, actions);
  cancel.addEventListener("click", remove);

  const makeOption = (label, value) => {
    const option = document.createElement("option");
    option.textContent = label;
    option.value = value;
    return option;
  };
  sourceField.select.appendChild(makeOption("Choose a source node…", ""));
  candidates.forEach((candidate, index) => {
    sourceField.select.appendChild(makeOption(sourceNodeDisplayTitle(candidate.source), String(index)));
  });

  let currentFields = [];
  function renderDestinationChoices() {
    targetField.select.replaceChildren();
    targetField.select.appendChild(makeOption("Choose a destination node…", ""));
    fieldsPanel.replaceChildren();
    currentFields = [];
    apply.disabled = true;
    const sourceIndex = Number(sourceField.select.value);
    if (!sourceField.select.value || !candidates[sourceIndex]) {
      status.textContent = "Select a source node to see matching destinations.";
      fieldsPanel.appendChild(status);
      return;
    }
    const candidate = candidates[sourceIndex];
    candidate.targets.forEach((target, index) => {
      targetField.select.appendChild(makeOption(targetNodeDisplayTitle(target), String(index)));
    });
    status.textContent = `Found ${candidate.targets.length} compatible destination node(s). Choose the exact one to update.`;
    fieldsPanel.appendChild(status);
  }

  function renderCompatibleFields() {
    fieldsPanel.replaceChildren();
    currentFields = [];
    apply.disabled = true;
    const sourceIndex = Number(sourceField.select.value);
    const targetIndex = Number(targetField.select.value);
    if (!sourceField.select.value || !targetField.select.value || !candidates[sourceIndex]) {
      status.textContent = "Choose both nodes to preview settings.";
      fieldsPanel.appendChild(status);
      return;
    }
    const candidate = candidates[sourceIndex];
    const target = candidate.targets[targetIndex];
    if (!target) {
      status.textContent = "Choose a destination node.";
      fieldsPanel.appendChild(status);
      return;
    }
    currentFields = compatibleWidgetFields(candidate.source, target);
    if (!currentFields.length) {
      status.textContent = "No unconnected, same-name widgets can safely receive settings on this destination.";
      fieldsPanel.appendChild(status);
      return;
    }
    status.textContent = `${currentFields.length} compatible setting(s). Uncheck anything you want to leave unchanged.`;
    fieldsPanel.appendChild(status);
    for (const field of currentFields) {
      const row = document.createElement("label");
      Object.assign(row.style, {
        display: "grid", gridTemplateColumns: "20px minmax(110px, 24%) minmax(0, 1fr) minmax(0, 1fr)",
        gap: "8px", alignItems: "start", padding: "7px 2px", borderTop: "1px solid var(--border-color, #444)",
      });
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = true;
      checkbox.dataset.kiriField = field.name;
      const name = document.createElement("span");
      name.textContent = field.name;
      name.style.fontWeight = "600";
      const sourceValue = document.createElement("pre");
      sourceValue.textContent = String(field.sourceValue);
      Object.assign(sourceValue.style, { margin: "0", whiteSpace: "pre-wrap", overflowWrap: "anywhere", font: "11px/1.4 monospace" });
      const targetValue = document.createElement("pre");
      targetValue.textContent = `Current: ${String(field.targetValue ?? "")}`;
      Object.assign(targetValue.style, { margin: "0", whiteSpace: "pre-wrap", overflowWrap: "anywhere", font: "11px/1.4 monospace", opacity: ".68" });
      row.append(checkbox, name, sourceValue, targetValue);
      fieldsPanel.appendChild(row);
    }
    apply.disabled = false;
  }

  sourceField.select.addEventListener("change", () => {
    targetField.select.value = "";
    renderDestinationChoices();
  });
  targetField.select.addEventListener("change", renderCompatibleFields);
  apply.addEventListener("click", async () => {
    const sourceIndex = Number(sourceField.select.value);
    const targetIndex = Number(targetField.select.value);
    const candidate = candidates[sourceIndex];
    const target = candidate?.targets?.[targetIndex];
    if (!candidate || !target) return;
    const selected = [...fieldsPanel.querySelectorAll('input[type="checkbox"][data-kiri-field]')]
      .filter((checkbox) => checkbox.checked)
      .map((checkbox) => currentFields.find((field) => field.name === checkbox.dataset.kiriField))
      .filter(Boolean);
    if (!selected.length) {
      window.alert("Kiri Image Loader: Select at least one setting to apply.");
      return;
    }
    apply.disabled = true;
    apply.textContent = "Applying…";
    try {
      const outcome = await applyNodeWidgetValues(target, selected);
      remove();
      const appliedText = outcome.applied.length ? outcome.applied.join(", ") : "none";
      const skippedText = outcome.skipped.length ? `\nSkipped: ${outcome.skipped.join(", ")}` : "";
      window.alert(`Kiri Image Loader: Applied ${outcome.applied.length} setting(s) to ${targetNodeDisplayTitle(target)}.\nUpdated: ${appliedText}${skippedText}`);
    } catch (error) {
      apply.disabled = false;
      apply.textContent = "Apply Selected Settings";
      window.alert(`Kiri Image Loader: ${error?.message || String(error)}`);
    }
  });

  dialog.append(header, note, form, fieldsHeading, fieldsPanel, footer);
  overlay.appendChild(dialog);
  overlay.addEventListener("pointerdown", (event) => { if (event.target === overlay) remove(); });
  overlay.addEventListener("keydown", (event) => { if (event.key === "Escape") remove(); });
  document.body.appendChild(overlay);
  close.focus?.();
}

function sourceSubmenuItems(node, sourceKey, label) {
  const descriptor = sourceKey === "loaded"
    ? loadedImageDescriptor(node)
    : effectiveImageDescriptor(node, sourceKey);
  const data = getMetadataState(node, sourceKey);
  const sourceDeleted = !!data?.deleted && data.sourceSignature === descriptorSignature(descriptor);
  const canManageFile = !!descriptor?.filename && !sourceDeleted;
  const canDeleteFile = canManageFile && descriptor.type !== "temp";
  const result = [];
  if (isOpenableEmbeddedWorkflow(data?.embedded_workflow)) {
    result.push({
      content: "Open Embedded Workflow",
      callback: () => openEmbeddedWorkflow(node, sourceKey),
    }, null);
  }
  result.push(
    { content: "Open", disabled: !canManageFile, callback: () => performFileAction(node, sourceKey, "open") },
    { content: "Open File Location", disabled: !canManageFile, callback: () => performFileAction(node, sourceKey, "reveal") },
    { content: "Delete Image (Recycle Bin)", disabled: !canDeleteFile, callback: () => performFileAction(node, sourceKey, "delete") },
    null,
    {
      content: "Copy",
      disabled: !canCopyImageSource(node, sourceKey),
      callback: () => copyImageSourceToClipboard(node, sourceKey, label),
    },
  );
  if (sourceKey === "loaded") result.push({ content: "Paste", callback: () => pasteImageIntoLoadedImage(node) });
  result.push(
    null,
    {
      content: "Edit Metadata…",
      disabled: !canManageFile || descriptor.type === "temp" || !/\.png$/i.test(descriptor.filename) || !Array.isArray(data?.editable_metadata),
      callback: () => showMetadataEditor(node, sourceKey, descriptor, data),
    },
    {
      content: "Apply Generation Data to Node…",
      disabled: sourceNodeCandidates(data, node).length === 0,
      callback: () => showApplyGenerationDataDialog(node, sourceKey, label, data),
    },
    { content: "COPY IMPORTANT METADATA", disabled: true },
    ...metadataCopyItems(data),
  );
  return result;
}

function sourceMenuEntry(node, sourceKey, label) {
  const descriptor = sourceKey === "loaded"
    ? loadedImageDescriptor(node)
    : effectiveImageDescriptor(node, sourceKey);
  if (sourceKey !== "loaded" && !descriptor) {
    const inputName = sourceKey === "image_a" ? "image_in" : "mask_in";
    const input = node?.inputs?.find((item) => item?.name === inputName);
    if (!isSocketConnected(input)) return { content: `${label} (not connected)`, disabled: true };
  }
  return {
    content: label,
    has_submenu: true,
    submenu: { options: sourceSubmenuItems(node, sourceKey, label), title: label },
    callback: () => {},
  };
}



function getKiriWidget(node, name) {
  return node?.widgets?.find((widget) => widget?.name === name) || null;
}

function getKiriWidgetValue(node, name, fallback) {
  const widget = getKiriWidget(node, name);
  return widget ? widget.value : fallback;
}

function setKiriWidgetValue(node, name, value) {
  const widget = getKiriWidget(node, name);
  if (!widget) return false;
  widget.value = value;
  widget.callback?.(value, app.canvas, node);
  node.setDirtyCanvas?.(true, true);
  return true;
}

function hideMetadataSettingsWidgets(node) {
  for (const name of ["custom_metadata_text", "embed_workflow", "metadata_export_settings"]) {
    const widget = getKiriWidget(node, name);
    if (!widget) continue;
    // Persisted hidden widgets: the values stay in widgets_values / PROMPT,
    // but neither the canvas node nor the Vue panel receives a visible row.
    widget.hidden = true;
    widget.options = widget.options || {};
    widget.options.hidden = true;
    widget.computeSize = () => [0, -4];
    widget.serialize = true;
    widget.options.serialize = true;
    if (widget.element?.style) widget.element.style.display = "none";
    if (widget.inputEl?.style) widget.inputEl.style.display = "none";
  }
}

function getMetadataExportSettings(node) {
  const defaults = {
    parameters_format: "a1111",
    save_positive_prompt: true,
    save_negative_prompt: true,
    save_node_names: true,
    platform_metadata: true,
    invokeai: true,
    novelai: true,
    universal_bundle: true,
    write_sidecar: false,
    write_easy_diffusion_task: true,
    visible_preview_image: "auto",
    save_images_from: "a",
    swap_comparison_sides: false,
    save_folder_organization: "none",
  };
  try {
    const parsed = JSON.parse(String(getKiriWidgetValue(node, "metadata_export_settings", "") || "{}"));
    if (parsed && typeof parsed === "object") {
      const settings = { ...defaults, ...parsed };
      const saveImagesFrom = String(settings.save_images_from || "a").toLowerCase();
      settings.save_images_from = ["a", "b", "both"].includes(saveImagesFrom) ? saveImagesFrom : "a";
      // Migrate v0.5.20's per-generation timestamp folders to the new shared
      // month/day grouping unless the workflow already has an explicit mode.
      const folderMode = Object.prototype.hasOwnProperty.call(parsed, "save_folder_organization")
        ? String(settings.save_folder_organization || "none").toLowerCase()
        : (parsed.organize_saves_by_month_datetime ? "month_day" : "none");
      settings.save_folder_organization = ["none", "month", "month_day"].includes(folderMode) ? folderMode : "none";
      delete settings.organize_saves_by_month_datetime;
      // Migrate v0.5.11's combined prompt switch without overriding either
      // explicit v0.5.12 branch preference.
      if (Object.prototype.hasOwnProperty.call(parsed, "save_prompts")) {
        if (!Object.prototype.hasOwnProperty.call(parsed, "save_positive_prompt")) settings.save_positive_prompt = !!parsed.save_prompts;
        if (!Object.prototype.hasOwnProperty.call(parsed, "save_negative_prompt")) settings.save_negative_prompt = !!parsed.save_prompts;
      }
      delete settings.save_prompts;
      return settings;
    }
  } catch (_) { /* Keep safe defaults for older workflows or malformed widget JSON. */ }
  return defaults;
}

const VISIBLE_PREVIEW_MODES = new Set(["auto", "image_a", "image_b", "loaded"]);

function getComparisonSidesSwapped(node) {
  return !!getMetadataExportSettings(node).swap_comparison_sides;
}

function setComparisonSidesSwapped(node, value) {
  const settings = getMetadataExportSettings(node);
  settings.swap_comparison_sides = !!value;
  setKiriWidgetValue(node, "metadata_export_settings", JSON.stringify(settings));
  // This is a preview-only presentation preference. Rebuild the current wipe
  // image immediately without touching A/B source identities, sockets or outputs.
  refreshNativeLivePreview(node);
  return settings.swap_comparison_sides;
}

function getVisiblePreviewMode(node) {
  const configured = String(getMetadataExportSettings(node).visible_preview_image || "auto");
  return VISIBLE_PREVIEW_MODES.has(configured) ? configured : "auto";
}

function getLoadedFallbackPreviewImages(node) {
  try {
    const readFallback = nativePreviewFallbackReaders.get(node);
    const images = typeof readFallback === "function" ? readFallback() : node?.imgs;
    if (!Array.isArray(images)) return [];
    return images.filter(isRenderableImageSource).slice(0, 1);
  } catch {
    return [];
  }
}

function getVisiblePreviewImages(node) {
  const mode = getVisiblePreviewMode(node);
  const state = livePreviewByNode.get(node);
  const imageA = isRenderableImageSource(state?.a) ? state.a : null;
  const imageB = isRenderableImageSource(state?.b) ? state.b : null;
  const loaded = getLoadedFallbackPreviewImages(node);

  if (mode === "image_a") return imageA ? [imageA] : [];
  if (mode === "image_b") return imageB ? [imageB] : [];
  if (mode === "loaded") return loaded;

  // Auto/Compare preserves the connected-input priority. A single available
  // branch is shown alone; only when neither A nor B has a preview do we use
  // the selected Loaded Image as the final visual fallback.
  if (imageA && imageB) {
    const rendered = isRenderableImageSource(state?.renderedImage)
      ? state.renderedImage
      : createComparisonCanvas(imageA, imageB, state?.slider ?? 0.5, getComparisonSidesSwapped(node));
    return rendered ? [rendered] : [imageA];
  }
  if (imageA) return [imageA];
  if (imageB) return [imageB];
  return loaded;
}

function setVisiblePreviewMode(node, mode) {
  const nextMode = VISIBLE_PREVIEW_MODES.has(String(mode)) ? String(mode) : "auto";
  const settings = getMetadataExportSettings(node);
  settings.visible_preview_image = nextMode;
  setKiriWidgetValue(node, "metadata_export_settings", JSON.stringify(settings));
  // Re-resolve current graph inputs and repaint immediately. The choice is
  // stored in the existing hidden serialized settings widget, so no extra
  // visible node widget or geometry change is introduced.
  try { syncLivePreviewFromUpstream(node); } catch (error) {
    reportPreviewIntegrationError(node, error, "visible preview selection");
  }
  refreshNativeLivePreview(node);
  return nextMode;
}

function setMetadataExportPreference(node, key, value) {
  const settings = getMetadataExportSettings(node);
  settings[key] = !!value;
  setKiriWidgetValue(node, "metadata_export_settings", JSON.stringify(settings));
  return settings;
}

function setMetadataExportOption(node, key, value) {
  const settings = getMetadataExportSettings(node);
  settings[key] = value;
  setKiriWidgetValue(node, "metadata_export_settings", JSON.stringify(settings));
  node.setDirtyCanvas?.(true, true);
  return settings;
}

function editCustomMetadata(node) {
  const existing = String(getKiriWidgetValue(node, "custom_metadata_text", "") || "");
  const overlay = document.createElement("div");
  Object.assign(overlay.style, {
    position: "fixed", inset: "0", zIndex: "100001", display: "flex",
    alignItems: "center", justifyContent: "center", padding: "18px",
    background: "rgba(0,0,0,0.62)",
  });
  const dialog = document.createElement("div");
  Object.assign(dialog.style, {
    width: "min(680px, 94vw)", maxHeight: "86vh", display: "flex", flexDirection: "column",
    gap: "10px", padding: "16px", borderRadius: "10px", background: "var(--comfy-menu-bg, #252525)",
    color: "var(--fg-color, #eee)", border: "1px solid var(--border-color, #555)",
    boxShadow: "0 12px 44px rgba(0,0,0,.5)", font: "13px sans-serif",
  });
  const title = document.createElement("div");
  title.textContent = "Kiri Custom Metadata";
  title.style.fontWeight = "700";
  title.style.fontSize = "15px";
  const hint = document.createElement("div");
  hint.textContent = "Enter any text or structured JSON to embed with the saved image. A connected CUSTOM META input overrides this text.";
  hint.style.opacity = "0.78";
  hint.style.lineHeight = "1.45";
  const textarea = document.createElement("textarea");
  textarea.value = existing;
  textarea.placeholder = "Artist: ...\nProject: ...\nNotes: ...\n\nOr paste JSON...";
  Object.assign(textarea.style, {
    width: "100%", minHeight: "240px", maxHeight: "55vh", resize: "vertical",
    boxSizing: "border-box", padding: "10px", borderRadius: "6px",
    color: "var(--input-text, #eee)", background: "var(--comfy-input-bg, #161616)",
    border: "1px solid var(--border-color, #555)", font: "13px/1.45 monospace",
  });
  const actions = document.createElement("div");
  Object.assign(actions.style, { display: "flex", justifyContent: "flex-end", gap: "8px" });
  const makeButton = (label, primary, callback) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    Object.assign(button.style, {
      border: "1px solid var(--border-color, #555)", borderRadius: "6px",
      padding: "7px 13px", cursor: "pointer", color: primary ? "#fff" : "inherit",
      background: primary ? "var(--comfy-menu-active-bg, #2563eb)" : "transparent",
      font: "inherit",
    });
    button.addEventListener("click", callback);
    return button;
  };
  const close = () => overlay.remove();
  actions.append(
    makeButton("Cancel", false, close),
    makeButton("Save Metadata", true, () => {
      setKiriWidgetValue(node, "custom_metadata_text", textarea.value);
      close();
    }),
  );
  dialog.append(title, hint, textarea, actions);
  overlay.appendChild(dialog);
  overlay.addEventListener("pointerdown", (event) => {
    if (event.target === overlay) close();
  });
  overlay.addEventListener("keydown", (event) => {
    if (event.key === "Escape") close();
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      setKiriWidgetValue(node, "custom_metadata_text", textarea.value);
      close();
    }
  });
  document.body.appendChild(overlay);
  textarea.focus();
  textarea.setSelectionRange(textarea.value.length, textarea.value.length);
}

function configureMetadataExports(node) {
  const settings = getMetadataExportSettings(node);
  if (!["a1111", "swarmui", "fooocus", "easy_diffusion"].includes(settings.parameters_format)) settings.parameters_format = "a1111";

  const overlay = document.createElement("div");
  Object.assign(overlay.style, {
    position: "fixed", inset: "0", zIndex: "100001", display: "flex",
    alignItems: "center", justifyContent: "center", padding: "18px",
    background: "rgba(0,0,0,0.66)",
  });
  const dialog = document.createElement("div");
  Object.assign(dialog.style, {
    width: "min(620px, 94vw)", maxHeight: "88vh", overflow: "auto",
    display: "flex", flexDirection: "column", gap: "12px", padding: "17px",
    borderRadius: "10px", background: "var(--comfy-menu-bg, #252525)",
    color: "var(--fg-color, #eee)", border: "1px solid var(--border-color, #555)",
    boxShadow: "0 12px 44px rgba(0,0,0,.5)", font: "13px sans-serif",
  });
  const title = document.createElement("div");
  title.textContent = "Metadata Export Compatibility";
  Object.assign(title.style, { fontWeight: "700", fontSize: "16px" });
  const intro = document.createElement("div");
  intro.textContent = "Kiri writes compatible parameter metadata for other apps independently from ComfyUI workflow embedding. The right-click Metadata Saving menu independently controls positive prompt text, negative prompt text, a plain node-name list, and other-platform metadata. Platform records contain generation parameters and resource details only, never ComfyUI node objects or graph links.";
  Object.assign(intro.style, { opacity: "0.8", lineHeight: "1.45" });

  const section = (label) => {
    const heading = document.createElement("div");
    heading.textContent = label;
    Object.assign(heading.style, { fontWeight: "700", marginTop: "3px" });
    return heading;
  };
  const select = document.createElement("select");
  Object.assign(select.style, {
    width: "100%", padding: "8px", borderRadius: "6px", boxSizing: "border-box",
    color: "var(--input-text, #eee)", background: "var(--comfy-input-bg, #161616)",
    border: "1px solid var(--border-color, #555)", font: "inherit",
  });
  for (const [value, label] of [
    ["a1111", "A1111 / Forge / Civitai-compatible parameters (recommended)"],
    ["swarmui", "SwarmUI native parameters JSON"],
    ["fooocus", "Fooocus native JSON scheme (Load Parameters)"],
    ["easy_diffusion", "Easy Diffusion native task JSON profile"],
  ]) {
    const option = document.createElement("option");
    option.value = value; option.textContent = label;
    select.appendChild(option);
  }
  select.value = settings.parameters_format;
  const selectHint = document.createElement("div");
  selectHint.textContent = "The PNG `parameters` field can hold only one native profile at a time. The separate Other-Platform Metadata switch controls all app/site exports, including Draw Things XMP/UserComment and NovelAI stealth-alpha metadata. It never embeds ComfyUI node graphs.";
  Object.assign(selectHint.style, { opacity: "0.7", lineHeight: "1.4", fontSize: "12px" });

  const checks = [];
  const addCheck = (key, label, description) => {
    const row = document.createElement("label");
    Object.assign(row.style, { display: "flex", gap: "9px", alignItems: "flex-start", padding: "7px 8px", borderRadius: "6px", background: "rgba(128,128,128,.10)", cursor: "pointer" });
    const input = document.createElement("input");
    input.type = "checkbox"; input.checked = !!settings[key]; input.style.marginTop = "2px";
    const textWrap = document.createElement("span");
    const name = document.createElement("span"); name.textContent = label; name.style.fontWeight = "600";
    const detail = document.createElement("div"); detail.textContent = description;
    Object.assign(detail.style, { opacity: "0.72", fontSize: "12px", marginTop: "2px", lineHeight: "1.35" });
    textWrap.append(name, detail); row.append(input, textWrap);
    checks.push({ key, input });
    return row;
  };
  const note = document.createElement("div");
  note.textContent = "Fooocus has its native JSON Load Parameters profile; Easy Diffusion gets a drag-and-drop task JSON sidecar. Draw Things receives its XMP description and JSON UserComment. NovelAI gets the stealth_pngcomp alpha payload (without NovelAI's signature/FEC). PixAI v2 and Tensor.Art TAMS API bodies require provider-native model IDs; SeaArt and TensorHub / Cozy Creator are portable handoff templates, not verified native image-import formats. Naifu receives a copyable local parameter handoff; import support depends on the specific Naifu fork.";
  Object.assign(note.style, { opacity: "0.74", lineHeight: "1.4", fontSize: "12px", padding: "8px 10px", borderLeft: "3px solid var(--border-color, #777)" });

  const actions = document.createElement("div");
  Object.assign(actions.style, { display: "flex", justifyContent: "flex-end", gap: "8px", marginTop: "4px" });
  const makeButton = (label, primary, callback) => {
    const button = document.createElement("button");
    button.type = "button"; button.textContent = label;
    Object.assign(button.style, {
      border: "1px solid var(--border-color, #555)", borderRadius: "6px", padding: "7px 13px",
      cursor: "pointer", color: primary ? "#fff" : "inherit",
      background: primary ? "var(--comfy-menu-active-bg, #2563eb)" : "transparent", font: "inherit",
    });
    button.addEventListener("click", callback); return button;
  };
  const close = () => overlay.remove();
  actions.append(
    makeButton("Cancel", false, close),
    makeButton("Save Export Settings", true, () => {
      settings.parameters_format = select.value;
      for (const item of checks) settings[item.key] = item.input.checked;
      setKiriWidgetValue(node, "metadata_export_settings", JSON.stringify(settings));
      close();
    }),
  );
  dialog.append(title, intro, section("Parameters field format"), select, selectHint,
    section("Additional native metadata"),
    addCheck("invokeai", "InvokeAI metadata", "Writes InvokeAI's core record. Model catalog identifiers are included only when a complete native identifier is available."),
    addCheck("novelai", "NovelAI metadata (including stealth PNG parameters)", "Writes Description / Comment plus NovelAI's stealth_pngcomp alpha payload. It does not claim a NovelAI signature or add optional FEC."),
    addCheck("universal_bundle", "Universal platform handoff bundle", "One kiri_interop JSON record containing Civitai resource metadata, structured PixAI v2 / Tensor.Art TAMS requests, and copyable Fooocus, Naifu, SeaArt and TensorHub handoffs."),
    addCheck("write_sidecar", "Also write a .kiri.json sidecar", "Creates a complete Kiri interchange manifest beside each PNG."),
    addCheck("write_easy_diffusion_task", "Also write an Easy Diffusion task JSON", "Creates a matching .easy-diffusion.json file that you can drag onto Easy Diffusion to load the prompt and available settings."),
    note, actions);
  overlay.appendChild(dialog);
  overlay.addEventListener("pointerdown", (event) => { if (event.target === overlay) close(); });
  overlay.addEventListener("keydown", (event) => { if (event.key === "Escape") close(); });
  document.body.appendChild(overlay);
  select.focus();
}

function saveFolderOrganizationLabel(mode) {
  if (mode === "month") return "Month";
  if (mode === "month_day") return "Month + Day";
  return "None";
}

function imageSavingMenuItems(node) {
  const settings = getMetadataExportSettings(node);
  const folderMode = settings.save_folder_organization;
  return [
    { content: `${settings.save_images_from === "a" ? "✓ " : ""}Input A only`, callback: () => setMetadataExportOption(node, "save_images_from", "a") },
    { content: `${settings.save_images_from === "b" ? "✓ " : ""}Input B only`, callback: () => setMetadataExportOption(node, "save_images_from", "b") },
    { content: `${settings.save_images_from === "both" ? "✓ " : ""}Both A and B`, callback: () => setMetadataExportOption(node, "save_images_from", "both") },
    null,
    {
      content: `Save Folder Organization: ${saveFolderOrganizationLabel(folderMode)}`,
      has_submenu: true,
      submenu: {
        title: "Save Folder Organization",
        options: [
          { content: `${folderMode === "none" ? "✓ " : ""}None (Save Path)`, callback: () => setMetadataExportOption(node, "save_folder_organization", "none") },
          { content: `${folderMode === "month" ? "✓ " : ""}Month (YYYY-MM)`, callback: () => setMetadataExportOption(node, "save_folder_organization", "month") },
          { content: `${folderMode === "month_day" ? "✓ " : ""}Month + Day (YYYY-MM/YYYY-MM-DD)`, callback: () => setMetadataExportOption(node, "save_folder_organization", "month_day") },
        ],
      },
      callback: () => {},
    },
  ];
}

function metadataSavingMenuItems(node) {
  const manual = String(getKiriWidgetValue(node, "custom_metadata_text", "") || "");
  const customInput = node?.inputs?.find((input) => input?.name === "custom_metadata");
  const connectedOverride = isSocketConnected(customInput);
  const settings = getMetadataExportSettings(node);
  return [
    {
      content: `Save Positive Prompt: ${settings.save_positive_prompt ? "ON" : "OFF"}`,
      callback: () => setMetadataExportPreference(node, "save_positive_prompt", !settings.save_positive_prompt),
    },
    {
      content: `Save Negative Prompt: ${settings.save_negative_prompt ? "ON" : "OFF"}`,
      callback: () => setMetadataExportPreference(node, "save_negative_prompt", !settings.save_negative_prompt),
    },
    {
      content: `Save Node Names (Nodes Used): ${settings.save_node_names ? "ON" : "OFF"}`,
      callback: () => setMetadataExportPreference(node, "save_node_names", !settings.save_node_names),
    },
    {
      content: `Other-Platform Metadata: ${settings.platform_metadata ? "ON" : "OFF"}`,
      callback: () => setMetadataExportPreference(node, "platform_metadata", !settings.platform_metadata),
    },
    null,
    {
      content: "Compatibility Export Settings…",
      callback: () => configureMetadataExports(node),
    },
    null,
    {
      content: `Edit Custom Metadata${manual.trim() ? " (configured)" : "…"}`,
      callback: () => editCustomMetadata(node),
    },
    {
      content: connectedOverride ? "Connected Custom Metadata Overrides Manual Text" : "No Connected Custom Metadata Input",
      disabled: true,
    },
    {
      content: "Clear Manual Custom Metadata",
      disabled: !manual,
      callback: () => setKiriWidgetValue(node, "custom_metadata_text", ""),
    },
  ];
}


function menuItemsFor(node) {
  syncAllMetadataSources(node);
  const embed = !!getKiriWidgetValue(node, "embed_workflow", false);
  return [
    {
      content: `Embed Complete Workflow: ${embed ? "ON" : "OFF"}`,
      callback: () => {
        setKiriWidgetValue(node, "embed_workflow", !embed);
        console.info(`[Kiri Image Loader] Full workflow embedding ${!embed ? "enabled" : "disabled"}.`);
      },
    },
    null,
    {
      content: "Visible Preview Image",
      has_submenu: true,
      submenu: {
        title: "Visible Preview Image",
        options: [
          { content: `${getVisiblePreviewMode(node) === "auto" ? "✓ " : ""}Auto / Compare (A + B)`, callback: () => setVisiblePreviewMode(node, "auto") },
          { content: `${getVisiblePreviewMode(node) === "image_a" ? "✓ " : ""}Image A only`, callback: () => setVisiblePreviewMode(node, "image_a") },
          { content: `${getVisiblePreviewMode(node) === "image_b" ? "✓ " : ""}Image B only`, callback: () => setVisiblePreviewMode(node, "image_b") },
          { content: `${getVisiblePreviewMode(node) === "loaded" ? "✓ " : ""}Loaded Image only`, callback: () => setVisiblePreviewMode(node, "loaded") },
          null,
          { content: `A on Right (Swap Sides): ${getComparisonSidesSwapped(node) ? "ON" : "OFF"}`, callback: () => setComparisonSidesSwapped(node, !getComparisonSidesSwapped(node)) },
        ],
      },
      callback: () => {},
    },
    null,
    sourceMenuEntry(node, "image_a", "Image A"),
    sourceMenuEntry(node, "image_b", "Image B"),
    null,
    sourceMenuEntry(node, "loaded", "Loaded Image"),
    null,
    {
      content: "Image Saving",
      has_submenu: true,
      submenu: { options: imageSavingMenuItems(node), title: "Image Saving" },
      callback: () => {},
    },
    null,
    {
      content: "Metadata Saving",
      has_submenu: true,
      submenu: { options: metadataSavingMenuItems(node), title: "Metadata Saving" },
      callback: () => {},
    },
  ];
}

function loadPreviewAsset(record) {
  // Points Editor-style inline image payload: the backend returns Base64
  // directly in the node's executed UI message, so no temp-file write or
  // secondary /view request is needed.
  if (!record) return Promise.resolve(null);
  const encoded = typeof record === "string"
    ? record
    : (record.data || record.base64 || record.image || "");
  if (!encoded) return Promise.resolve(null);

  const mime = typeof record === "object" && record.mime
    ? record.mime
    : "image/jpeg";
  const source = encoded.startsWith("data:")
    ? encoded
    : `data:${mime};base64,${encoded}`;

  return new Promise((resolve) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => {
      console.warn("[Kiri Image Loader] Could not decode inline live preview image.");
      resolve(null);
    };
    image.src = source;
  });
}

function unwrapUIValue(source, key) {
  const value = source?.[key] ?? source?.ui?.[key] ?? source?.output?.[key] ?? source?.output?.ui?.[key];
  return Array.isArray(value) ? value[0] : value;
}

function extractLivePreviewPayload(output) {
  // Match Points Editor's UI contract: Python returns one-element arrays
  // (for example ui.bg_image = [base64]); ComfyUI exposes their first values
  // directly on the message passed to node.onExecuted.
  const mode = unwrapUIValue(output, "kiri_preview_mode");
  const a = unwrapUIValue(output, "kiri_preview_a");
  const b = unwrapUIValue(output, "kiri_preview_b");
  if (mode !== undefined || a !== undefined || b !== undefined) {
    return { mode: mode || (a && b ? "compare" : "single"), a: a || null, b: b || null };
  }

  // Read the earlier v0.4.0 nested shape as a compatibility fallback.
  const legacyCandidates = [
    output?.kiri_live_preview,
    output?.ui?.kiri_live_preview,
    output?.output?.kiri_live_preview,
    output?.output?.ui?.kiri_live_preview,
  ];
  for (const candidate of legacyCandidates) {
    if (candidate !== undefined && candidate !== null) {
      return Array.isArray(candidate) ? candidate[0] : candidate;
    }
  }
  return null;
}

async function updateLivePreview(node, output) {
  const serial = (livePreviewSerialByNode.get(node) || 0) + 1;
  livePreviewSerialByNode.set(node, serial);
  const unwrapNested = (key) => {
    let value = unwrapUIValue(output, key);
    if (Array.isArray(value) && Array.isArray(value[0])) value = value[0];
    return value;
  };
  const savedRecords = unwrapNested("kiri_saved_files");
  const savedRecordsA = unwrapNested("kiri_saved_files_a");
  const savedRecordsB = unwrapNested("kiri_saved_files_b");
  const firstSavedRecord = Array.isArray(savedRecordsA) ? savedRecordsA[0]
    : (Array.isArray(savedRecords) ? savedRecords[0] : savedRecords);
  const firstSavedRecordB = Array.isArray(savedRecordsB) ? savedRecordsB[0] : savedRecordsB;
  const runtimeA = parseRuntimeMetadata(unwrapNested("kiri_runtime_metadata_a"));
  const runtimeB = parseRuntimeMetadata(unwrapNested("kiri_runtime_metadata_b"));
  const savedOutputA = descriptorFromOutputRecord(firstSavedRecord, node);
  const savedOutputB = descriptorFromOutputRecord(firstSavedRecordB, node);
  const previousExecution = executionMetadataByNode.get(node);
  executionMetadataByNode.set(node, {
    serial: (previousExecution?.serial || 0) + 1,
    savedOutputA,
    savedOutputB,
    metadataA: runtimeA,
    metadataB: runtimeB,
    sourceIdentities: {
      image_a: connectedSourceIdentity(node, "image_a"),
      image_b: connectedSourceIdentity(node, "image_b"),
    },
  });
  const payload = extractLivePreviewPayload(output);

  if (!payload || payload.mode === "file") {
    livePreviewByNode.delete(node);
    clearPersistedLivePreview(node);
    refreshNativeLivePreview(node, null);
    syncAllMetadataSources(node);
    node.setDirtyCanvas?.(true, true);
    return;
  }

  const state = {
    mode: payload.mode === "compare" ? "compare" : "single",
    a: null,
    b: null,
    slider: livePreviewByNode.get(node)?.slider ?? 0.5,
  };
  livePreviewByNode.set(node, state);

  const [imageA, imageB] = await Promise.all([
    loadPreviewAsset(payload.a),
    loadPreviewAsset(payload.b),
  ]);
  if (livePreviewSerialByNode.get(node) !== serial) return;

  state.a = imageA;
  state.b = imageB;
  if (!state.a && !state.b) {
    livePreviewByNode.delete(node);
    clearPersistedLivePreview(node);
    refreshNativeLivePreview(node, null);
  } else {
    state.mode = state.a && state.b ? "compare" : "single";
    persistLivePreview(node, payload, state);
    refreshNativeLivePreview(node, state);
  }
  syncAllMetadataSources(node);
  node.setDirtyCanvas?.(true, true);
}

function drawContainedImage(ctx, image, rect) {
  if (!isRenderableImageSource(image)) return false;
  const sourceWidth = Number(image.naturalWidth || image.videoWidth || image.width);
  const sourceHeight = Number(image.naturalHeight || image.videoHeight || image.height);
  const scale = Math.min(rect[2] / sourceWidth, rect[3] / sourceHeight);
  const width = sourceWidth * scale;
  const height = sourceHeight * scale;
  const x = rect[0] + (rect[2] - width) / 2;
  const y = rect[1] + (rect[3] - height) / 2;
  try {
    ctx.drawImage(image, x, y, width, height);
    return true;
  } catch (error) {
    console.debug("[Kiri Image Loader] Skipped an unavailable upstream preview image.", error);
    return false;
  }
}

function createComparisonCanvas(imageA, imageB, slider = 0.5, swapSides = false) {
  const validA = isRenderableImageSource(imageA) ? imageA : null;
  const validB = isRenderableImageSource(imageB) ? imageB : null;
  const base = validA || validB;
  if (!base) return null;
  if (!validA || !validB) return base;
  imageA = validA;
  imageB = validB;

  const sourceWidth = Math.max(1, Math.round(Number(base.naturalWidth || base.width)));
  const sourceHeight = Math.max(1, Math.round(Number(base.naturalHeight || base.height)));
  // This canvas is display-only; never duplicate a multi-megapixel generation
  // at full resolution while ComfyUI is painting the graph. Source tensors/files
  // and downstream output values are untouched. The cap is ample for the
  // on-node preview and avoids browser memory spikes for tiled/upscaled outputs.
  const scale = Math.min(
    1,
    Math.sqrt(MAX_COMPARISON_PREVIEW_PIXELS / (sourceWidth * sourceHeight)),
    MAX_COMPARISON_PREVIEW_EDGE / Math.max(sourceWidth, sourceHeight)
  );
  const width = Math.max(1, Math.round(sourceWidth * scale));
  const height = Math.max(1, Math.round(sourceHeight * scale));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  // ComfyUI's native image-preview widget checks naturalWidth/naturalHeight
  // before drawing. A canvas is a valid drawImage source, so expose matching
  // read-only dimensions and let the standard widget render it normally.
  try {
    Object.defineProperty(canvas, "naturalWidth", { configurable: true, get: () => canvas.width });
    Object.defineProperty(canvas, "naturalHeight", { configurable: true, get: () => canvas.height });
  } catch { /* width/height remain available to older preview implementations */ }

  const ctx = canvas.getContext("2d");
  if (!ctx) return base;
  const rect = [0, 0, width, height];
  ctx.fillStyle = "#171717";
  ctx.fillRect(0, 0, width, height);
  const leftImage = swapSides ? imageB : imageA;
  const rightImage = swapSides ? imageA : imageB;
  drawContainedImage(ctx, leftImage, rect);

  const split = Math.max(0, Math.min(1, slider));
  const splitX = Math.round(width * split);
  ctx.save();
  ctx.beginPath();
  ctx.rect(splitX, 0, width - splitX, height);
  ctx.clip();
  drawContainedImage(ctx, rightImage, rect);
  ctx.restore();

  // Clean comparer-style divider: deliberately no A/B labels or badges.
  ctx.save();
  ctx.strokeStyle = "rgba(255,255,255,0.96)";
  ctx.lineWidth = Math.max(1, width / 600);
  ctx.beginPath();
  ctx.moveTo(splitX, 0);
  ctx.lineTo(splitX, height);
  ctx.stroke();
  const radius = Math.max(7, Math.min(13, width / 65));
  const centerY = height / 2;
  ctx.beginPath();
  ctx.arc(splitX, centerY, radius, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(20,20,20,0.88)";
  ctx.fill();
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 1.25;
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(splitX - radius * 0.4, centerY);
  ctx.lineTo(splitX + radius * 0.4, centerY);
  ctx.stroke();
  ctx.restore();
  return canvas;
}

function refreshNativeLivePreview(node, state = livePreviewByNode.get(node)) {
  if (state && (state.a || state.b)) {
    state.renderedImage = state.a && state.b
      ? createComparisonCanvas(state.a, state.b, state.slider, getComparisonSidesSwapped(node))
      : (state.a || state.b);
  }
  node.setDirtyCanvas?.(true, true);
}

function graphPointFromPointer(canvas, event) {
  // ComfyUI CanvasPointer events already carry graph-space coordinates.
  if (Number.isFinite(event?.canvasX) && Number.isFinite(event?.canvasY)) {
    return [event.canvasX, event.canvasY];
  }
  try {
    const canvasPoint = canvas.convertEventToCanvas?.(event);
    const graphPoint = canvasPoint && canvas.convertCanvasToGraph?.(canvasPoint);
    if (Array.isArray(graphPoint) && graphPoint.length >= 2 && graphPoint.every(Number.isFinite)) {
      return graphPoint;
    }
  } catch { /* use the DOM transform fallback */ }

  const element = canvas.canvas;
  const bounds = element?.getBoundingClientRect?.();
  if (!element || !bounds || !canvas.ds) return null;
  const x = (event.clientX - bounds.left) * (element.width / Math.max(1, bounds.width));
  const y = (event.clientY - bounds.top) * (element.height / Math.max(1, bounds.height));
  const scale = Number(canvas.ds.scale) || 1;
  const offset = canvas.ds.offset || [0, 0];
  return [(x - Number(offset[0] || 0)) / scale, (y - Number(offset[1] || 0)) / scale];
}

function comparisonHandlePoint(rect, slider) {
  if (!Array.isArray(rect) || rect.length < 4) return null;
  return [
    rect[0] + rect[2] * Math.max(0, Math.min(1, Number(slider) || 0)),
    rect[1] + rect[3] / 2,
  ];
}

function isNearComparisonHandle(node, state, graphPoint, canvas) {
  if (!node || getVisiblePreviewMode(node) !== "auto" || !state?.a || !state?.b || state.mode !== "compare" || !graphPoint) return false;
  const rect = currentPreviewRectangles(node)[0];
  const handle = comparisonHandlePoint(rect, state.slider);
  if (!handle) return false;

  const localX = graphPoint[0] - Number(node.pos?.[0] || 0);
  const localY = graphPoint[1] - Number(node.pos?.[1] || 0);
  // The visible handle is small at normal node zoom. Give that visible point
  // a comfortable circular grab target (18 CSS px), but do not make the whole
  // preview or invisible left/right zones draggable.
  const zoom = Math.max(0.1, Number(canvas?.ds?.scale) || 1);
  const grabRadius = 18 / zoom;
  return Math.hypot(localX - handle[0], localY - handle[1]) <= grabRadius;
}

function updateComparisonSliderAtGraphPoint(node, state, point, grabOffsetX = 0) {
  if (!point) return false;
  const rect = currentPreviewRectangles(node)[0];
  if (!rect || !(rect[2] > 0)) return false;
  // Keep the original grab offset so picking up the edge of the handle does
  // not cause the divider to jump underneath the cursor on pointer-down.
  const localX = point[0] - Number(node.pos?.[0] || 0) - Number(grabOffsetX || 0);
  state.slider = Math.max(0, Math.min(1, (localX - rect[0]) / rect[2]));
  persistComparisonSlider(node, state);
  refreshNativeLivePreview(node, state);
  return true;
}

function updateComparisonSliderFromPointer(canvas, node, state, event, grabOffsetX = 0) {
  return updateComparisonSliderAtGraphPoint(
    node, state, graphPointFromPointer(canvas, event), grabOffsetX
  );
}

function installCompareWidgetPointer(node, canvas) {
  if (!canvas?.pointer || !node) return false;
  const widget = getImageWidget(node);
  if (!widget) return false;
  const installed = compareWidgetHandlers.get(widget);
  if (installed && widget.onPointerDown === installed.wrapper) return true;

  const previous = typeof widget.onPointerDown === "function" ? widget.onPointerDown : null;
  const wrapper = function (pointer, ownerNode, ownerCanvas) {
    const activeNode = ownerNode || node;
    const activeCanvas = ownerCanvas || canvas;
    const downEvent = pointer?.eDown;
    const downPoint = graphPointFromPointer(activeCanvas, downEvent);
    const state = livePreviewByNode.get(activeNode);

    // Only the visible center handle claims this gesture. Everywhere else the
    // stock IMAGEUPLOAD widget and ComfyUI pointer behavior remain untouched.
    if (!downPoint || !isNearComparisonHandle(activeNode, state, downPoint, activeCanvas)) {
      return previous ? previous.call(this, pointer, activeNode, activeCanvas) : false;
    }

    const rect = currentPreviewRectangles(activeNode)[0];
    const handle = comparisonHandlePoint(rect, state.slider);
    const grabOffsetX = downPoint[0] - Number(activeNode.pos?.[0] || 0) - handle[0];
    let finished = false;
    const update = (event) => {
      const point = graphPointFromPointer(activeCanvas, event);
      if (point) updateComparisonSliderAtGraphPoint(activeNode, state, point, grabOffsetX);
    };
    const finish = (event, applyFinalPosition) => {
      if (finished) return;
      finished = true;
      if (applyFinalPosition && event) update(event);
    };

    // CanvasPointer owns capture, click/drag thresholds, and cancellation.
    // Its onDrag callback receives graph-space pointer coordinates.
    pointer.onDrag = (moveEvent) => update(moveEvent);
    pointer.onDragEnd = (upEvent) => finish(upEvent, true);
    pointer.onDragCancel = () => finish(null, false);
    pointer.finally = () => finish(null, false);
    return true;
  };

  compareWidgetHandlers.set(widget, { wrapper, previous });
  widget.onPointerDown = wrapper;
  return true;
}

function installPreviewInteractionSurface(node) {
  if (!node || typeof node.getWidgetOnPos !== "function") return false;
  const installed = previewSurfaceHandlers.get(node);
  if (installed && node.getWidgetOnPos === installed.wrapper) return true;

  const previous = node.getWidgetOnPos;
  const surfaceWidget = {
    // Virtual hit-target only: it is returned by getWidgetOnPos for the image
    // preview, but is deliberately NOT inserted into node.widgets. It adds no
    // visible control, widget row, serialized value, or node height.
    name: "kiri_preview_interaction_surface",
    type: "custom",
    value: "",
    serialize: false,
    onPointerDown(pointer, ownerNode, ownerCanvas) {
      const activeNode = ownerNode || node;
      const activeCanvas = ownerCanvas || app.canvas;
      const downEvent = pointer?.eDown;
      const downPoint = graphPointFromPointer(activeCanvas, downEvent);
      const state = livePreviewByNode.get(activeNode);
      const isHandle = downPoint && isNearComparisonHandle(activeNode, state, downPoint, activeCanvas);

      // Claim every pointer gesture over the displayed image. This prevents
      // LiteGraph from interpreting the preview as node background and moving
      // the entire node. A click may still select Kiri, but dragging the image
      // itself does nothing.
      pointer.onClick = () => activeCanvas?.processSelect?.(activeNode, pointer.eDown);
      pointer.onDoubleClick = () => {};
      pointer.onDragStart = () => {};
      pointer.onDrag = () => {};
      pointer.onDragEnd = () => {};
      pointer.onDragCancel = () => {};
      pointer.finally = () => {};

      // Only the visible centre handle controls the divider. Retain the exact
      // grab offset, and let CanvasPointer own the gesture lifecycle.
      if (isHandle && getVisiblePreviewMode(activeNode) === "auto" && state?.a && state?.b && state.mode === "compare") {
        const rect = currentPreviewRectangles(activeNode)[0];
        const handle = comparisonHandlePoint(rect, state.slider);
        const grabOffsetX = downPoint[0] - Number(activeNode.pos?.[0] || 0) - handle[0];
        let finished = false;
        const update = (event) => {
          const point = graphPointFromPointer(activeCanvas, event);
          if (point) updateComparisonSliderAtGraphPoint(activeNode, state, point, grabOffsetX);
        };
        const finish = (event, applyFinalPosition) => {
          if (finished) return;
          finished = true;
          if (applyFinalPosition && event) update(event);
        };
        pointer.onDrag = (moveEvent) => update(moveEvent);
        pointer.onDragEnd = (upEvent) => finish(upEvent, true);
        pointer.onDragCancel = () => finish(null, false);
        pointer.finally = () => finish(null, false);
      }

      // Returning true is LiteGraph's documented signal to skip its normal
      // node/widget handling for this pointer gesture. Explicit no-op drag
      // handlers above also prevent a canvas-pan fallback on older CanvasPointer
      // builds that infer panning from an unconfigured pointer.
      return true;
    },
    // Legacy LiteGraph dispatches widget.mouse instead of CanvasPointer. The
    // widget hit still consumes the gesture before normal node dragging.
    mouse() { return true; },
  };

  const wrapper = function (canvasX, canvasY, includeDisabled = false) {
    if (!this.flags?.collapsed && Number.isFinite(canvasX) && Number.isFinite(canvasY)) {
      const localX = canvasX - Number(this.pos?.[0] || 0);
      const localY = canvasY - Number(this.pos?.[1] || 0);
      if (currentPreviewRectangles(this).some((rect) => rectContains(rect, localX, localY))) {
        return surfaceWidget;
      }
    }
    return previous.call(this, canvasX, canvasY, includeDisabled);
  };

  node.getWidgetOnPos = wrapper;
  previewSurfaceHandlers.set(node, { previous, wrapper, surfaceWidget });
  return true;
}

function findPreviewSurfaceAtPoint(canvas, graphPoint) {
  if (!graphPoint) return null;
  const nodes = canvas.graph?._nodes || app.graph?._nodes || [];
  for (const node of [...nodes].reverse()) {
    if (!isKiriNode(node) || node.flags?.collapsed) continue;
    const localX = graphPoint[0] - Number(node.pos?.[0] || 0);
    const localY = graphPoint[1] - Number(node.pos?.[1] || 0);
    if (currentPreviewRectangles(node).some((rect) => rectContains(rect, localX, localY))) {
      return node;
    }
  }
  return null;
}

function findComparisonHandleAtPoint(canvas, graphPoint) {
  const nodes = canvas.graph?._nodes || app.graph?._nodes || [];
  for (const node of [...nodes].reverse()) {
    if (!isKiriNode(node)) continue;
    const state = livePreviewByNode.get(node);
    if (isNearComparisonHandle(node, state, graphPoint, canvas)) return { node, state };
  }
  return null;
}

function attachComparePointerHandler(canvas) {
  if (!canvas || comparePointerCanvases.has(canvas) || !canvas.canvas?.addEventListener) return;
  comparePointerCanvases.add(canvas);
  const element = canvas.canvas;
  const modernPointerApi = !!canvas.pointer;

  // Cursor feedback is passive; the actual modern drag is registered through
  // widget.onPointerDown so it participates in ComfyUI's own CanvasPointer UX.
  element.addEventListener("pointermove", (event) => {
    if (compareDragByCanvas.has(canvas)) return;
    const point = graphPointFromPointer(canvas, event);
    const nearHandle = point && !!findComparisonHandleAtPoint(canvas, point);
    if (element.style) element.style.cursor = nearHandle ? "ew-resize" : "";
  }, true);

  if (modernPointerApi) return;

  // Compatibility path for older LiteGraph frontends without CanvasPointer.
  // Claim the entire preview surface so it cannot start a node drag. Only the
  // visible circular handle starts a comparer drag; the rest is inert.
  element.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || compareDragByCanvas.has(canvas)) return;
    const point = graphPointFromPointer(canvas, event);
    if (!point) return;
    const previewNode = findPreviewSurfaceAtPoint(canvas, point);
    if (!previewNode) return;
    const target = findComparisonHandleAtPoint(canvas, point);
    if (!target) {
      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation?.();
      return;
    }

    const rect = currentPreviewRectangles(target.node)[0];
    const handle = comparisonHandlePoint(rect, target.state.slider);
    const grabOffsetX = point[0] - Number(target.node.pos?.[0] || 0) - handle[0];
    const drag = { node: target.node, state: target.state, pointerId: event.pointerId, grabOffsetX };
    compareDragByCanvas.set(canvas, drag);
    try {
      if (event.pointerId !== undefined && typeof element.setPointerCapture === "function") {
        element.setPointerCapture(event.pointerId);
      }
    } catch { drag.documentFallback = true; }

    updateComparisonSliderFromPointer(canvas, drag.node, drag.state, event, drag.grabOffsetX);
    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation?.();
  }, true);

  const onMove = (event) => {
    const drag = compareDragByCanvas.get(canvas);
    if (!drag) return;
    if (drag.pointerId !== undefined && event.pointerId !== undefined && drag.pointerId !== event.pointerId) return;
    updateComparisonSliderFromPointer(canvas, drag.node, drag.state, event, drag.grabOffsetX);
    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation?.();
  };
  const onUp = (event) => {
    const drag = compareDragByCanvas.get(canvas);
    if (!drag) return;
    if (drag.pointerId !== undefined && event?.pointerId !== undefined && drag.pointerId !== event.pointerId) return;
    compareDragByCanvas.delete(canvas);
    try {
      if (drag.pointerId !== undefined && typeof element.releasePointerCapture === "function" &&
          element.hasPointerCapture?.(drag.pointerId)) {
        element.releasePointerCapture(drag.pointerId);
      }
    } catch { /* pointer may already have been released */ }
    if (element.style) element.style.cursor = "";
    event?.preventDefault?.();
    event?.stopPropagation?.();
    event?.stopImmediatePropagation?.();
  };
  element.addEventListener("pointermove", onMove, true);
  element.addEventListener("pointerup", onUp, true);
  element.addEventListener("pointercancel", onUp, true);
  element.addEventListener("lostpointercapture", onUp, true);
}
function rectContains(rect, x, y) {
  return Array.isArray(rect) && rect.length >= 4 &&
    x >= rect[0] && y >= rect[1] && x <= rect[0] + rect[2] && y <= rect[1] + rect[3];
}

function getPreviewRectangles(node, preferredImage = null) {
  const liveState = livePreviewByNode.get(node);
  const liveImage = preferredImage || liveState?.a || liveState?.b || null;

  // Modern thumbnail-preview widgets expose exact image hit rectangles.
  if (!liveImage && node?.imageIndex == null && Array.isArray(node?.imageRects) && node.imageRects.length) {
    return node.imageRects.filter((rect) => Array.isArray(rect) && rect.length >= 4);
  }

  const imgs = liveImage ? [liveImage] : (Array.isArray(node?.imgs) ? node.imgs : []);
  if (!imgs.length) return [];

  const imageIndex = Number.isInteger(node.imageIndex) ? node.imageIndex : 0;
  const img = imgs[Math.min(Math.max(imageIndex, 0), imgs.length - 1)];
  const naturalWidth = Number(img?.naturalWidth || img?.width);
  const naturalHeight = Number(img?.naturalHeight || img?.height);
  if (!(naturalWidth > 0 && naturalHeight > 0)) return [];

  // A newer preview widget publishes its own vertical position and height.
  const previewWidget = node.widgets?.find((widget) =>
    /image.?preview/i.test(String(widget?.name || "")) || widget?.type === "image_preview"
  );

  let x;
  let y;
  let width;
  let height;
  if (previewWidget) {
    const top = Number(previewWidget.y ?? previewWidget.last_y ?? 0);
    const availableHeight = Number(
      previewWidget.computedHeight ?? previewWidget.height ?? previewWidget.options?.height ?? 220
    );
    const availableWidth = Number(node.size?.[0] || 320);
    const drawableHeight = Math.max(1, availableHeight - 15);
    const scale = Math.min(availableWidth / naturalWidth, drawableHeight / naturalHeight, 1);
    width = naturalWidth * scale;
    height = naturalHeight * scale;
    x = (availableWidth - width) / 2;
    y = top + (drawableHeight - height) / 2;
    return [[x, y, width, height]];
  }

  // Legacy IMAGEUPLOAD nodes draw the preview immediately after their widgets.
  // Compute the end of the widget stack rather than relying on a fixed y-offset,
  // since upload controls and ComfyUI versions can vary slightly.
  let widgetsBottom = 0;
  for (const widget of node.widgets || []) {
    const widgetY = Number(widget?.last_y ?? widget?.y);
    if (!Number.isFinite(widgetY)) continue;
    let widgetHeight = Number(widget?.height ?? widget?.options?.height);
    if (!(widgetHeight > 0)) {
      try {
        const measured = widget?.computeSize?.(Number(node.size?.[0] || 320));
        widgetHeight = Number(Array.isArray(measured) ? measured[1] : measured?.height);
      } catch { /* use the conservative fallback below */ }
    }
    if (!(widgetHeight > 0)) widgetHeight = widget?.type === "button" ? 28 : 22;
    widgetsBottom = Math.max(widgetsBottom, widgetY + widgetHeight);
  }

  const nodeWidth = Number(node.size?.[0] || 320);
  const nodeHeight = Number(node.size?.[1] || 420);
  const previewTop = widgetsBottom + 4;
  const availableWidth = Math.max(1, nodeWidth - 48);
  const availableHeight = Math.max(1, nodeHeight - previewTop - 24);
  const scale = Math.min(availableWidth / naturalWidth, availableHeight / naturalHeight, 1);
  width = naturalWidth * scale;
  height = naturalHeight * scale;
  x = (nodeWidth - width) / 2;
  y = previewTop + (availableHeight - height) / 2;
  return [[x, y, width, height]];
}

function currentPreviewRectangles(node) {
  const connected = hasConnectedImageInput(node);
  const state = livePreviewByNode.get(node);
  const imgs = Array.isArray(node?.imgs) ? node.imgs : [];
  const index = Math.min(Math.max(Number(node?.imageIndex) || 0, 0), Math.max(0, imgs.length - 1));
  // Prefer the connected source's dimensions for slider hit-testing. The
  // selected-file preview remains untouched and is only a geometry fallback.
  const visibleImages = getVisiblePreviewImages(node);
  const preferredImage = visibleImages[0] || (connected ? (state?.a || state?.b || null) : (imgs[index] || null));
  return getPreviewRectangles(node, preferredImage);
}

function isInsideImagePreview(node, event, canvas) {
  if (!node || !event) return false;
  const graphX = Number.isFinite(event.canvasX) ? event.canvasX : Number(canvas?.graph_mouse?.[0]);
  const graphY = Number.isFinite(event.canvasY) ? event.canvasY : Number(canvas?.graph_mouse?.[1]);
  if (!Number.isFinite(graphX) || !Number.isFinite(graphY)) return false;
  const localX = graphX - Number(node.pos?.[0] || 0);
  const localY = graphY - Number(node.pos?.[1] || 0);
  return currentPreviewRectangles(node).some((rect) => rectContains(rect, localX, localY));
}

function fallbackDomMenu(items, event, node) {
  activeFallbackMenuCleanup?.();
  const oldMenu = document.querySelector(".kiri-image-loader-context-menu");
  oldMenu?.remove();

  const menu = document.createElement("div");
  menu.className = "kiri-image-loader-context-menu";
  Object.assign(menu.style, {
    position: "fixed", zIndex: "100000", minWidth: "220px", maxWidth: "360px",
    maxHeight: "70vh", overflowY: "auto", padding: "5px", borderRadius: "8px",
    background: "#202020", color: "#eee", border: "1px solid #555",
    boxShadow: "0 5px 22px #0009", font: "13px sans-serif",
    left: `${Math.max(4, event.clientX || 0)}px`, top: `${Math.max(4, event.clientY || 0)}px`,
  });
  const heading = document.createElement("div");
  heading.textContent = "Kiri Image Loader";
  Object.assign(heading.style, { padding: "7px 9px", fontWeight: "600", borderBottom: "1px solid #444" });
  menu.appendChild(heading);

  for (const item of items) {
    if (item === null) {
      const divider = document.createElement("div");
      divider.style.borderTop = "1px solid #444";
      divider.style.margin = "4px 2px";
      menu.appendChild(divider);
      continue;
    }
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = item.content;
    button.disabled = !!item.disabled;
    Object.assign(button.style, {
      display: "block", width: "100%", textAlign: "left", border: "0", borderRadius: "4px",
      padding: "6px 9px", background: "transparent", color: item.disabled ? "#999" : "#eee",
      cursor: item.disabled ? "default" : "pointer", font: "inherit",
    });
    button.addEventListener("mouseenter", () => { if (!button.disabled) button.style.background = "#444"; });
    button.addEventListener("mouseleave", () => { button.style.background = "transparent"; });
    button.addEventListener("click", () => {
      item.callback?.(node);
      removeMenu();
    });
    menu.appendChild(button);
  }
  document.body.appendChild(menu);

  const cleanup = () => {
    document.removeEventListener("pointerdown", close, true);
    document.removeEventListener("keydown", keydown, true);
  };
  const removeMenu = () => {
    if (menu.isConnected) menu.remove();
    cleanup();
    if (activeFallbackMenuCleanup === cleanup) activeFallbackMenuCleanup = null;
  };
  const close = (click) => {
    if (!menu.contains(click.target)) removeMenu();
  };
  const keydown = (keyEvent) => {
    if (keyEvent.key === "Escape") removeMenu();
  };
  activeFallbackMenuCleanup = cleanup;
  document.addEventListener("pointerdown", close, true);
  document.addEventListener("keydown", keydown, true);
}

function showKiriMenu(node, event) {
  const items = menuItemsFor(node);
  const LiteGraph = window.LiteGraph;
  if (LiteGraph?.ContextMenu) {
    try {
      new LiteGraph.ContextMenu(items, { event, extra: node, title: "Kiri Image Loader" });
      return;
    } catch (error) {
      console.warn("[Kiri Image Loader] Native menu unavailable; using fallback.", error);
    }
  }
  fallbackDomMenu(items, event, node);
}

function patchCanvas(canvas) {
  if (!canvas) return;
  attachComparePointerHandler(canvas);

  if (!patchedDrawCanvases.has(canvas) && typeof canvas.drawNode === "function") {
    patchedDrawCanvases.add(canvas);
    const originalDrawNode = canvas.drawNode;
    canvas.drawNode = function (node, ctx, ...args) {
      if (isKiriNode(node)) {
        // This hook executes inside ComfyUI's hottest rendering path. Kiri's
        // optional preview/metadata helpers are strictly isolated: malformed,
        // incomplete, or mid-conversion subgraph state must never escape into
        // the host renderer or prevent the native node from drawing.
        try {
          if (!livePreviewByNode.has(node)) void restorePersistedLivePreview(node);
          syncLivePreviewFromUpstream(node);
          syncAllMetadataSources(node);
          installNativePreviewSource(node);
          patchNativePreviewWidgets(node);
          installPreviewInteractionSurface(node);
          installCompareWidgetPointer(node, this);
        } catch (error) {
          reportPreviewIntegrationError(node, error, "canvas integration");
        }
      }
      return originalDrawNode.call(this, node, ctx, ...args);
    };
  }

  if (patchedCanvases.has(canvas) || typeof canvas.processContextMenu !== "function") return;
  patchedCanvases.add(canvas);
  const originalProcessContextMenu = canvas.processContextMenu;

  canvas.processContextMenu = function (node, event) {
    if (isKiriNode(node)) {
      // Kiri's compact controls must remain reachable even before an image is
      // selected or connected; use the whole node context menu surface.
      showKiriMenu(node, event);
      return;
    }
    return originalProcessContextMenu.call(this, node, event);
  };
}

app.registerExtension({
  name: "Kiri.ImageLoader.MetadataClipboard",

  setup() {
    patchCanvas(app.canvas);
  },

  nodeCreated(node) {
    if (isKiriNode(node)) attachAutomaticReader(node);
    patchCanvas(app.canvas);
  },

  loadedGraphNode(node) {
    if (isKiriNode(node)) attachAutomaticReader(node);
    patchCanvas(app.canvas);
  },
});
