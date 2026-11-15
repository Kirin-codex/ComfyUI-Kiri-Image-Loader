/* Lightweight browser-stub regression tests for Kiri's upstream preview routing. */
const fs = require('fs');
const vm = require('vm');
const path = require('path');
const assert = require('assert');

const sourcePath = path.join(__dirname, '..', 'web', 'kiri_image_loader.js');
let source = fs.readFileSync(sourcePath, 'utf8').replace(/^import \{ app \} from ".*";\s*/m, 'const app = globalThis.__app;\n');
source += `\nglobalThis.__kiriTest = {
  installNativePreviewSource, patchNativePreviewWidgets, syncLivePreviewFromUpstream, livePreviewByNode,
  attachAutomaticReader,
  installCompareWidgetPointer, isNearComparisonHandle, comparisonHandlePoint,
  attachComparePointerHandler, compareDragByCanvas,
  connectedImageDescriptor, traceImageFileSource, descriptorFromOutputRecord, descriptorSignature,
  sourceSubmenuItems, setMetadataState, getMetadataState, menuItemsFor,
  canCopyImageSource, copyImageSourceToClipboard, pasteImageIntoLoadedImage,
  uploadClipboardImageToLoadedImage, isRenderableImageSource, getSourceNodePreviewImage, createComparisonCanvas,
  openEmbeddedWorkflow, isOpenableEmbeddedWorkflow, embeddedWorkflowTabName,
  metadataCopyItems, metadataEssentials, runtimeMetadataCopyItems, updateLivePreview, effectiveImageDescriptor,
  restorePersistedLivePreview, persistLivePreview, persistComparisonSlider, clearPersistedLivePreview,
  getVisiblePreviewMode, getVisiblePreviewImages, setVisiblePreviewMode,
  isSubgraphBoundaryNode, resolveSubgraphOutputSource, getConnectedSource, findInputLink,
  patchCanvas, syncAllMetadataSources,
};\n`;

const graphNodes = new Map();
const links = new Map();
const graph = {
  id: 'root-graph-uuid',
  _nodes: [],
  links,
  getNodeById(id) { return graphNodes.get(String(id)); },
  getLink(id) { return links.get(id); },
};
const app = {
  graph,
  canvas: null,
  nodeOutputs: {},
  nodePreviewImages: {},
  registerExtension() {},
};

function fakeCanvas() {
  const ctx = {
    fillRect() {}, save() {}, restore() {}, beginPath() {}, rect() {}, clip() {},
    drawImage() {}, moveTo() {}, lineTo() {}, stroke() {}, arc() {}, fill() {},
    putImageData() {}, getImageData(_x, _y, width, height) { return { data: new Uint8ClampedArray(width * height * 4) }; },
    set fillStyle(_v) {}, set strokeStyle(_v) {}, set lineWidth(_v) {},
  };
  return {
    width: 1,
    height: 1,
    getContext() { return ctx; },
  };
}

class FakeImage {
  constructor() { this.complete = false; this.naturalWidth = 0; this.naturalHeight = 0; this.width = 0; this.height = 0; }
  set src(value) { this._src = value; this.naturalWidth = 64; this.naturalHeight = 64; this.width = 64; this.height = 64; this.complete = true; this.onload?.(); }
  get src() { return this._src; }
}

const sandbox = {
  console,
  process,
  URLSearchParams,
  Map,
  WeakMap,
  WeakSet,
  Set,
  Array,
  Number,
  String,
  Object,
  Promise,
  Math,
  Image: FakeImage,
  document: { createElement(tag) { assert.strictEqual(tag, 'canvas'); return fakeCanvas(); } },
  window: { location: { href: 'http://localhost/' } },
  URL,
  fetch: async () => ({ ok: true, json: async () => ({ copy_items: [] }) }),
  navigator: {},
  setTimeout,
  queueMicrotask,
  globalThis: null,
  __app: app,
};
sandbox.globalThis = sandbox;
vm.runInNewContext(source, sandbox, { filename: sourcePath });
const api = sandbox.__kiriTest;

function makeKiriNode(id) {
  const node = {
    id,
    type: 'KiriImageLoader',
    comfyClass: 'KiriImageLoader',
    graph,
    inputs: [
      { name: 'image_in', link: null },
      { name: 'mask_in', link: null },
    ],
    outputs: [
      { name: 'A', type: 'IMAGE,MASK' },
      { name: 'B', type: 'IMAGE,MASK' },
      { name: 'C', type: 'IMAGE' },
    ],
    imgs: [{ tag: 'selected-file', naturalWidth: 100, naturalHeight: 70 }],
    widgets: [{ name: 'image', value: 'selected-file.png' }],
    getInputLink(index) {
      const id = this.inputs[index]?.link;
      return id == null ? null : graph.getLink(id);
    },
    setDirtyCanvas() {},
  };
  graphNodes.set(String(id), node);
  graph._nodes.push(node);
  return node;
}

const upstreamA = {
  id: 101,
  type: 'LoadImage',
  comfyClass: 'LoadImage',
  imgs: [{ tag: 'upstream-A', naturalWidth: 120, naturalHeight: 80 }],
  outputs: [{ type: 'IMAGE', name: 'IMAGE' }, { type: 'MASK', name: 'MASK' }],
};
graphNodes.set('101', upstreamA);
graph._nodes.push(upstreamA);

const upstreamB = {
  id: 102,
  type: 'CustomImageSource',
  imgs: [{ tag: 'upstream-B', naturalWidth: 120, naturalHeight: 80 }],
  outputs: [{ type: 'IMAGE', name: 'IMAGE' }],
};

// When a Kiri A/B socket is genuinely unconnected, its output defaults to the
// selected Loaded Image. Chained Kiri nodes must be able to preview that exact
// output instead of seeing an empty slot or the combined native canvas.
const unconnectedFallbackKiri = makeKiriNode(1200);
api.installNativePreviewSource(unconnectedFallbackKiri);
assert.strictEqual(api.getSourceNodePreviewImage(unconnectedFallbackKiri, 0).tag, 'selected-file',
  'Unconnected Kiri output A must expose the selected Loaded Image as its output fallback.');
assert.strictEqual(api.getSourceNodePreviewImage(unconnectedFallbackKiri, 1).tag, 'selected-file',
  'Unconnected Kiri output B must expose the selected Loaded Image as its output fallback.');

const downstreamFromA = makeKiriNode(1201);
links.set(50, { id: 50, origin_id: 1200, origin_slot: 0, target_id: 1201, target_slot: 0 });
downstreamFromA.inputs[0].link = 50;
assert.strictEqual(api.syncLivePreviewFromUpstream(downstreamFromA), true,
  'A downstream Kiri input A should receive a live preview from an upstream Kiri A fallback.');
assert.strictEqual(api.livePreviewByNode.get(downstreamFromA).a.tag, 'selected-file');

const downstreamFromB = makeKiriNode(1202);
links.set(51, { id: 51, origin_id: 1200, origin_slot: 1, target_id: 1202, target_slot: 1 });
downstreamFromB.inputs[1].link = 51;
assert.strictEqual(api.syncLivePreviewFromUpstream(downstreamFromB), true,
  'A downstream Kiri input B should receive a live preview from an upstream Kiri B fallback.');
assert.strictEqual(api.livePreviewByNode.get(downstreamFromB).b.tag, 'selected-file');

// Conversely, an upstream Kiri input that IS connected but has no live image
// must not leak its own selected-file preview into that connected output.
const connectedButUnrenderedKiri = makeKiriNode(1203);
links.set(52, { id: 52, origin_id: 101, origin_slot: 0, target_id: 1203, target_slot: 0 });
connectedButUnrenderedKiri.inputs[0].link = 52;
assert.strictEqual(api.getSourceNodePreviewImage(connectedButUnrenderedKiri, 0), null,
  'Connected-but-unrendered upstream Kiri branch must not substitute its Loaded Image.');
graphNodes.set('102', upstreamB);
graph._nodes.push(upstreamB);

const kiri = makeKiriNode(999);
assert.strictEqual(api.getSourceNodePreviewImage(kiri, 2).tag, 'selected-file', 'Output C preview must be the selected Loaded Image.');
assert.strictEqual(api.traceImageFileSource(kiri, 2).filename, 'selected-file.png', 'Output C metadata provenance must point to the selected file.');
links.set(1, { id: 1, origin_id: 101, origin_slot: 0, target_id: 999, target_slot: 0 });
kiri.inputs[0].link = 1;
api.installNativePreviewSource(kiri);
assert.strictEqual(api.syncLivePreviewFromUpstream(kiri), true, 'A source should resolve immediately');
assert.strictEqual(kiri.imgs[0], upstreamA.imgs[0], 'Connected A should replace the selected file in the native preview');

// A change to the upstream Load Image selector is visible on the next graph redraw.
const refreshedA = { tag: 'upstream-A-refreshed', naturalWidth: 120, naturalHeight: 80 };
upstreamA.imgs = [refreshedA];
assert.strictEqual(api.syncLivePreviewFromUpstream(kiri), true, 'Changed upstream image should refresh Kiri');
assert.strictEqual(kiri.imgs[0], refreshedA);

// Selecting another local file must update only the fallback, not the visible connected preview.
kiri.imgs = [{ tag: 'new-selected-file', naturalWidth: 100, naturalHeight: 70 }];
kiri.widgets[0].value = 'new-selected-file.png';
assert.strictEqual(kiri.imgs[0], upstreamA.imgs[0], 'Connected preview must remain visible after file selector updates');

// Add B and ensure the native preview now receives a comparison canvas.
links.set(2, { id: 2, origin_id: 102, origin_slot: 0, target_id: 999, target_slot: 1 });
kiri.inputs[1].link = 2;
assert.strictEqual(api.syncLivePreviewFromUpstream(kiri), true, 'B source should update comparison state');
assert.notStrictEqual(kiri.imgs[0], upstreamA.imgs[0], 'Two connected sources should produce a composite canvas');
assert.strictEqual(kiri.imgs[0].width, 120);
assert.strictEqual(kiri.imgs[0].height, 80);
assert.strictEqual(api.getSourceNodePreviewImage(kiri, 2).tag, 'new-selected-file', 'C must remain the current Loaded Image even while A and B are connected.');
assert.strictEqual(api.traceImageFileSource(kiri, 2).filename, 'new-selected-file.png');

// The built-in Load Image MASK socket derives the familiar inverted-alpha display.
links.set(3, { id: 3, origin_id: 101, origin_slot: 1, target_id: 999, target_slot: 1 });
kiri.inputs[1].link = 3;
assert.strictEqual(api.syncLivePreviewFromUpstream(kiri), true, 'LoadImage MASK output should refresh the comparison');
assert.notStrictEqual(api.livePreviewByNode.get(kiri).b, refreshedA, 'Mask source should be converted to its alpha preview');

// The native ImagePreviewWidget must receive that same live image through its options.
function ImagePreviewWidget() {}
const widget = new ImagePreviewWidget();
let seenOptions = null;
widget.drawWidget = function (_ctx, options) { seenOptions = options; };
kiri.widgets.push(widget);
api.patchNativePreviewWidgets(kiri);
widget.drawWidget({}, { previewImages: [{ tag: 'old-selected' }], width: 300 });
assert.strictEqual(seenOptions.previewImages[0], kiri.imgs[0], 'Modern native preview widget should get Kiri live image');

// Disconnecting returns to the most recently selected file preview.
kiri.inputs[0].link = null;
kiri.inputs[1].link = null;
api.livePreviewByNode.delete(kiri);
assert.strictEqual(kiri.imgs[0].tag, 'new-selected-file', 'Selected-file fallback should return after disconnect');

// Source-aware metadata resolution follows a connected Load Image file.
const loadImageWithFile = {
  id: 103,
  type: 'LoadImage',
  comfyClass: 'LoadImage',
  widgets: [{ name: 'image', value: 'uploaded/subfolder/original.png' }],
  inputs: [],
  outputs: [{ name: 'IMAGE', type: 'IMAGE' }, { name: 'MASK', type: 'MASK' }],
};
graphNodes.set('103', loadImageWithFile);
graph._nodes.push(loadImageWithFile);
links.set(4, { id: 4, origin_id: 103, origin_slot: 0, target_id: 999, target_slot: 0 });
kiri.inputs[0].link = 4;
let sourceDescriptor = api.connectedImageDescriptor(kiri, 'image_a');
assert.strictEqual(sourceDescriptor.filename, 'uploaded/subfolder/original.png');
assert.strictEqual(sourceDescriptor.type, 'input');
assert.strictEqual(sourceDescriptor.originNode, loadImageWithFile);

// A single-image transform is traced back to its unambiguous Load Image source.
const processor = {
  id: 104,
  type: 'SimpleImageTransform',
  inputs: [{ name: 'image', type: 'IMAGE', link: 5 }],
  outputs: [{ name: 'IMAGE', type: 'IMAGE' }],
  getInputLink(index) { return graph.getLink(this.inputs[index]?.link); },
};
graphNodes.set('104', processor);
graph._nodes.push(processor);
links.set(5, { id: 5, origin_id: 103, origin_slot: 0, target_id: 104, target_slot: 0 });
links.set(6, { id: 6, origin_id: 104, origin_slot: 0, target_id: 999, target_slot: 0 });
kiri.inputs[0].link = 6;
sourceDescriptor = api.connectedImageDescriptor(kiri, 'image_a');
assert.strictEqual(sourceDescriptor.filename, 'uploaded/subfolder/original.png');
assert.strictEqual(sourceDescriptor.tracedThrough, true);
assert.strictEqual(api.canCopyImageSource(kiri, 'image_a'), false,
  'A transformed source with no output pixels must not offer Copy of its different ancestor image.');
// Direct Load Image sources can copy their actual file even before a preview is evaluated.
kiri.inputs[0].link = 4;
sourceDescriptor = api.connectedImageDescriptor(kiri, 'image_a');
assert.strictEqual(sourceDescriptor.tracedThrough, undefined);
assert.strictEqual(api.canCopyImageSource(kiri, 'image_a'), true);

// Saved preview records are also traceable to ComfyUI output-directory files.
const outputDescriptor = api.descriptorFromOutputRecord({
  url: '/view?filename=final.png&type=output&subfolder=job_7'
}, processor);
assert.strictEqual(outputDescriptor.filename, 'final.png');
assert.strictEqual(outputDescriptor.type, 'output');
assert.strictEqual(outputDescriptor.subfolder, 'job_7');
const externalOutputDescriptor = api.descriptorFromOutputRecord({
  filename: 'Kirin Image 654321.png', type: 'kiri_external', path_token: 'opaque-path-token'
}, processor);
assert.strictEqual(externalOutputDescriptor.type, 'kiri_external');
assert.strictEqual(externalOutputDescriptor.pathToken, 'opaque-path-token');

// Each source gets its own submenu and its own file/metadata actions.
const sourceSignature = api.descriptorSignature(sourceDescriptor);
api.setMetadataState(kiri, 'image_a', {
  sourceSignature,
  filename: sourceDescriptor.filename,
  embedded_workflow: { nodes: [{ id: 1, type: 'LoadImage' }], links: [] },
  editable_metadata: [{ keyword: 'prompt', value: 'before edit', format: 'tEXt' }],
  metadata_editable: true,
  copy_items: [{ category: 'Prompt', label: 'Positive Prompt', value: 'a test prompt' }],
});
const sourceMenu = api.sourceSubmenuItems(kiri, 'image_a', 'Image A');
assert.strictEqual(sourceMenu[0].content, 'Open Embedded Workflow');
assert.strictEqual(sourceMenu[2].content, 'Open');
assert.strictEqual(sourceMenu[3].content, 'Open File Location');
assert.strictEqual(sourceMenu[4].content, 'Delete Image (Recycle Bin)');
assert.ok(sourceMenu.some((item) => item?.content === 'Edit Metadata…' && !item.disabled));
assert.ok(sourceMenu.some((item) => item?.content === 'Copy Positive Prompt'));
assert.ok(sourceMenu.some((item) => item?.content === 'Copy'), 'Image A submenu must expose a pixel-copy action.');
assert.ok(sourceMenu.find((item) => item?.content === 'Copy')?.disabled === false, 'A file-backed Image A source should be copyable.');

// The main Kiri context menu offers three independently-targeted source submenus.
kiri.widgets = [{ name: 'image', value: 'selected/fallback.png' }];
const loadedDescriptor = { filename: 'selected/fallback.png', type: 'input', subfolder: '', originNode: kiri };
api.setMetadataState(kiri, 'loaded', {
  sourceSignature: api.descriptorSignature(loadedDescriptor),
  filename: loadedDescriptor.filename,
  copy_items: [{ category: 'Image', label: 'Seed', value: '1234' }],
});
api.setMetadataState(kiri, 'image_a', {
  sourceSignature: api.descriptorSignature(sourceDescriptor),
  filename: sourceDescriptor.filename,
  embedded_workflow: { nodes: [{ id: 1, type: 'LoadImage' }], links: [] },
  copy_items: [{ category: 'Prompt', label: 'Positive Prompt', value: 'source A prompt' }],
});
links.set(7, { id: 7, origin_id: 103, origin_slot: 1, target_id: 999, target_slot: 1 });
kiri.inputs[1].link = 7;
const sourceBDescriptor = api.connectedImageDescriptor(kiri, 'image_b');
api.setMetadataState(kiri, 'image_b', {
  sourceSignature: api.descriptorSignature(sourceBDescriptor),
  filename: sourceBDescriptor.filename,
  copy_items: [{ category: 'Generation', label: 'Steps', value: '28' }],
});
const mainMenu = api.menuItemsFor(kiri);
assert.strictEqual(mainMenu[0].content, 'Embed Complete Workflow: OFF', 'Embedding remains reachable from the root menu with no image selected.');
assert.strictEqual(mainMenu[2].content, 'Visible Preview Image');
assert.strictEqual(mainMenu[4].content, 'Image A');
assert.strictEqual(mainMenu[5].content, 'Image B');
assert.strictEqual(mainMenu[7].content, 'Loaded Image');
assert.ok(mainMenu[4].submenu.options.some((item) => item?.content === 'Copy Positive Prompt'));
assert.ok(mainMenu[4].submenu.options.some((item) => item?.content === 'Open Embedded Workflow'));
assert.ok(mainMenu[4].submenu.options.some((item) => item?.content === 'Copy' && !item.disabled));
assert.ok(mainMenu[5].submenu.options.some((item) => item?.content === 'Copy' && !item.disabled));
assert.ok(mainMenu[5].submenu.options.some((item) => item?.content === 'Copy KSampler Settings'));
assert.ok(!mainMenu[5].submenu.options.some((item) => item?.content === 'Copy Steps'), 'Individual metadata fields no longer create a right-click action each.');
assert.ok(!mainMenu[5].submenu.options.some((item) => item?.content === 'Open Embedded Workflow'), 'Workflow action must remain source-specific');
assert.ok(mainMenu[7].submenu.options.some((item) => item?.content === 'Copy Seed(s)'));
assert.ok(mainMenu[7].submenu.options.some((item) => item?.content === 'Copy' && !item.disabled));
assert.ok(mainMenu[7].submenu.options.some((item) => item?.content === 'Paste'), 'Loaded Image submenu must provide a clipboard paste action.');
assert.ok(!mainMenu[7].submenu.options.some((item) => item?.content === 'Open Embedded Workflow'), 'Loaded Image must not inherit a connected source workflow');

// Only prompt/seed/KSampler copy commands remain; the rest is available as read-only metadata.
const compactActions = api.metadataCopyItems({ copy_items: [
  { category: 'Prompts', label: 'Positive Prompt', value: 'positive' },
  { category: 'Prompts', label: 'Negative Prompt', value: 'negative' },
  { category: 'Generation', label: 'Seed', value: '42' },
  { category: 'Generation', label: 'Steps', value: '28' },
  { category: 'Models', label: 'Checkpoint', value: 'model.safetensors' },
  { category: 'LoRAs', label: 'LoRA · style', value: 'style.safetensors' },
] });
const copyActions = compactActions.filter((item) => item?.callback && item.content.startsWith('Copy '));
assert.strictEqual(JSON.stringify(Array.from(copyActions, (item) => item.content)), JSON.stringify([
  'Copy Positive Prompt', 'Copy Negative Prompt', 'Copy Seed(s)', 'Copy KSampler Settings'
]));

const multilineRuntimeItems = api.runtimeMetadataCopyItems({ graph: {
  prompts: { positive: 'positive from multiline string', negative: 'negative from multiline string' },
  text_encoders_and_prompts: [],
  samplers: [{ settings: { seed: 555, steps: 24, cfg: 6 } }],
} });
const multilineEssentials = api.metadataEssentials({ runtime_metadata: { graph: {
  prompts: { positive: 'positive from multiline string', negative: 'negative from multiline string' },
  samplers: [{ settings: { seed: 555, steps: 24, cfg: 6 } }],
} } });
assert.strictEqual(multilineEssentials.positiveItem.value, 'positive from multiline string');
assert.strictEqual(multilineEssentials.negativeItem.value, 'negative from multiline string');
assert.ok(multilineRuntimeItems.some((item) => item.label === 'Positive Prompt' && item.value === 'positive from multiline string'));
assert.ok(multilineRuntimeItems.some((item) => item.label === 'Negative Prompt' && item.value === 'negative from multiline string'));
const emptyPromptMenu = api.metadataCopyItems({ copy_items: [] });
assert.ok(emptyPromptMenu.some((item) => item?.content === 'Copy Positive Prompt' && item.disabled));
assert.ok(emptyPromptMenu.some((item) => item?.content === 'Copy Negative Prompt' && item.disabled));
assert.ok(compactActions.some((item) => item?.content === 'View All Metadata…'));

// A just-decoded generated A is mapped to Kiri's own saved output; B keeps its own runtime graph metadata.
const generatedA = { id: 200, type: 'VAEDecode', inputs: [{ name: 'samples', type: 'LATENT', link: null }], outputs: [{ name: 'IMAGE', type: 'IMAGE' }] };
const generatedB = { id: 201, type: 'VAEDecode', inputs: [{ name: 'samples', type: 'LATENT', link: null }], outputs: [{ name: 'IMAGE', type: 'IMAGE' }] };
graphNodes.set('200', generatedA); graphNodes.set('201', generatedB);
graph._nodes.push(generatedA, generatedB);
links.set(70, { id: 70, origin_id: 200, origin_slot: 0, target_id: 999, target_slot: 0 });
links.set(71, { id: 71, origin_id: 201, origin_slot: 0, target_id: 999, target_slot: 1 });
kiri.inputs[0].link = 70; kiri.inputs[1].link = 71;
const generatedRecordA = { graph: { text_encoders_and_prompts: [{ branch: 'positive', values: { text: 'generated A prompt' } }], samplers: [{ settings: { seed: 123, steps: 30, cfg: 6 } }] } };
const generatedRecordB = { graph: { text_encoders_and_prompts: [{ branch: 'positive', values: { text: 'generated B prompt' } }], samplers: [{ settings: { seed: 456, steps: 24 } }] } };
(async () => {
  await api.updateLivePreview(kiri, {
    kiri_preview_mode: ['compare'],
    kiri_preview_a: ['dGVzdA=='],
    kiri_preview_b: ['dGVzdEI='],
    kiri_saved_files: [[{ filename: 'Kiri_generated.png', subfolder: 'Kiri', type: 'output' }]],
    kiri_runtime_metadata_a: [JSON.stringify(generatedRecordA)],
    kiri_runtime_metadata_b: [JSON.stringify(generatedRecordB)],
  });
  const generatedDescriptorA = api.effectiveImageDescriptor(kiri, 'image_a');
  assert.strictEqual(generatedDescriptorA.filename, 'Kiri_generated.png');
  assert.strictEqual(generatedDescriptorA.type, 'output');
  const generatedMetadataB = api.getMetadataState(kiri, 'image_b');
  assert.ok(generatedMetadataB?.runtime);
  assert.ok(generatedMetadataB.copy_items.some((item) => item.value === 'generated B prompt'));
  assert.ok(api.metadataCopyItems(generatedMetadataB).some((item) => item?.content === 'Copy Seed(s)'));
  const previewState = api.livePreviewByNode.get(kiri);
  assert.strictEqual(previewState?.mode, 'compare');
  assert.strictEqual(previewState?.a?.naturalWidth, 64, 'VAE Decode execution payload is rendered in the Kiri preview.');
  assert.strictEqual(previewState?.b?.naturalWidth, 64, 'The B-side execution payload is rendered alongside A.');
  assert.ok(kiri.properties?.kiri_live_preview?.a?.data === 'dGVzdA==',
    'The latest generated preview payload is saved in serialized node properties.');

  // Simulate ComfyUI switching away from a workflow and rebuilding its node
  // objects from serialized workflow JSON. The comparison must return without
  // requiring a new execution, and only for the same A/B source connections.
  const restoredPreviewNode = makeKiriNode(1002);
  links.set(72, { id: 72, origin_id: 200, origin_slot: 0, target_id: 1002, target_slot: 0 });
  links.set(73, { id: 73, origin_id: 201, origin_slot: 0, target_id: 1002, target_slot: 1 });
  restoredPreviewNode.inputs[0].link = 72;
  restoredPreviewNode.inputs[1].link = 73;
  restoredPreviewNode.properties = JSON.parse(JSON.stringify(kiri.properties));
  api.installNativePreviewSource(restoredPreviewNode);
  assert.strictEqual(await api.restorePersistedLivePreview(restoredPreviewNode), true,
    'A saved comparison should restore after a workflow tab/node reconstruction.');
  const restoredState = api.livePreviewByNode.get(restoredPreviewNode);
  assert.strictEqual(restoredState?.mode, 'compare');
  assert.strictEqual(restoredState?.a?.naturalWidth, 64);
  assert.strictEqual(restoredState?.b?.naturalWidth, 64);
  assert.ok(restoredState?.renderedImage, 'Restoration rebuilds the live comparison canvas.');
  restoredState.slider = 0.73;
  api.persistComparisonSlider(restoredPreviewNode, restoredState);
  assert.strictEqual(restoredPreviewNode.properties.kiri_live_preview.slider, 0.73,
    'Dragging the wipe divider updates the serialized preview position.');

  // Exercise the real node onConfigure wrapper installed by the extension.
  const lifecycleRestoredNode = makeKiriNode(1004);
  links.set(76, { id: 76, origin_id: 200, origin_slot: 0, target_id: 1004, target_slot: 0 });
  links.set(77, { id: 77, origin_id: 201, origin_slot: 0, target_id: 1004, target_slot: 1 });
  lifecycleRestoredNode.inputs[0].link = 76;
  lifecycleRestoredNode.inputs[1].link = 77;
  lifecycleRestoredNode.properties = JSON.parse(JSON.stringify(kiri.properties));
  api.attachAutomaticReader(lifecycleRestoredNode);
  lifecycleRestoredNode.onConfigure?.({});
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.strictEqual(api.livePreviewByNode.get(lifecycleRestoredNode)?.mode, 'compare',
    'The node onConfigure lifecycle automatically restores the saved comparison.');

  const mismatchedPreviewNode = makeKiriNode(1003);
  links.set(74, { id: 74, origin_id: 200, origin_slot: 0, target_id: 1003, target_slot: 0 });
  links.set(75, { id: 75, origin_id: 101, origin_slot: 0, target_id: 1003, target_slot: 1 });
  mismatchedPreviewNode.inputs[0].link = 74;
  mismatchedPreviewNode.inputs[1].link = 75;
  mismatchedPreviewNode.properties = JSON.parse(JSON.stringify(kiri.properties));
  assert.strictEqual(await api.restorePersistedLivePreview(mismatchedPreviewNode), false,
    'A saved comparison must not restore when either source connection has changed.');
  assert.strictEqual(api.livePreviewByNode.has(mismatchedPreviewNode), false);
  assert.strictEqual(mismatchedPreviewNode.properties.kiri_live_preview, undefined,
    'A stale snapshot is removed after a changed source connection is detected.');
  console.log('Kiri frontend preview runtime tests passed.');
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});


// Empty / bypassed upstream sources must clear stale comparison pixels rather
// than drawing invalid dimensions or keeping a previous execution's preview.
const emptyUpstream = {
  id: 105,
  type: 'VAE Decode',
  imgs: [],
  outputs: [{ name: 'IMAGE', type: 'IMAGE' }],
};
graphNodes.set('105', emptyUpstream);
graph._nodes.push(emptyUpstream);
const emptyKiri = makeKiriNode(1001);
links.set(20, { id: 20, origin_id: 105, origin_slot: 0, target_id: 1001, target_slot: 0 });
emptyKiri.inputs[0].link = 20;
api.installNativePreviewSource(emptyKiri);
api.livePreviewByNode.set(emptyKiri, {
  source: 'graph', mode: 'single', a: refreshedA, b: null, renderedImage: refreshedA, slider: 0.5,
});
assert.strictEqual(api.syncLivePreviewFromUpstream(emptyKiri), false, 'An empty upstream decode should not create a new preview.');
assert.strictEqual(api.livePreviewByNode.has(emptyKiri), false, 'A stale graph preview must be cleared when all connected sources are empty.');
assert.strictEqual(emptyKiri.imgs[0].tag, 'selected-file', 'Auto preview should use Loaded Image when the connected upstream branch has no preview.');
assert.strictEqual(api.isRenderableImageSource({ width: 0, height: 64 }), false);
assert.strictEqual(api.isRenderableImageSource({ width: 64, height: 64 }), true);
assert.strictEqual(api.createComparisonCanvas(refreshedA, { width: 0, height: 0 }), refreshedA, 'Invalid B must degrade to valid A instead of breaking the preview.');

// A chained Kiri node's combined canvas preview must never masquerade as an
// exact A/B output when that output is empty or has not executed yet.
const chainedEmpty = makeKiriNode(1002);
// Keep its selected-file fallback intact; the stale combined canvas lives only in livePreviewByNode.
api.livePreviewByNode.set(chainedEmpty, {
  source: 'graph', mode: 'single', a: null, b: null,
  renderedImage: refreshedA, slider: 0.5,
});
assert.strictEqual(api.getSourceNodePreviewImage(chainedEmpty, 0).tag, 'selected-file', 'Unconnected A output defaults to the selected file.');
assert.strictEqual(api.getSourceNodePreviewImage(chainedEmpty, 1).tag, 'selected-file', 'Unconnected B output defaults to the selected file.');
assert.strictEqual(api.getSourceNodePreviewImage(chainedEmpty, 2).tag, 'selected-file', 'Chained Kiri output C exposes Loaded Image independently of its empty A/B previews.');
const secondEmptyDownstream = makeKiriNode(1003);
const uninitializedDecode = {
  id: 106, type: 'VAEDecode', imgs: [], outputs: [{ type: 'IMAGE', name: 'IMAGE' }],
};
graphNodes.set('106', uninitializedDecode);
graph._nodes.push(uninitializedDecode);
links.set(21, { id: 21, origin_id: 1002, origin_slot: 0, target_id: 1003, target_slot: 0 });
links.set(22, { id: 22, origin_id: 106, origin_slot: 0, target_id: 1003, target_slot: 1 });
secondEmptyDownstream.inputs[0].link = 21;
secondEmptyDownstream.inputs[1].link = 22;
api.installNativePreviewSource(secondEmptyDownstream);
api.livePreviewByNode.set(secondEmptyDownstream, {
  source: 'graph', mode: 'compare', a: refreshedA, b: refreshedA,
  renderedImage: refreshedA, slider: 0.5,
});
assert.strictEqual(api.syncLivePreviewFromUpstream(secondEmptyDownstream), true);
assert.strictEqual(api.livePreviewByNode.get(secondEmptyDownstream).mode, 'single');
assert.strictEqual(api.livePreviewByNode.get(secondEmptyDownstream).a.tag, 'selected-file', "Upstream Kiri A fallback must override the downstream node's own Loaded Image.");
assert.strictEqual(api.livePreviewByNode.get(secondEmptyDownstream).b, null, 'The empty VAE Decode branch remains unavailable rather than substituting a stale image.');

// Mixed graphs must degrade to a single preview when either A or B has no
// usable source, without trying to compare an invalid canvas.
const mixedAValid = makeKiriNode(1004);
links.set(23, { id: 23, origin_id: 101, origin_slot: 0, target_id: 1004, target_slot: 0 });
links.set(24, { id: 24, origin_id: 106, origin_slot: 0, target_id: 1004, target_slot: 1 });
mixedAValid.inputs[0].link = 23;
mixedAValid.inputs[1].link = 24;
assert.strictEqual(api.syncLivePreviewFromUpstream(mixedAValid), true);
const mixedAState = api.livePreviewByNode.get(mixedAValid);
assert.strictEqual(mixedAState.mode, 'single');
assert.strictEqual(mixedAState.a, upstreamA.imgs[0]);
assert.strictEqual(mixedAState.b, null);
assert.strictEqual(mixedAState.renderedImage, upstreamA.imgs[0]);

const mixedBValid = makeKiriNode(1005);
links.set(25, { id: 25, origin_id: 1002, origin_slot: 0, target_id: 1005, target_slot: 0 });
links.set(26, { id: 26, origin_id: 102, origin_slot: 0, target_id: 1005, target_slot: 1 });
mixedBValid.inputs[0].link = 25;
mixedBValid.inputs[1].link = 26;
assert.strictEqual(api.syncLivePreviewFromUpstream(mixedBValid), true);
const mixedBState = api.livePreviewByNode.get(mixedBValid);
assert.strictEqual(mixedBState.mode, 'compare', 'An unconnected upstream Kiri A defaults to its Loaded Image, so both branches are previewable.');
assert.strictEqual(mixedBState.a.tag, 'selected-file');
assert.strictEqual(mixedBState.b, upstreamB.imgs[0]);
assert.ok(mixedBState.renderedImage, 'Both a Kiri loaded-file fallback and a live upstream image should produce a comparison canvas.');


// Subgraph boundary regression: ComfyUI SubgraphNode.getInputLink(slot) resolves
// an INTERNAL output link (not the host's external input). Kiri must read the
// parent-graph input slot link directly and follow the selected subgraph output
// through resolveSubgraphOutputLink(), including nested subgraphs.
const innerGraph = {
  id: 'inner-graph-uuid',
  _nodes: [],
  getNodeById(id) { return this._nodes.find((node) => String(node.id) === String(id)) || null; },
  getLink() { return null; },
  links: new Map(),
};
const innerImageA = { tag: 'subgraph-output-A', width: 120, height: 80, naturalWidth: 120, naturalHeight: 80 };
const innerImageB = { tag: 'subgraph-output-B', width: 96, height: 72, naturalWidth: 96, naturalHeight: 72 };
const innerLoadA = {
  id: 1, graph: innerGraph, type: 'LoadImage', comfyClass: 'LoadImage',
  widgets: [{ name: 'image', value: 'subgraph-A.png' }], imgs: [innerImageA],
  inputs: [], outputs: [{ name: 'IMAGE', type: 'IMAGE' }],
};
const innerLoadB = {
  id: 2, graph: innerGraph, type: 'LoadImage', comfyClass: 'LoadImage',
  widgets: [{ name: 'image', value: 'subgraph-B.png' }], imgs: [innerImageB],
  inputs: [], outputs: [{ name: 'IMAGE', type: 'IMAGE' }],
};
innerGraph._nodes.push(innerLoadA, innerLoadB);
const subgraphDefinition = { id: 'subgraph-definition-uuid', outputNode: { slots: [{}, {}] } };
const subgraphHost = {
  id: 1100, type: 'subgraph-definition-uuid', constructor: { name: 'SubgraphNode' },
  graph, subgraph: subgraphDefinition,
  inputs: [
    { name: 'pixels', type: 'IMAGE', link: 27 },
    { name: 'vae', type: 'VAE', link: null },
    { name: 'latent', type: 'LATENT', link: null },
    { name: 'extra', type: 'IMAGE', link: null },
  ],
  outputs: [{ name: 'decoded', type: 'IMAGE' }, { name: 'switched', type: 'IMAGE' }],
  // This intentionally emulates the current ComfyUI SubgraphNode override:
  // it is NOT a normal external input-link resolver and throws for a missing
  // matching output slot. Kiri must never call it for the host's inputs.
  getInputLink(slot) {
    const outputSlot = this.subgraph.outputNode.slots[slot];
    if (!outputSlot) throw new RangeError('subgraph internal output slot out of range');
    return outputSlot.innerLink || null;
  },
  resolveSubgraphOutputLink(slot) {
    const outputNode = slot === 0 ? innerLoadA : slot === 1 ? innerLoadB : null;
    if (!outputNode) return undefined;
    return { outputNode, output: outputNode.outputs[0], link: { origin_slot: 0 } };
  },
};
graphNodes.set('1100', subgraphHost);
graph._nodes.push(subgraphHost);
links.set(27, { id: 27, origin_id: 101, origin_slot: 0, target_id: 1100, target_slot: 0 });
assert.strictEqual(api.isSubgraphBoundaryNode(subgraphHost), true);
assert.strictEqual(api.getConnectedSource(subgraphHost, 'pixels').node, upstreamA,
  'External SubgraphNode inputs must resolve from parent graph link IDs without invoking its output-oriented getInputLink override.');
assert.strictEqual(api.getSourceNodePreviewImage(subgraphHost, 0), innerImageA,
  'Subgraph output slot 0 must resolve its exact interior image output.');
assert.strictEqual(api.getSourceNodePreviewImage(subgraphHost, 1), innerImageB,
  'Subgraph output slot 1 must not accidentally reuse output slot 0.');
const subgraphMetadataConsumer = makeKiriNode(1104);
links.set(29, { id: 29, origin_id: 1100, origin_slot: 0, target_id: 1104, target_slot: 0 });
subgraphMetadataConsumer.inputs[0].link = 29;
assert.strictEqual(api.connectedImageDescriptor(subgraphMetadataConsumer, 'image_a').filename, 'subgraph-A.png',
  'Metadata tracing must follow the exact image file source inside a subgraph.');

const subgraphDownstream = makeKiriNode(1101);
links.set(28, { id: 28, origin_id: 1100, origin_slot: 1, target_id: 1101, target_slot: 1 });
subgraphDownstream.inputs[1].link = 28;
assert.strictEqual(api.syncLivePreviewFromUpstream(subgraphDownstream), true);
assert.strictEqual(api.livePreviewByNode.get(subgraphDownstream).b, innerImageB,
  'A Kiri B input connected to a subgraph output must resolve the correct interior image preview.');
assert.strictEqual(api.connectedImageDescriptor(subgraphDownstream, 'image_b').filename, 'subgraph-B.png');

// Nested subgraph outputs resolve recursively with graph-scoped node identity.
const deepGraph = { id: 'deep-graph-uuid', _nodes: [] };
const deepImage = { tag: 'nested-subgraph-preview', width: 40, height: 30, naturalWidth: 40, naturalHeight: 30 };
const deepLoad = {
  id: 1, graph: deepGraph, type: 'LoadImage', comfyClass: 'LoadImage',
  widgets: [{ name: 'image', value: 'deep-source.png' }], imgs: [deepImage],
  inputs: [], outputs: [{ name: 'IMAGE', type: 'IMAGE' }],
};
const nestedDefinition = { id: 'nested-subgraph-definition', outputNode: { slots: [{}] } };
const nestedHost = {
  id: 3, graph: innerGraph, subgraph: nestedDefinition,
  inputs: [], outputs: [{ type: 'IMAGE' }],
  resolveSubgraphOutputLink(slot) {
    return slot === 0 ? { outputNode: deepLoad, output: deepLoad.outputs[0], link: { origin_slot: 0 } } : undefined;
  },
};
const outerNestedHost = {
  id: 1102, graph, subgraph: { id: 'outer-nested-definition' },
  inputs: [], outputs: [{ type: 'IMAGE' }],
  resolveSubgraphOutputLink(slot) {
    return slot === 0 ? { outputNode: nestedHost, output: nestedHost.outputs[0], link: { origin_slot: 0 } } : undefined;
  },
};
assert.strictEqual(api.getSourceNodePreviewImage(outerNestedHost, 0), deepImage,
  'Nested subgraph preview resolution must recurse to the actual interior image.');
assert.strictEqual(api.traceImageFileSource(outerNestedHost, 0).filename, 'deep-source.png');

// Preview-only comparison canvases are bounded for large tiled/upscaled images.
const hugeA = { width: 8192, height: 8192, naturalWidth: 8192, naturalHeight: 8192 };
const hugeB = { width: 8192, height: 8192, naturalWidth: 8192, naturalHeight: 8192 };
const safeCompare = api.createComparisonCanvas(hugeA, hugeB, 0.5);
assert.ok(safeCompare.width * safeCompare.height <= 2_000_000,
  'A preview comparison canvas must not duplicate huge source images at full resolution.');
assert.ok(Math.max(safeCompare.width, safeCompare.height) <= 2048);

// An exception in Kiri's optional renderer integration must never suppress the
// native ComfyUI draw function.
const brokenWidgetNode = {
  id: 1103, type: 'KiriImageLoader', comfyClass: 'KiriImageLoader', graph,
  inputs: [], outputs: [], imgs: [],
  get widgets() { throw new Error('simulated frontend proxy failure'); },
  getWidgetOnPos() { return null; },
  setDirtyCanvas() {},
};
let nativeDrawCalls = 0;
const guardedCanvas = {
  drawNode() { nativeDrawCalls += 1; },
  processContextMenu() {},
};
api.patchCanvas(guardedCanvas);
assert.doesNotThrow(() => guardedCanvas.drawNode(brokenWidgetNode, {}));
assert.strictEqual(nativeDrawCalls, 1, 'ComfyUI native rendering must continue even if a Kiri preview helper encounters a bad proxy.');


// Visible preview source selection and final Loaded Image fallback.
const selectablePreviewNode = makeKiriNode(1105);
const loadedFallbackPreview = { tag: 'selected-loaded-image', width: 88, height: 66, naturalWidth: 88, naturalHeight: 66 };
selectablePreviewNode.imgs = [loadedFallbackPreview];
let serializedPreviewSettings = '{}';
selectablePreviewNode.widgets.push({
  name: 'metadata_export_settings',
  get value() { return serializedPreviewSettings; },
  set value(value) { serializedPreviewSettings = value; },
  callback() {},
});
links.set(30, { id: 30, origin_id: 101, origin_slot: 0, target_id: 1105, target_slot: 0 });
links.set(31, { id: 31, origin_id: 102, origin_slot: 0, target_id: 1105, target_slot: 1 });
selectablePreviewNode.inputs[0].link = 30;
selectablePreviewNode.inputs[1].link = 31;
api.installNativePreviewSource(selectablePreviewNode);
assert.strictEqual(api.getVisiblePreviewMode(selectablePreviewNode), 'auto', 'Auto/Compare is the default preview mode for older workflows.');
assert.strictEqual(api.syncLivePreviewFromUpstream(selectablePreviewNode), true);
assert.notStrictEqual(selectablePreviewNode.imgs[0], loadedFallbackPreview, 'Connected A/B preview takes priority over Loaded Image in Auto mode.');
assert.strictEqual(selectablePreviewNode.imgs[0].width, 120, 'Auto mode displays the A/B comparison canvas.');

api.setVisiblePreviewMode(selectablePreviewNode, 'image_a');
assert.strictEqual(selectablePreviewNode.imgs[0], upstreamA.imgs[0], 'Image A mode displays only A.');
assert.notStrictEqual(selectablePreviewNode.imgs[0], upstreamB.imgs[0], 'Image A mode must ignore B when both are connected.');
assert.strictEqual(api.isNearComparisonHandle(selectablePreviewNode, api.livePreviewByNode.get(selectablePreviewNode), [0, 0], { ds: { scale: 1 } }), false,
  'A-only mode must not retain an invisible wipe handle or drag target.');
assert.strictEqual(api.getVisiblePreviewMode(selectablePreviewNode), 'image_a');
assert.strictEqual(JSON.parse(serializedPreviewSettings).visible_preview_image, 'image_a', 'Preview choice persists in the existing hidden serialized setting.');
api.setVisiblePreviewMode(selectablePreviewNode, 'image_b');
assert.strictEqual(selectablePreviewNode.imgs[0], upstreamB.imgs[0], 'Image B mode displays only B.');
api.setVisiblePreviewMode(selectablePreviewNode, 'loaded');
assert.strictEqual(selectablePreviewNode.imgs[0], loadedFallbackPreview, 'Loaded Image mode ignores connected A/B previews.');
api.setVisiblePreviewMode(selectablePreviewNode, 'auto');
assert.notStrictEqual(selectablePreviewNode.imgs[0], loadedFallbackPreview, 'Returning to Auto restores connected A/B comparison priority.');

// When connected A and B both have no preview yet (for example empty VAE
// Decode or not-yet-executed subgraph outputs), Auto mode uses Loaded Image as
// the final fallback rather than leaving the preview region empty.
const noConnectedPreviewNode = makeKiriNode(1106);
noConnectedPreviewNode.imgs = [loadedFallbackPreview];
noConnectedPreviewNode.widgets.push({ name: 'metadata_export_settings', value: '{}', callback() {} });
links.set(32, { id: 32, origin_id: 105, origin_slot: 0, target_id: 1106, target_slot: 0 });
links.set(33, { id: 33, origin_id: 106, origin_slot: 0, target_id: 1106, target_slot: 1 });
noConnectedPreviewNode.inputs[0].link = 32;
noConnectedPreviewNode.inputs[1].link = 33;
api.installNativePreviewSource(noConnectedPreviewNode);
assert.strictEqual(api.syncLivePreviewFromUpstream(noConnectedPreviewNode), false, 'Empty upstream branches should not create an invalid preview.');
assert.strictEqual(noConnectedPreviewNode.imgs[0], loadedFallbackPreview, 'Auto mode falls back to the selected Loaded Image when neither A nor B has a preview.');
function FallbackImagePreviewWidget() {}
const fallbackPreviewWidget = new FallbackImagePreviewWidget();
let fallbackWidgetOptions = null;
fallbackPreviewWidget.drawWidget = function (_ctx, options) { fallbackWidgetOptions = options; };
noConnectedPreviewNode.widgets.push(fallbackPreviewWidget);
api.patchNativePreviewWidgets(noConnectedPreviewNode);
fallbackPreviewWidget.drawWidget({}, { previewImages: [] });
assert.strictEqual(fallbackWidgetOptions.previewImages[0], loadedFallbackPreview, 'Modern preview widget must also receive the final Loaded Image fallback.');

// A forced branch mode is strict: an unavailable chosen branch remains blank
// rather than silently substituting B, A, or Loaded Image.
api.setVisiblePreviewMode(noConnectedPreviewNode, 'image_a');
assert.strictEqual(noConnectedPreviewNode.imgs.length, 0, 'A-only mode remains blank if A is unavailable.');
api.setVisiblePreviewMode(noConnectedPreviewNode, 'image_b');
assert.strictEqual(noConnectedPreviewNode.imgs.length, 0, 'B-only mode remains blank if B is unavailable.');
api.setVisiblePreviewMode(noConnectedPreviewNode, 'loaded');
assert.strictEqual(noConnectedPreviewNode.imgs[0], loadedFallbackPreview);
api.setVisiblePreviewMode(noConnectedPreviewNode, 'auto');
assert.strictEqual(noConnectedPreviewNode.imgs[0], loadedFallbackPreview);

const previewMenu = api.menuItemsFor(selectablePreviewNode);
const previewMenuEntry = previewMenu.find((item) => item && item.content === 'Visible Preview Image');
assert.ok(previewMenuEntry?.has_submenu, 'The preview selector is available in Kiri right-click menu.');
assert.strictEqual(previewMenuEntry.submenu.options.map((item) => item.content.replace(/^✓ /, '')).join('|'),
  'Auto / Compare (A + B)|Image A only|Image B only|Loaded Image only');
