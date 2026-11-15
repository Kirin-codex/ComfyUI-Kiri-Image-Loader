/* Verifies modern CanvasPointer dragging and the old-frontend handle-only fallback. */
const fs = require('fs');
const vm = require('vm');
const path = require('path');
const assert = require('assert');

const sourcePath = path.join(__dirname, '..', 'web', 'kiri_image_loader.js');
let source = fs.readFileSync(sourcePath, 'utf8').replace(/^import \{ app \} from ".*";\s*/m, 'const app = globalThis.__app;');
source += `\nglobalThis.__kiriDragTest = {
  installCompareWidgetPointer, installPreviewInteractionSurface, isNearComparisonHandle, livePreviewByNode,
  attachComparePointerHandler, compareDragByCanvas
};\n`;

function fakePreviewCanvas() {
  const context = {
    fillRect() {}, save() {}, restore() {}, beginPath() {}, rect() {}, clip() {},
    drawImage() {}, moveTo() {}, lineTo() {}, stroke() {}, arc() {}, fill() {},
    set fillStyle(_v) {}, set strokeStyle(_v) {}, set lineWidth(_v) {},
  };
  return { width: 1, height: 1, getContext() { return context; } };
}

class FakeEventTarget {
  constructor() { this.listeners = new Map(); this.style = {}; this.captured = new Set(); }
  addEventListener(type, callback) {
    const list = this.listeners.get(type) || [];
    list.push(callback);
    this.listeners.set(type, list);
  }
  dispatch(type, event) {
    event.preventDefault ||= function () { this.defaultPrevented = true; };
    event.stopPropagation ||= function () { this.propagationStopped = true; };
    event.stopImmediatePropagation ||= function () { this.immediatePropagationStopped = true; };
    for (const callback of this.listeners.get(type) || []) callback(event);
  }
  setPointerCapture(id) { this.captured.add(id); }
  hasPointerCapture(id) { return this.captured.has(id); }
  releasePointerCapture(id) { this.captured.delete(id); }
}

const graph = { _nodes: [] };
const app = { graph, registerExtension() {} };
const sandbox = {
  console, URLSearchParams, Map, WeakMap, WeakSet, Set, Array, Number, String,
  Object, Promise, Math, Image: class {}, document: { createElement: fakePreviewCanvas },
  window: { location: { href: 'http://localhost/' } }, URL, navigator: {}, setTimeout,
  queueMicrotask, globalThis: null, __app: app,
};
sandbox.globalThis = sandbox;
vm.runInNewContext(source, sandbox, { filename: sourcePath });
const api = sandbox.__kiriDragTest;

function makeNode(id) {
  const calls = [];
  const imageWidget = {
    name: 'image',
    onPointerDown() { calls.push('original-widget-handler'); return false; },
  };
  const node = {
    id,
    type: 'KiriImageLoader',
    comfyClass: 'KiriImageLoader',
    pos: [0, 0],
    size: [120, 110],
    widgets: [imageWidget, { name: 'image_preview', y: 0, height: 100 }],
    imgs: [{ naturalWidth: 120, naturalHeight: 80, tag: 'selected' }],
    inputs: [],
    imgs: [],
    setDirtyCanvas() {},
  };
  graph._nodes.push(node);
  const state = {
    mode: 'compare', slider: 0.5,
    a: { naturalWidth: 120, naturalHeight: 80 },
    b: { naturalWidth: 120, naturalHeight: 80 },
  };
  api.livePreviewByNode.set(node, state);
  return { node, state, imageWidget, calls };
}

// Modern ComfyUI: the visible widget claims the drag through CanvasPointer.
{
  const { node, state, imageWidget, calls } = makeNode(42);
  const canvas = {
    pointer: {}, ds: { scale: 1 }, graph,
    convertEventToCanvas(event) { return [event.canvasX, event.canvasY]; },
    convertCanvasToGraph(point) { return point; },
    canvas: new FakeEventTarget(),
  };

  assert.strictEqual(api.installCompareWidgetPointer(node, canvas), true);
  assert.ok(api.isNearComparisonHandle(node, state, [60, 42.5], canvas));
  assert.ok(!api.isNearComparisonHandle(node, state, [18, 42.5], canvas), 'clicking elsewhere in the preview is not a comparer control');

  const bodyPointer = { eDown: { canvasX: 18, canvasY: 42.5 } };
  assert.strictEqual(imageWidget.onPointerDown(bodyPointer, node, canvas), false);
  assert.deepStrictEqual(calls, ['original-widget-handler']);
  assert.strictEqual(state.slider, 0.5);

  const pointer = { eDown: { canvasX: 60, canvasY: 42.5 } };
  assert.strictEqual(imageWidget.onPointerDown(pointer, node, canvas), true);
  assert.strictEqual(typeof pointer.onDrag, 'function');
  assert.strictEqual(typeof pointer.onDragEnd, 'function');
  assert.strictEqual(typeof pointer.onDragCancel, 'function');
  assert.strictEqual(typeof pointer.finally, 'function');
  pointer.onDrag({ canvasX: 90, canvasY: 42.5 });
  assert.ok(Math.abs(state.slider - 0.75) < 0.0001, 'drag moves the divider across the preview');
  pointer.onDragEnd({ canvasX: 90, canvasY: 42.5 });
  pointer.finally();
  assert.ok(Math.abs(state.slider - 0.75) < 0.0001);

  // Picking up near the edge of the enlarged grab target must not jump.
  const edgePointer = { eDown: { canvasX: 96, canvasY: 42.5 } };
  assert.strictEqual(imageWidget.onPointerDown(edgePointer, node, canvas), true);
  assert.ok(Math.abs(state.slider - 0.75) < 0.0001, 'pointer-down preserves grab offset');
  edgePointer.onDrag({ canvasX: 45, canvasY: 42.5 });
  assert.ok(Math.abs(state.slider - 0.325) < 0.0001, 'grab offset keeps handle under pointer');
  edgePointer.onDragCancel();
  edgePointer.finally();
}

// The preview itself is a virtual interaction surface: clicking or dragging
// anywhere on the image must not move the node. Only its visible handle moves
// the comparison divider. The virtual surface adds no visible widget.
{
  const { node, state } = makeNode(44);
  const originalWidget = { name: 'normal-widget' };
  node.getWidgetOnPos = () => originalWidget;
  const originalGetter = node.getWidgetOnPos;
  assert.strictEqual(api.installPreviewInteractionSurface(node), true);

  const surface = node.getWidgetOnPos(24, 42.5);
  assert.ok(surface && surface.name === 'kiri_preview_interaction_surface', 'image preview returns the virtual interaction surface');
  assert.strictEqual(node.widgets.some((item) => item === surface), false, 'virtual surface is not inserted into visible widgets');

  const selected = [];
  const canvas = { processSelect(target) { selected.push(target); } };
  const bodyPointer = { eDown: { canvasX: 24, canvasY: 42.5 } };
  assert.strictEqual(surface.onPointerDown(bodyPointer, node, canvas), true, 'preview surface claims body gestures');
  assert.strictEqual(typeof bodyPointer.onDragStart, 'function', 'body gesture is consumed before node-drag fallback');
  assert.strictEqual(typeof bodyPointer.onDrag, 'function');
  bodyPointer.onDrag({ canvasX: 50, canvasY: 42.5 });
  bodyPointer.onDragEnd({ canvasX: 50, canvasY: 42.5 });
  assert.strictEqual(state.slider, 0.5, 'dragging the image body does not move the comparer divider');
  bodyPointer.onClick();
  assert.deepStrictEqual(selected, [node], 'a simple click can still select the Kiri node');

  const handlePointer = { eDown: { canvasX: 60, canvasY: 42.5 } };
  assert.strictEqual(surface.onPointerDown(handlePointer, node, { ds: { scale: 1 }, processSelect() {} }), true);
  handlePointer.onDrag({ canvasX: 90, canvasY: 42.5 });
  assert.ok(Math.abs(state.slider - 0.75) < 0.0001, 'only the visible handle drag moves the divider');

  assert.strictEqual(node.getWidgetOnPos(130, 42.5), originalWidget, 'outside the preview, normal widgets are returned');
  assert.strictEqual(node.getWidgetOnPos(24, 1), originalWidget, 'above the preview, normal widgets are returned');
  assert.strictEqual(node.widgets.length, 2, 'no permanent control or widget row was added');
  assert.strictEqual(typeof originalGetter, 'function');
}

// Older LiteGraph: pointer capture is used, but the same visible-handle-only hit test applies.
{
  const { node, state } = makeNode(43);
  const element = new FakeEventTarget();
  const canvas = {
    ds: { scale: 1 }, graph, canvas: element,
    convertEventToCanvas(event) { return [event.clientX, event.clientY]; },
    convertCanvasToGraph(point) { return point; },
  };
  api.attachComparePointerHandler(canvas);

  const bodyDown = { button: 0, pointerId: 7, clientX: 18, clientY: 42.5 };
  element.dispatch('pointerdown', bodyDown);
  assert.strictEqual(api.compareDragByCanvas.has(canvas), false, 'body click does not start legacy drag');
  assert.strictEqual(state.slider, 0.5, 'body click does not nudge divider');
  assert.strictEqual(bodyDown.defaultPrevented, true, 'legacy preview body press is consumed before LiteGraph node dragging');
  assert.strictEqual(bodyDown.immediatePropagationStopped, true, 'legacy preview body press cannot reach canvas node-drag handling');

  const handleDown = { button: 0, pointerId: 7, clientX: 60, clientY: 42.5 };
  element.dispatch('pointerdown', handleDown);
  assert.strictEqual(api.compareDragByCanvas.has(canvas), true, 'visible handle begins drag');
  assert.strictEqual(element.hasPointerCapture(7), true, 'drag captures the pointer');
  element.dispatch('pointermove', { pointerId: 7, clientX: 90, clientY: 42.5 });
  assert.ok(Math.abs(state.slider - 0.75) < 0.0001, 'captured pointer movement updates divider');
  element.dispatch('pointerup', { pointerId: 7 });
  assert.strictEqual(api.compareDragByCanvas.has(canvas), false, 'pointer-up ends drag');
  assert.strictEqual(element.hasPointerCapture(7), false, 'pointer capture is released');
}

console.log('Kiri comparer drag runtime tests passed.');
