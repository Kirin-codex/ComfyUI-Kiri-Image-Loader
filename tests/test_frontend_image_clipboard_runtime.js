/* Browser-stub integration tests for Kiri Image Loader image clipboard actions. */
const fs = require('fs');
const vm = require('vm');
const path = require('path');
const assert = require('assert');

const sourcePath = path.join(__dirname, '..', 'web', 'kiri_image_loader.js');
let source = fs.readFileSync(sourcePath, 'utf8').replace(/^import \{ app \} from ".*";\s*/m, 'const app = globalThis.__app;\n');
source += `\nglobalThis.__kiriClipboardTest = {
  sourceSubmenuItems, canCopyImageSource, copyImageSourceToClipboard,
  pasteImageIntoLoadedImage, uploadClipboardImageToLoadedImage,
  livePreviewByNode, descriptorSignature, setMetadataState,
};\n`;

const app = { graph: { _nodes: [], getNodeById() { return null; }, links: new Map() }, canvas: null, registerExtension() {} };
const alerts = [];
const fetchCalls = [];
const writes = [];
let pasteItems = [];
let uploadForm = null;

class FakeCanvas {
  constructor() { this.width = 1; this.height = 1; this.draws = 0; }
  getContext() {
    return { drawImage: () => { this.draws += 1; }, fillRect() {}, putImageData() {}, getImageData() { return { data: new Uint8ClampedArray(4) }; } };
  }
  toBlob(callback, mime = 'image/png') {
    callback(new Blob(['png-image-data'], { type: mime }));
  }
}
class FakeClipboardItem {
  constructor(data) { this.data = data; this.types = Object.keys(data); }
  async getType(type) { return await this.data[type]; }
}

const clipboard = {
  async write(items) {
    for (const item of items) writes.push(await item.getType('image/png'));
  },
  async read() { return pasteItems; },
};

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
  Blob,
  File,
  FormData,
  ClipboardItem: FakeClipboardItem,
  Image: class {},
  createImageBitmap: async () => ({ width: 3, height: 2, close() {} }),
  document: {
    createElement(tag) {
      assert.strictEqual(tag, 'canvas');
      return new FakeCanvas();
    },
  },
  window: {
    location: { href: 'http://localhost/' },
    ClipboardItem: FakeClipboardItem,
    alert(message) { alerts.push(String(message)); },
  },
  URL,
  fetch: async (url, options = {}) => {
    fetchCalls.push({ url: String(url), options });
    if (String(url).startsWith('/view?') || String(url).startsWith('/kiri-image-loader/image?')) {
      return { ok: true, blob: async () => new Blob(['source-image'], { type: 'image/jpeg' }) };
    }
    if (String(url) === '/upload/image') {
      uploadForm = options.body;
      return { ok: true, json: async () => ({ name: 'Kiri Clipboard 001.png', subfolder: '', type: 'input' }) };
    }
    if (String(url) === '/kiri-image-loader/files') {
      return { ok: true, json: async () => ({ ok: true, files: ['Kiri Clipboard 001.png', 'loaded.png'] }) };
    }
    if (String(url).startsWith('/kiri-image-loader/metadata?')) {
      return { ok: true, json: async () => ({ ok: true, copy_items: [] }) };
    }
    return { ok: false, status: 404, json: async () => ({ error: 'not found' }) };
  },
  navigator: { clipboard },
  setTimeout,
  queueMicrotask,
  globalThis: null,
  __app: app,
};
sandbox.globalThis = sandbox;
vm.runInNewContext(source, sandbox, { filename: sourcePath });
const api = sandbox.__kiriClipboardTest;

function makeNode() {
  const imageWidget = {
    name: 'image',
    value: 'loaded.png',
    options: { values: ['loaded.png'] },
    callback(value) { this.value = value; },
  };
  const node = {
    id: 77,
    type: 'KiriImageLoader',
    comfyClass: 'KiriImageLoader',
    graph: app.graph,
    inputs: [
      { name: 'image_in', link: 1 },
      { name: 'mask_in', link: 2 },
    ],
    outputs: [],
    widgets: [imageWidget],
    setDirtyCanvas() {},
  };
  app.graph._nodes.push(node);
  return node;
}

async function pause() { await new Promise((resolve) => setTimeout(resolve, 15)); }

(async () => {
  const node = makeNode();
  const imageA = { width: 31, height: 19, naturalWidth: 31, naturalHeight: 19, tag: 'actual-A-pixels' };
  const imageB = { width: 13, height: 23, naturalWidth: 13, naturalHeight: 23, tag: 'actual-B-pixels' };
  api.livePreviewByNode.set(node, { source: 'graph', mode: 'compare', a: imageA, b: imageB, renderedImage: imageA, slider: 0.5 });

  assert.ok(api.canCopyImageSource(node, 'loaded'));
  assert.ok(api.canCopyImageSource(node, 'image_a'));
  assert.ok(api.canCopyImageSource(node, 'image_b'));
  const aMenu = api.sourceSubmenuItems(node, 'image_a', 'Image A');
  const bMenu = api.sourceSubmenuItems(node, 'image_b', 'Image B');
  const loadedMenu = api.sourceSubmenuItems(node, 'loaded', 'Loaded Image');
  assert.ok(aMenu.some((item) => item?.content === 'Copy' && !item.disabled));
  assert.ok(bMenu.some((item) => item?.content === 'Copy' && !item.disabled));
  assert.ok(loadedMenu.some((item) => item?.content === 'Copy' && !item.disabled));
  assert.ok(loadedMenu.some((item) => item?.content === 'Paste'));

  // The source menu callbacks must copy actual PNG pixel data for each source.
  for (const [menu, label] of [[aMenu, 'Image A'], [bMenu, 'Image B'], [loadedMenu, 'Loaded Image']]) {
    menu.find((item) => item?.content === 'Copy').callback();
    await pause();
  }
  assert.strictEqual(writes.length, 3, 'Loaded, A and B each place an image on the clipboard.');
  assert.ok(writes.every((blob) => blob.type === 'image/png' && blob.size > 0));
  assert.ok(fetchCalls.some((call) => call.url.startsWith('/view?filename=loaded.png')), 'Loaded Image copies the selected file pixels.');

  // Pasting an image uploads it to ComfyUI input and selects the new file.
  pasteItems = [new FakeClipboardItem({ 'image/jpeg': new Blob(['clipboard-image'], { type: 'image/jpeg' }) })];
  api.pasteImageIntoLoadedImage(node);
  await pause();
  await pause();
  assert.ok(uploadForm, 'Paste must POST the clipboard image to ComfyUI upload.');
  assert.strictEqual(uploadForm.get('image').type, 'image/png', 'Non-PNG clipboard formats are normalized into a valid PNG.');
  assert.ok(uploadForm.get('image').name.endsWith('.png'));
  assert.strictEqual(node.widgets[0].value, 'Kiri Clipboard 001.png');
  assert.ok(node.widgets[0].options.values.includes('Kiri Clipboard 001.png'));
  assert.ok(fetchCalls.some((call) => call.url === '/upload/image'));

  // Empty/missing sources must disable Copy rather than copy a stale canvas.
  node.inputs[0].link = null;
  node.inputs[1].link = null;
  api.livePreviewByNode.delete(node);
  assert.strictEqual(api.canCopyImageSource(node, 'image_a'), false);
  assert.strictEqual(api.canCopyImageSource(node, 'image_b'), false);
  console.log('Kiri Image Loader clipboard runtime tests passed.');
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
