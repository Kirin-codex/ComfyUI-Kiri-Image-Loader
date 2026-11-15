/* Browser-stub regression test for the PNG metadata editor UI and save route. */
const fs = require('fs');
const vm = require('vm');
const path = require('path');
const assert = require('assert');

const sourcePath = path.join(__dirname, '..', 'web', 'kiri_image_loader.js');
let source = fs.readFileSync(sourcePath, 'utf8').replace(/^import \{ app \} from ".*";\s*/m, 'const app = globalThis.__app;');
source += `\nglobalThis.__kiriTest = { showMetadataEditor, showFullMetadataDialog, getMetadataState, buildMetadataEditorModel } ;\n`;

class FakeElement {
  constructor(tagName) {
    this.tagName = tagName;
    this.style = {};
    this.children = [];
    this.listeners = {};
    this.value = '';
    this.textContent = '';
    this.scrollTop = 0;
    this.disabled = false;
  }
  append(...items) { this.children.push(...items); }
  appendChild(item) { this.children.push(item); return item; }
  addEventListener(name, callback) { (this.listeners[name] ||= []).push(callback); }
  async click() {
    for (const callback of this.listeners.click || []) await callback({ target: this });
  }
  remove() {
    const parent = this.parentNode;
    if (parent) parent.children = parent.children.filter((item) => item !== this);
    this.removed = true;
  }
  focus() {}
  setAttribute() {}
}

const document = {
  body: new FakeElement('body'),
  createElement(tagName) {
    const item = new FakeElement(tagName);
    const append = item.append.bind(item);
    item.append = (...children) => {
      for (const child of children) child.parentNode = item;
      append(...children);
    };
    const appendChild = item.appendChild.bind(item);
    item.appendChild = (child) => { child.parentNode = item; return appendChild(child); };
    return item;
  },
};
const alerts = [];
const requests = [];
const sandbox = {
  console,
  URLSearchParams,
  Map, WeakMap, WeakSet, Set, Array, Number, String, Object, Promise, Math, JSON, URL,
  document,
  navigator: {},
  window: { location: { href: 'http://localhost/' }, alert(message) { alerts.push(message); } },
  fetch: async (url, options = {}) => {
    requests.push({ url, options });
    if (options.method === 'POST') {
      return { ok: true, json: async () => ({ ok: true, message: 'Updated 2 embedded PNG metadata field(s).' }) };
    }
    return { ok: true, json: async () => ({
      ok: true,
      metadata_editable: true,
      editable_metadata: [
        { keyword: 'prompt', value: 'updated prompt', format: 'iTXt' },
        { keyword: 'negative_prompt', value: 'updated negative', format: 'iTXt' },
      ],
      copy_items: [],
    }) };
  },
  setTimeout, queueMicrotask,
  globalThis: null,
  __app: { canvas: null, registerExtension() {} },
};
sandbox.globalThis = sandbox;
vm.runInNewContext(source, sandbox, { filename: sourcePath });
const api = sandbox.__kiriTest;

const node = { setDirtyCanvas() {} };
const descriptor = { filename: 'edited-image.png', type: 'output', subfolder: 'results', pathToken: '' };
const kiriRecord = {
  schema_version: 1,
  collector: 'Kiri Image Loader',
  graph: {
    prompts: {
      positive: '1girl, sitting, room, smiling',
      negative: 'blurry, low quality',
    },
    samplers: [{
      class_type: 'KSampler',
      settings: { seed: 123456, steps: 30, cfg: 6.5, sampler_name: 'euler', scheduler: 'normal', denoise: 1 },
    }],
  },
};
const data = {
  metadata_editable: true,
  width: 1024,
  height: 1280,
  copy_items: [],
  editable_metadata: [
    { keyword: 'prompt', value: JSON.stringify({ '10': { class_type: 'CLIPTextEncode', inputs: { text: 'old graph prompt' } } }), format: 'tEXt' },
    { keyword: 'kiri_metadata', value: JSON.stringify(kiriRecord), format: 'iTXt' },
    { keyword: 'parameters', value: 'old graph prompt\\nSteps: 30, Seed: 123456', format: 'iTXt' },
    { keyword: 'comment', value: 'preserve this note', format: 'tEXt' },
  ],
};

(async () => {
  const model = api.buildMetadataEditorModel(data);
  assert.strictEqual(model.positive, '1girl, sitting, room, smiling');
  assert.strictEqual(model.negative, 'blurry, low quality');
  assert.deepStrictEqual(Array.from(model.parameters, (item) => item.label), [
    'Seed', 'Steps', 'CFG', 'Sampler', 'Scheduler', 'Denoise', 'Width', 'Height',
  ]);
  assert.deepStrictEqual(Array.from(model.additional, (entry) => entry.keyword), ['comment'], 'Internal prompt/Kiri/parameters chunks should not appear as raw editable fields.');

  api.showMetadataEditor(node, 'image_a', descriptor, data);
  const overlay = document.body.children[0];
  assert.ok(overlay, 'Editor overlay should be attached to the document.');
  const dialog = overlay.children[0];
  const body = dialog.children[2];
  const sections = body.children.filter((child) => child._kiriSection);
  assert.deepStrictEqual(Array.from(sections, (section) => section._kiriSection), [
    'Prompts', 'Parameters', 'Additional Metadata',
  ]);
  assert.ok(!sections.some((section) => section._kiriSection === 'Kiri Metadata'));

  const promptSection = sections[0];
  const positiveEditor = promptSection.children[1].children[1];
  assert.strictEqual(positiveEditor.value, '1girl, sitting, room, smiling');
  positiveEditor.value = 'edited positive prompt';
  const negativeEditor = promptSection.children[2].children[1];
  assert.strictEqual(negativeEditor.value, 'blurry, low quality');
  negativeEditor.value = 'edited negative prompt';
  const parameterSection = sections[1];
  const seedInput = parameterSection.children[1].children[1];
  assert.strictEqual(seedInput.value, '123456');
  seedInput.value = '987654';
  const widthInput = parameterSection.children.find((row) => row.children?.[0]?.textContent === 'Width').children[1];
  assert.strictEqual(widthInput.value, '1024');
  widthInput.value = '1536';

  const footer = dialog.children[3];
  const actionGroup = footer.children[1];
  const saveButton = actionGroup.children[1];
  await saveButton.click();

  const post = requests.find((request) => request.options.method === 'POST');
  assert.ok(post, 'Save should call the local file-action endpoint.');
  const payload = JSON.parse(post.options.body);
  assert.strictEqual(payload.action, 'edit_metadata');
  assert.strictEqual(payload.filename, 'edited-image.png');
  assert.strictEqual(payload.type, 'output');
  assert.strictEqual(payload.editor_values.positive_prompt, 'edited positive prompt');
  assert.strictEqual(payload.editor_values.negative_prompt, 'edited negative prompt');
  assert.strictEqual(payload.editor_values.image_width, '1536');
  assert.deepStrictEqual(payload.editor_values.parameters[0], {
    kind: 'sampler', sampler_index: 0, key: 'seed', label: 'Seed', value: '987654',
  });
  assert.deepStrictEqual(payload.metadata_entries.map((entry) => entry.keyword).sort(), [
    'comment', 'kiri_metadata', 'parameters', 'prompt',
  ]);
  assert.ok(requests.some((request) => String(request.url).startsWith('/kiri-image-loader/metadata?')), 'Editor should refresh metadata after save.');
  assert.strictEqual(overlay.removed, true);
  assert.ok(alerts.some((message) => message.includes('Updated 2 embedded PNG metadata field(s).')));

  // Workflow-OFF files must remain free of ComfyUI graph data when edited.
  const textOnlyDescriptor = { filename: 'text-only.png', type: 'output', subfolder: '', pathToken: '' };
  const textOnlyData = {
    metadata_editable: true,
    width: 512,
    height: 768,
    copy_items: [
      { category: 'Prompts', label: 'Positive Prompt', value: 'text-only positive' },
      { category: 'Prompts', label: 'Negative Prompt', value: 'text-only negative' },
      { category: 'Generation', label: 'Steps', value: '24' },
      { category: 'Generation', label: 'Seed', value: '777' },
      { category: 'ComfyUI Nodes', label: 'Nodes Used', value: 'Custom Sampler\nCustom Prompt Node' },
    ],
    editable_metadata: [
      { keyword: 'parameters', value: 'text-only positive\nNegative prompt: text-only negative\nSteps: 24, Seed: 777', format: 'iTXt' },
      { keyword: 'Kiri Nodes Used', value: 'Custom Sampler\nCustom Prompt Node', format: 'tEXt' },
      { keyword: 'Kiri Metadata Mode', value: 'ComfyUI Workflow OFF; Other-Platform Metadata OFF; no ComfyUI graph data', format: 'tEXt' },
    ],
  };
  api.showMetadataEditor(node, 'image_a', textOnlyDescriptor, textOnlyData);
  const textOnlyOverlay = document.body.children[document.body.children.length - 1];
  const textOnlyDialog = textOnlyOverlay.children[0];
  assert.ok(textOnlyDialog.children[1].textContent.includes('no ComfyUI-loadable node graph'));
  assert.ok(textOnlyDialog.children[1].textContent.includes('will not add JSON workflow data'));
  const textOnlyBody = textOnlyDialog.children[2];
  const textOnlySections = textOnlyBody.children.filter((child) => child._kiriSection);
  assert.deepStrictEqual(Array.from(textOnlySections, (section) => section._kiriSection), ['Prompts', 'Parameters']);
  const textOnlyNegative = textOnlySections[0].children[2].children[1];
  textOnlyNegative.value = 'edited text-only negative';
  await textOnlyDialog.children[3].children[1].children[1].click();
  const textOnlyPost = requests.filter((request) => request.options.method === 'POST').slice(-1)[0];
  const textOnlyPayload = JSON.parse(textOnlyPost.options.body);
  assert.deepStrictEqual(Array.from(textOnlyPayload.metadata_entries, (entry) => entry.keyword).sort(), [
    'Kiri Metadata Mode', 'Kiri Nodes Used', 'parameters',
  ].sort());
  assert.strictEqual(textOnlyPayload.editor_values.negative_prompt, 'edited text-only negative');

  // Long platform metadata keys and values must stay in their own columns.
  api.showFullMetadataDialog({
    copy_items: [
      { category: 'Platform Handoff', label: 'platform_handoff.pixai_api.loras[0].name', value: 'A'.repeat(180) },
      { category: 'Platform Handoff', label: 'platform_handoff.tensor_art.loras[1].sha256', value: '4d3501b807ad8a917a36b20b343adae597680fcee794d101edd432dfeb449dee' },
    ],
  });
  const viewerOverlay = document.body.children[document.body.children.length - 1];
  assert.ok(viewerOverlay, 'Full metadata viewer should open.');
  const viewerDialog = viewerOverlay.children[0];
  const viewerBody = viewerDialog.children[1];
  const viewerSection = viewerBody.children[0];
  const viewerRow = viewerSection.children[1];
  const viewerLabel = viewerRow.children[0];
  const viewerValue = viewerRow.children[1];
  assert.ok(viewerRow.style.gridTemplateColumns.includes('minmax(0'), 'Both viewer grid columns must be shrinkable.');
  assert.strictEqual(viewerLabel.style.minWidth, '0', 'Long metadata labels must be allowed to shrink inside their grid cell.');
  assert.strictEqual(viewerLabel.style.overflowWrap, 'anywhere', 'Long dotted/underscored metadata keys must wrap rather than spill into the value column.');
  assert.strictEqual(viewerValue.style.minWidth, '0', 'Metadata values must not force the grid wider than the viewer.');
  assert.strictEqual(viewerValue.style.overflowWrap, 'anywhere', 'Long metadata values must wrap inside their own column.');
  console.log('Kiri structured PNG metadata editor and viewer runtime tests passed.');
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
