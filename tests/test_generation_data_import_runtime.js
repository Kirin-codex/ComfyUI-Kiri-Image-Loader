/* Regression tests for explicit, type-safe generation-data imports. */
const fs = require('fs');
const vm = require('vm');
const path = require('path');
const assert = require('assert');

const sourcePath = path.join(__dirname, '..', 'web', 'kiri_image_loader.js');
let source = fs.readFileSync(sourcePath, 'utf8').replace(/^import \{ app \} from ".*";\s*/m, 'const app = globalThis.__app;');
source += `\nglobalThis.__kiriImportTest = {
  generationDataForSource, nodeClassType, sourceNodeCandidates, compatibleWidgetFields,
  applyNodeWidgetValues, targetWidgetForInput, sourceSubmenuItems, setMetadataState
};\n`;

const graph = { id: 'current-workflow-uuid', _nodes: [], changeCount: 0, change() { this.changeCount++; } };
const app = { graph, canvas: { dirty: 0, setDirty() { this.dirty++; } }, registerExtension() {} };
const alerts = [];
const sandbox = {
  console, URLSearchParams, Map, WeakMap, WeakSet, Set, Array, Number, String, Object,
  Promise, Math, JSON, URL, navigator: {},
  window: { alert(message) { alerts.push(message); } },
  document: {}, setTimeout, queueMicrotask, globalThis: null, __app: app,
};
sandbox.globalThis = sandbox;
vm.runInNewContext(source, sandbox, { filename: sourcePath });
const api = sandbox.__kiriImportTest;

function widget(name, value, type = 'number') {
  return { name, value, type, callbackCalls: 0, callback(next) { this.value = next; this.callbackCalls++; } };
}
function targetNode(id, type, widgets, inputs = []) {
  const node = {
    id, type, comfyClass: type, graph, widgets, inputs,
    dirty: 0, setDirtyCanvas() { this.dirty++; },
  };
  graph._nodes.push(node);
  return node;
}

const samplerSource = {
  id: '8', class_type: 'KSamplerAdvanced', title: 'Source Refinement',
  inputs: {
    noise_seed: 998877, steps: 32, cfg: 6.25, sampler_name: 'dpmpp_2m', scheduler: 'karras',
    positive: ['3', 0], negative: ['4', 0], model: ['2', 0], latent_image: ['5', 0],
  },
};
const sourceData = {
  generation_data: {
    schema_version: 1, source: 'Kiri structured graph', workflow_id: 'other-workflow',
    nodes: [samplerSource, {
      id: '9', class_type: 'KSamplerAdvanced', title: 'Source First Pass',
      inputs: { noise_seed: 123, steps: 12, cfg: 3.0 },
    }],
  },
};
const firstPass = targetNode('44', 'KSamplerAdvanced', [
  widget('noise_seed', 111), widget('steps', 20), widget('cfg', 4.0),
]);
const refinement = targetNode('99', 'KSamplerAdvanced', [
  widget('noise_seed', 222), widget('steps', 18), widget('cfg', 5.0),
  widget('sampler_name', 'euler', 'combo'), widget('scheduler', 'normal', 'combo'),
], [{ name: 'positive', link: 400 }, { name: 'model', link: 401 }, { name: 'steps', link: null }]);
const ordinarySampler = targetNode('101', 'KSampler', [widget('steps', 8), widget('cfg', 1.0)]);
const imageNode = targetNode('111', 'KiriImageLoader', []);
imageNode.widgets.push({ name: 'image', value: 'source.png' });

api.setMetadataState(imageNode, 'loaded', { generation_data: sourceData.generation_data, copy_items: [] });
const imageMenu = api.sourceSubmenuItems(imageNode, 'loaded', 'Loaded Image');
const applyMenuItem = imageMenu.find((item) => item?.content === 'Apply Generation Data to Node…');
assert.ok(applyMenuItem && !applyMenuItem.disabled, 'Loaded Image source menu exposes the generation-data importer when compatible targets exist.');

const resolved = api.generationDataForSource(sourceData);
assert.strictEqual(resolved.nodes.length, 2);
const sourceCandidates = api.sourceNodeCandidates(sourceData, imageNode);
assert.strictEqual(sourceCandidates.length, 2, 'Both original KSamplerAdvanced stages should be independently selectable.');
assert.ok(sourceCandidates.every((candidate) => candidate.targets.length === 2), 'Candidates match same exact implementation type, not ID or canvas order.');
assert.ok(!sourceCandidates.some((candidate) => candidate.targets.includes(ordinarySampler)), 'KSampler and KSamplerAdvanced must not cross-match.');
assert.strictEqual(refinement.id, '99', "Destination identity comes from the selected target choice.");

const available = api.compatibleWidgetFields(samplerSource, refinement);
assert.deepStrictEqual(Array.from(available, (field) => field.name), [
  'noise_seed', 'steps', 'cfg', 'sampler_name', 'scheduler',
]);
assert.ok(!available.some((field) => ['positive', 'negative', 'model', 'latent_image'].includes(field.name)), 'Links and connected socket values are never imported.');

(async () => {
  const imported = await api.applyNodeWidgetValues(refinement, available.filter((field) => ['noise_seed', 'steps', 'sampler_name'].includes(field.name)));
  assert.deepStrictEqual(Array.from(imported.applied), ['noise_seed', 'steps', 'sampler_name']);
  assert.deepStrictEqual(Array.from(imported.skipped), []);
  assert.strictEqual(refinement.widgets.find((item) => item.name === 'noise_seed').value, 998877);
  assert.strictEqual(refinement.widgets.find((item) => item.name === 'steps').value, 32);
  assert.strictEqual(refinement.widgets.find((item) => item.name === 'sampler_name').value, 'dpmpp_2m');
  assert.strictEqual(refinement.widgets.find((item) => item.name === 'sampler_name').options.values.includes('dpmpp_2m'), true, 'Missing combo values are added to the local option list.');
  assert.strictEqual(refinement.widgets.find((item) => item.name === 'cfg').value, 5.0, 'Unselected values remain untouched.');
  assert.strictEqual(firstPass.widgets.find((item) => item.name === 'noise_seed').value, 111, 'Other same-type destination nodes remain untouched.');
  assert.strictEqual(refinement.dirty, 1);
  assert.ok(graph.changeCount >= 1);

  // A connected scalar-named destination input is excluded even if it has a widget.
  refinement.inputs.find((item) => item.name === 'steps').link = 999;
  assert.ok(!api.compatibleWidgetFields(samplerSource, refinement).some((field) => field.name === 'steps'));

  // Runtime-generated metadata uses the same graph path as a disk-inspected PNG.
  const runtime = api.generationDataForSource({ runtime_metadata: { graph: { active_upstream_nodes: [samplerSource] } } });
  assert.strictEqual(runtime.nodes[0].id, '8');

  // Kiri LoRA Loader hidden canonical JSON remains supported and its own
  // onConfigure hook is called to rebuild the dynamic row UI.
  const loraGraph = { id: 'current-workflow-uuid', change() {} };
  const lora = {
    id: '123', type: 'KeryulaPowerLoraLoader', comfyClass: 'KeryulaPowerLoraLoader', graph: loraGraph,
    widgets: [widget('loras_json', '[]', 'text'), widget('trigger_lora', '', 'text'), widget('trigger_words', '', 'text')],
    inputs: [], configureCalls: 0, async _kLorasPromise() {},
    onConfigure() { this.configureCalls++; }, setDirtyCanvas() {},
  };
  const loraResult = await api.applyNodeWidgetValues(lora, [
    { name: 'loras_json', sourceValue: '[{"on":true,"lora":"style.safetensors","strength_model":0.7,"strength_clip":0.5}]' },
    { name: 'trigger_lora', sourceValue: 'style.safetensors' },
    { name: 'trigger_words', sourceValue: 'style_tag' },
  ]);
  assert.strictEqual(loraResult.applied.length, 3);
  assert.strictEqual(lora.configureCalls, 1, 'Kiri LoRA loader rebuilds its dynamic rows after canonical JSON import.');

  console.log('Kiri generation-data import runtime tests passed.');
})().catch((error) => { console.error(error); process.exitCode = 1; });
