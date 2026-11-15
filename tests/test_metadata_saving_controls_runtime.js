/* Regression tests for Kiri's hidden metadata settings and right-click controls. */
const fs = require('fs');
const vm = require('vm');
const path = require('path');
const assert = require('assert');

const sourcePath = path.join(__dirname, '..', 'web', 'kiri_image_loader.js');
let source = fs.readFileSync(sourcePath, 'utf8').replace(/^import \{ app \} from ".*";\s*/m, 'const app = globalThis.__app;');
assert.ok(source.includes('minmax(0, 34%) minmax(0, 1fr) auto'), 'Metadata viewer reserves an independent action column.');
assert.ok(source.includes('overflowWrap: \"anywhere\"'), 'Long metadata keys and values must wrap inside their own columns.');
assert.ok(source.includes('copyButton.textContent = \"Copy JSON\"'), 'JSON handoff rows expose a one-click Copy JSON action.');
source += `\nglobalThis.__kiriTest = { hideMetadataSettingsWidgets, imageSavingMenuItems, metadataSavingMenuItems, menuItemsFor, setKiriWidgetValue, metadataCopyItems, getMetadataExportSettings };\n`;
const app = { canvas: null, registerExtension() {} };
const sandbox = {
  console, URLSearchParams, Map, WeakMap, WeakSet, Set, Array, Number, String, Object,
  Promise, Math, JSON, URL, navigator: {}, window: { alert() {} },
  document: { createElement() { return {}; }, body: { appendChild() {} } },
  setTimeout, queueMicrotask, globalThis: null, __app: app,
};
sandbox.globalThis = sandbox;
vm.runInNewContext(source, sandbox, { filename: sourcePath });
const api = sandbox.__kiriTest;

const customWidget = { name: 'custom_metadata_text', type: 'text', value: 'Artist: Keryula', options: {}, computeSize() { return [300, 60]; } };
const workflowWidget = { name: 'embed_workflow', type: 'toggle', value: false, options: {}, computeSize() { return [100, 24]; } };
const exportWidget = { name: 'metadata_export_settings', type: 'text', value: '{\"parameters_format\":\"a1111\"}', options: {}, computeSize() { return [100, 24]; } };
const node = {
  id: 12,
  widgets: [customWidget, workflowWidget, exportWidget],
  inputs: [
    { name: 'image_in', link: null },
    { name: 'mask_in', link: null },
    { name: 'custom_metadata', link: null },
  ],
  setDirtyCanvas() {},
};

api.hideMetadataSettingsWidgets(node);
for (const widget of [customWidget, workflowWidget, exportWidget]) {
  assert.strictEqual(widget.hidden, true);
  assert.strictEqual(widget.options.hidden, true);
  assert.deepStrictEqual(Array.from(widget.computeSize()), [0, -4]);
  assert.strictEqual(widget.serialize, true);
  assert.strictEqual(widget.options.serialize, true);
}

let items = api.metadataSavingMenuItems(node);
assert.ok(items.some((item) => item?.content === 'Compatibility Export Settings…'));
assert.ok(items.some((item) => item?.content?.startsWith('Edit Custom Metadata')));
assert.ok(items.some((item) => item?.content === 'Clear Manual Custom Metadata' && !item.disabled));
assert.ok(items.some((item) => item?.content === 'Save Positive Prompt: ON'));
assert.ok(items.some((item) => item?.content === 'Save Negative Prompt: ON'));
assert.ok(items.some((item) => item?.content === 'Save Node Names (Nodes Used): ON'));
assert.ok(items.some((item) => item?.content === 'Other-Platform Metadata: ON'));
assert.ok(!items.some((item) => item?.content?.includes('Save Images From')), 'Image-save routing no longer appears in Metadata Saving.');
let imageItems = api.imageSavingMenuItems(node);
assert.ok(imageItems.some((item) => item?.content === '✓ Input A only'), 'Image saving defaults to A to preserve prior behavior.');
assert.ok(imageItems.some((item) => item?.content?.endsWith('Input B only')));
assert.ok(imageItems.some((item) => item?.content?.endsWith('Both A and B')));
imageItems.find((item) => item?.content?.endsWith('Input B only')).callback();
assert.strictEqual(api.getMetadataExportSettings(node).save_images_from, 'b');
imageItems = api.imageSavingMenuItems(node);
imageItems.find((item) => item?.content?.endsWith('Both A and B')).callback();
assert.strictEqual(api.getMetadataExportSettings(node).save_images_from, 'both');
imageItems = api.imageSavingMenuItems(node);
imageItems.find((item) => item?.content?.endsWith('Input A only')).callback();
assert.strictEqual(api.getMetadataExportSettings(node).save_images_from, 'a');
items.find((item) => item?.content === 'Save Positive Prompt: ON').callback();
assert.strictEqual(api.getMetadataExportSettings(node).save_positive_prompt, false);
assert.strictEqual(api.getMetadataExportSettings(node).save_negative_prompt, true);
items = api.metadataSavingMenuItems(node);
items.find((item) => item?.content === 'Save Negative Prompt: ON').callback();
assert.strictEqual(api.getMetadataExportSettings(node).save_negative_prompt, false);
items = api.metadataSavingMenuItems(node);
items.find((item) => item?.content === 'Save Node Names (Nodes Used): ON').callback();
assert.strictEqual(api.getMetadataExportSettings(node).save_node_names, false);
items = api.metadataSavingMenuItems(node);
items.find((item) => item?.content === 'Other-Platform Metadata: ON').callback();
assert.strictEqual(api.getMetadataExportSettings(node).platform_metadata, false);

// The workflow switch is now at the top-level context menu, even with no image selected.
let root = api.menuItemsFor(node);
assert.ok(root[0].content.includes('OFF'), 'Workflow embedding should default off for privacy.');
root[0].callback();
assert.strictEqual(workflowWidget.value, true);
root = api.menuItemsFor(node);
assert.ok(root[0].content.includes('ON'));
assert.strictEqual(root[2].content, 'Visible Preview Image');
assert.strictEqual(root[4].content, 'Image A (not connected)');
assert.strictEqual(root[5].content, 'Image B (not connected)');
assert.strictEqual(root[7].content, 'Loaded Image');
const imageSavingRoot = root.find((item) => item?.content === 'Image Saving');
const metadataSavingRoot = root.find((item) => item?.content === 'Metadata Saving');
assert.ok(imageSavingRoot?.has_submenu, 'Image Saving is its own top-level context-menu category.');
assert.strictEqual(imageSavingRoot.submenu.title, 'Image Saving');
assert.deepStrictEqual(Array.from(imageSavingRoot.submenu.options, (item) => item.content.replace(/^✓ /, '')), ['Input A only', 'Input B only', 'Both A and B']);
assert.ok(metadataSavingRoot?.has_submenu);
assert.ok(!metadataSavingRoot.submenu.options.some((item) => item?.content?.includes('Input A only') || item?.content?.includes('Input B only') || item?.content?.includes('Both A and B')),
  'Metadata Saving no longer contains the A/B/Both image-saving options.');

node.inputs[2].link = 77;
items = api.metadataSavingMenuItems(node);
assert.ok(items.some((item) => item?.content === 'Connected Custom Metadata Overrides Manual Text'));

const onlyImportant = api.metadataCopyItems({ copy_items: [
  { label: 'Positive Prompt', value: 'p' }, { label: 'Negative Prompt', value: 'n' },
  { label: 'Seed', value: '42' }, { label: 'Steps', value: '30' },
  { label: 'Checkpoint', value: 'model.safetensors' }, { label: 'LoRA Name', value: 'style.safetensors' },
] });
assert.strictEqual(JSON.stringify(Array.from(onlyImportant.filter((item) => item?.callback && item.content.startsWith('Copy ')), (item) => item.content)), JSON.stringify([
  'Copy Positive Prompt', 'Copy Negative Prompt', 'Copy Seed(s)', 'Copy KSampler Settings'
]));
const legacyWidget = { name: 'metadata_export_settings', type: 'text', value: '{\"save_prompts\":false}', options: {} };
const legacyNode = { widgets: [legacyWidget], inputs: [], setDirtyCanvas() {} };
const migrated = api.getMetadataExportSettings(legacyNode);
assert.strictEqual(migrated.save_positive_prompt, false);
assert.strictEqual(migrated.save_negative_prompt, false);
assert.ok(!Object.prototype.hasOwnProperty.call(migrated, 'save_prompts'));
console.log('Kiri metadata-saving control runtime tests passed.');
