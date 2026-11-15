/* Runtime tests for opening embedded ComfyUI workflows in a separate workflow tab. */
const fs = require('fs');
const vm = require('vm');
const path = require('path');
const assert = require('assert');

const sourcePath = path.join(__dirname, '..', 'web', 'kiri_image_loader.js');
let source = fs.readFileSync(sourcePath, 'utf8').replace(/^import \{ app \} from ".*";\s*/m, 'const app = globalThis.__app;');
source += `\nglobalThis.__kiriTest = {
  openEmbeddedWorkflow, isOpenableEmbeddedWorkflow, embeddedWorkflowTabName,
  sourceSubmenuItems, setMetadataState
};\n`;

const alerts = [];
const calls = [];
let createdWorkflow = null;
const workflowData = {
  id: 'embedded-id',
  last_node_id: 4,
  last_link_id: 3,
  nodes: [{ id: 4, type: 'LoadImage', widgets_values: ['reference.png'] }],
  links: [],
  groups: [],
  config: {},
  extra: {},
  version: 0.4,
};
const workflowStore = {
  createNewTemporary(name, data) {
    calls.push({ method: 'createNewTemporary', name, data });
    createdWorkflow = { path: `/temp/${name}`, activeState: data };
    return createdWorkflow;
  },
  async openWorkflow(workflow) {
    calls.push({ method: 'openWorkflow', workflow });
  },
};
const app = { extensionManager: { workflow: workflowStore }, registerExtension() {} };
const sandbox = {
  console,
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
  JSON,
  Math,
  window: { alert: (message) => alerts.push(message) },
  document: { createElement: () => ({}) },
  navigator: {},
  setTimeout,
  queueMicrotask,
  globalThis: null,
  __app: app,
};
sandbox.globalThis = sandbox;
vm.runInNewContext(source, sandbox, { filename: sourcePath });
const api = sandbox.__kiriTest;

async function run() {
  assert.strictEqual(api.isOpenableEmbeddedWorkflow(workflowData), true);
  assert.strictEqual(api.isOpenableEmbeddedWorkflow({ nodes: [], edges: [] }), false);
  assert.strictEqual(api.embeddedWorkflowTabName({ filename: 'folder\\portrait.final.png' }), 'portrait.final - Embedded Workflow.json');

  const node = { widgets: [{ name: 'image', value: 'fallback.png' }] };
  api.setMetadataState(node, 'image_a', {
    filename: 'connected-A.png',
    embedded_workflow: workflowData,
  });
  const opened = await api.openEmbeddedWorkflow(node, 'image_a');
  assert.strictEqual(opened, true);
  assert.strictEqual(calls.length, 2);
  assert.strictEqual(calls[0].method, 'createNewTemporary');
  assert.strictEqual(calls[0].name, 'connected-A - Embedded Workflow.json');
  assert.deepStrictEqual(JSON.parse(JSON.stringify(calls[0].data)), workflowData);
  assert.strictEqual(calls[1].method, 'openWorkflow');
  assert.strictEqual(calls[1].workflow, createdWorkflow);
  assert.strictEqual(alerts.length, 0);

  // The payload passed to ComfyUI is detached from the cached metadata object.
  calls[0].data.nodes[0].widgets_values[0] = 'mutated.png';
  assert.strictEqual(workflowData.nodes[0].widgets_values[0], 'reference.png');

  // Older frontends without the workflow-tab API must not replace the current graph.
  app.extensionManager.workflow = {};
  const blocked = await api.openEmbeddedWorkflow(node, 'image_a');
  assert.strictEqual(blocked, false);
  assert.ok(alerts.some((message) => message.includes('current workflow was left unchanged')));

  // Metadata with no actual ComfyUI graph cannot trigger the action.
  api.setMetadataState(node, 'image_b', { filename: 'other.png', embedded_workflow: null });
  const noWorkflow = await api.openEmbeddedWorkflow(node, 'image_b');
  assert.strictEqual(noWorkflow, false);
  assert.strictEqual(alerts.length, 2);

  console.log('Kiri embedded workflow runtime tests passed.');
}

run().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
