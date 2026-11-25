## 0.5.21
- Replaced the per-generation date/time directory option with a three-way **Save Folder Organization** choice under **Image Saving**: **None (Save Path)**, **Month (`YYYY-MM`)**, or **Month + Day (`YYYY-MM/YYYY-MM-DD`)**. Month mode puts all generations for that month directly in its month folder; Month + Day puts all generations for that local calendar day directly in its day folder. No per-generation timestamp directories are created.
- New workflows default to **None**, preserving flat-folder saving. Existing v0.5.20 workflows with the former date/time-folder toggle enabled migrate to **Month + Day**; explicit new-mode values take precedence.
- Kept A/B side reversal, shared A/B save destination per execution, collision-safe filenames, and companion metadata placement unchanged.
- Added regression coverage for all three modes, automatic month/day rollover folder paths, v0.5.20 preference migration, and A/B shared-directory behavior.

## 0.5.20
- Added a persistent **A on Right (Swap Sides)** option under **Visible Preview Image**. The comparison renderer can now place A on the right and B on the left without swapping input sockets, passthrough outputs, metadata sources, or save targets. The choice is stored in the existing hidden settings widget and is restored with the workflow.
- Added optional **Date & Time Folders** organization under **Image Saving**. When enabled, images are saved beneath `Save Path/YYYY-MM/YYYY-MM-DD_HH-MM-SS/`; A and B from one execution share the same timestamp folder. The setting is OFF by default and leaves existing flat-path behavior unchanged.
- Added regression coverage for both comparison orientations, persistent menu settings, date-folder creation, A/B shared-folder behavior, and sidecar/Easy Diffusion companion-file placement.

## 0.5.19
- Made generated A/B comparison previews persistent across ComfyUI workflow-tab switches and node reconstruction. The latest downscaled preview payloads are saved in the Kiri node's serialized properties and decoded again when that workflow tab is restored; no new generation is required.
- Preserved the comparison divider position with the saved preview. The restore path verifies that the current A/B input sources still match the sources used for the saved preview, preventing a comparison from being reused after its connections change.
- Kept the preview state separate from image metadata, workflow embedding preferences, IMAGE/MASK passthrough, and file-saving behavior. Persistence is capped at 8 MiB of combined encoded preview data so unusually large preview payloads do not silently inflate workflow files without limit.
- Added frontend runtime coverage for workflow JSON serialize/restore, two-sided comparison reconstruction, and rejecting stale previews when either source connection changes.

## 0.5.18
- Moved the persistent **Input A only / Input B only / Both A and B** save-target choices out of **Metadata Saving** into their own top-level **Image Saving** right-click menu category.
- Kept the existing `save_images_from` setting, serialized storage, default of Input A, selected-state indicators, and backend A/B/Both save behavior unchanged. No node widgets, sockets, geometry, or image-saving logic were changed.
- Added runtime regression coverage confirming the new top-level category, all three choices, setting persistence/update behavior, and absence of image-saving choices from Metadata Saving.

## 0.5.17
- Fixed Kiri-to-Kiri live preview fallback: when an upstream Kiri A or B input is genuinely unconnected, its corresponding output preview now resolves to that node's selected Loaded Image. A downstream Kiri node therefore receives a live A/B preview and compares it against its own other branch instead of appearing empty while ignoring the upstream Loaded Image.
- Changed both unconnected A and B runtime outputs to default to the selected Loaded Image (IMAGE). A/B remain dual IMAGE/MASK passthrough sockets: actually connected MASK values still pass through unchanged; a connected-but-empty socket still receives a typed neutral placeholder and never falls back to the selected file. To use a file's alpha mask, connect a MASK source explicitly.
- Added persistent right-click **Metadata Saving → Save Images From** choices: **Input A only** (default), **Input B only**, and **Both A and B**. A remains the default so existing workflows preserve the previous save behavior.
- Added branch-aware B saving and metadata. B files are collected from the B input's own active upstream graph and tracked as saved B outputs, so downstream metadata/copy tracing does not mistake B for A.
- When both branches are saved, each is written independently. The existing collision-safe allocator prevents overwrites and appends numbered suffixes (`_00001_`, `_00002_`, etc.) when multiple branches, batch samples, or prior files would otherwise share a name.
- Added regressions for Kiri-to-Kiri A and B selected-file preview fallback, connected-but-empty branch isolation, A/B/Both save routing, branch-specific metadata, B-only output tracking, duplicate-name numbering, and preservation of connected IMAGE/MASK passthrough.

## 0.5.16
- Made image saving explicitly opt-in: both **Save Path** and **File Name** now default to blank. If either is blank, Kiri does not create an output directory or write an image; A/B/C outputs still pass through/resolve normally. A supplied path must pass the existing managed-relative or absolute-path validation, and an invalid destination is reported without breaking image routing.
- Added output **C (IMAGE)** as a dedicated output for the selected **Loaded Image**. C remains the selected file image even when both A and B are occupied by connected upstream tensors.
- Preserved independent A/B semantics: a usable connected A or B tensor passes through unchanged; when an input is genuinely unconnected it keeps the selected file IMAGE (A) or alpha MASK (B) fallback. Connected-but-empty inputs continue to receive typed neutral placeholders. C always independently exposes the selected file image, or a safe neutral IMAGE if no file is selected/available.
- Changed live-preview payload generation to exclude neutral safety placeholders. Auto / Compare therefore displays valid A/B pixels when present and falls back to the selected Loaded Image when neither branch has a usable preview.
- Made frontend preview and metadata tracing output-slot aware for chained Kiri nodes: A traces only A, B only B, and C resolves only the selected Loaded Image, without accidentally traversing unrelated A/B graphs.
- Added regressions for empty save fields, partial save configuration, invalid paths, all A/B fallback combinations, C with A+B connected, missing Loaded Image, preview/metadata routing for output C, and output-aware graph collection.

## 0.5.15
- Added a persistent right-click **Visible Preview Image** selector with **Auto / Compare (A + B)**, **Image A only**, **Image B only**, and **Loaded Image only** modes. The choice is stored in the existing hidden serialized metadata-settings widget and adds no visible controls or node geometry.
- Auto / Compare remains the default: both available connected sources display as the A/B wipe; a single available source displays alone; if neither A nor B has a usable preview, the selected Loaded Image is shown as the final fallback.
- Forced A-only and B-only modes are strict and do not substitute another source when the chosen branch has no preview. Loaded Image-only always shows the selected image.
- The selected preview affects only presentation. It does not change IMAGE/MASK tensor outputs, file saving, metadata source routing, clipboard behavior or workflow connections.
- Added frontend runtime regressions for persistent preview selection, default A/B priority, each forced display mode, strict missing-branch behavior, Loaded Image fallback when both inputs lack previews, and modern native preview-widget routing.

## 0.5.14
- Fixed ComfyUI subgraph-boundary handling for connected IMAGE/MASK preview and metadata tracing. Kiri no longer treats `SubgraphNode.getInputLink(slot)` as an ordinary external-input resolver; that method is output-oriented on ComfyUI SubgraphNode instances. External inputs are resolved through the boundary node's own parent-graph input link IDs.
- Added exact output-slot resolution through ComfyUI's public `resolveSubgraphOutputLink(slot)` API, recursively following nested subgraphs to the actual interior node/output. This prevents A/B preview or metadata tracing from accidentally selecting the wrong interior branch when the subgraph exposes multiple outputs.
- Added graph-scoped preview/output lookup keys because ComfyUI node IDs are local to each graph/subgraph layer; retained the legacy local-ID lookup as a compatibility fallback.
- Isolated optional Kiri preview, metadata and comparison-widget integration from ComfyUI's native canvas draw loop. A transient proxy, stale link, missing slot or malformed preview record is logged at most once per affected Kiri node and cannot suppress the native `drawNode()` call.
- Bounded display-only A/B comparison canvases to a maximum 2048-pixel edge and 2-megapixel area so very large tiled/upscaled previews do not trigger a second full-resolution allocation. Source tensors and saved images are unchanged.
- Added runtime regressions for SubgraphNode's output-oriented `getInputLink` behavior, exact multi-output and nested-subgraph traversal, graph-scoped node IDs, subgraph-to-Kiri A/B routing, high-resolution preview bounds, and renderer exception isolation.

# Changelog

## 0.5.13
- Added source-specific **Copy** actions for Loaded Image, Image A, and Image B. Copy exports actual image pixels as PNG; A/B prefer exact source output slots or saved Kiri outputs, not an unrelated upstream ancestor image.
- Added **Paste** to the Loaded Image submenu. Clipboard images are normalized to PNG, uploaded into ComfyUI's input folder through its existing upload endpoint, then selected in Kiri. A manual Ctrl+V/Cmd+V paste target is provided when clipboard-read permissions are unavailable.
- Hardened empty/bypassed IMAGE/MASK handling for mixed sources. Valid branches pass through unchanged; connected-but-empty branches receive typed finite neutral tensors; unconnected branches retain their normal selected-file fallback. Synthetic placeholders are never saved as generated outputs.
- Preserved known spatial dimensions and inferred IMAGE versus MASK from empty tensor layout when dual-type graph socket metadata is ambiguous.
- Fixed chained-Kiri preview routing so an upstream node's combined/selected-file canvas preview cannot masquerade as an empty A/B output slot. Stale comparison previews are cleared when all connected sources are empty.
- Added clipboard copy/paste integration tests and regressions for empty VAE Decode, bypassed branches, chained Kiri A/B outputs, mixed IMAGE/MASK values, and selected-file fallback interactions.

## 0.5.12
- Replaced the combined prompt privacy preference with two independent, persistent right-click switches: **Save Positive Prompt** and **Save Negative Prompt**. Existing v0.5.11 `save_prompts` settings migrate to both new switches unless an explicit branch setting is already present.
- Prompt suppression is branch-specific across saved A metadata, B runtime metadata, the ComfyUI API prompt, full canvas workflow widget values, and every platform parameter adapter. Shared prompt nodes remain only when at least one branch that uses them is allowed.
- Prompt privacy changes only prompt text. KSampler steps/CFG/sampler/scheduler/seed/denoise, checkpoint and VAE names/hashes, LoRA names/strengths/hashes, and other model-resource provenance remain intact.
- Fixed settings-only A1111 parameters: an omitted positive prompt is represented as an empty prompt section followed by a `Negative prompt:` boundary, so ComfyUI can import a usable basic sampler graph without turning the settings line into positive prompt text. The metadata reader now also recognizes `Steps:` at the beginning of older settings-only strings.
- Improved parameter parsing for Kiri's checkpoint/model labels and LoRA numbered name/strength/hash fields, keeping generation settings and resource details visible when prompt text is disabled.
- Updated app/site adapter export and sidecar generation to independently honor positive/negative prompt preferences.
- Added regression tests for each prompt toggle independently, backward compatibility with v0.5.11 settings, platform JSON payloads, blank-prompt ComfyUI parameter parsing, and retained sampler/checkpoint/LoRA data.

## 0.5.11
- Added three independent, persistent right-click → Metadata Saving preferences: **Save Positive/Negative Prompts**, **Save Node Names (Nodes Used)**, and **Other-Platform Metadata**. These apply independently of **Embed Complete Workflow**.
- Prompt saving can be disabled for the saved Image A record and Image B runtime metadata; platform parameter exports also omit prompt text. When full workflow embedding remains enabled, prompt widget values are blanked while graph topology is retained.
- The optional **Kiri Nodes Used** record is plain text only and contains custom display titles when available, otherwise node implementation names. Disabling it suppresses this standalone list; complete workflow embedding still includes names intrinsically present in the graph.
- Other-platform metadata can remain enabled while ComfyUI workflow embedding is OFF. Such app/site JSON is projected strictly from generation parameters and model/resource provenance, excluding ComfyUI node objects, IDs, class types, input dictionaries, links and workflow arrays.
- Other-platform metadata can be disabled separately, returning to the plain A1111-style parameter profile and suppressing platform-specific records/companion files.
- Updated metadata editor handling so workflow-OFF images cannot regain ComfyUI graph payloads; preserve NovelAI's exact `Comment` keyword while rejecting unrelated lowercase `comment` fields.
- Updated README documentation and expanded tests for all three independent preferences and platform-only metadata without ComfyUI graph serialization.

## 0.5.10
- Replaced the previous partial workflow-OFF exclusion with a strict text-only mode. OFF now writes only plain A1111-style `parameters`, a newline-separated plain-text **Kiri Nodes Used** list (custom node display titles where available), and a plain-text mode marker. It writes no JSON chunks of any kind, no node IDs/inputs/link arrays, no platform adapter records, no `prompt`/`workflow`/Kiri graph records, no custom JSON, no Draw Things EXIF/XMP JSON, no NovelAI stealth payload, and no `.kiri.json` or Easy Diffusion JSON companion files.
- In OFF mode, even if a JSON-based profile is selected in Compatibility Export Settings, the image uses the plain A1111-style text profile. Full structured Kiri metadata, all selected local-app/site compatibility exports, and optional companion JSON files remain available with **Embed Complete Workflow: ON**.
- Updated the metadata reader to expose the OFF-mode node-name list under **ComfyUI Nodes** without interpreting it as graph data. Updated **Edit Metadata…** so saving changes to an OFF-mode image cannot reintroduce JSON, graph IDs, platform JSON, or other structured payloads.
- Consolidated positive and negative prompt inputs under one **Prompts** category in **Edit Metadata…**, removing the repeated **Negative Prompt** category/field heading.
- Added regression coverage for OFF mode with every export profile, all platform JSON switches, custom metadata, `extra_pnginfo`, sidecars and NovelAI/Draw Things outputs enabled; verified the exact two/three text-chunk allow-list, custom display-name-only node list, absent JSON/graph/sidecars, and text-only edit round trip.

## 0.5.9
- Fixed undefined checkpoint/LoRA references in the universal interchange manifest that could interrupt metadata saves; the adapter now uses the matching release version.
- Expanded export profiles: Fooocus-native JSON can own the PNG `parameters` field, and Easy Diffusion task JSON is written as a matching `.easy-diffusion.json` companion by default for drag-and-drop parameter loading.
- Improved SwarmUI envelope fields and stopped forwarding whole-file SHA-256 values as SwarmUI tensor hashes. InvokeAI exports now include portable scheduler/strength fields while omitting fake catalog identifiers and host-version claims.
- Added Draw Things XMP (`XML:com.adobe.xmp` PNG iTXt) with `dc:description` and structured `exif:UserComment`, alongside the EXIF UserComment record. Improved XMP RDF parsing so nested `rdf:Alt` / `rdf:li` values retain their parent property semantics; standalone EXIF UserComment JSON is also parsed.
- Added NovelAI `stealth_pngcomp` alpha-channel metadata writing when the image has enough pixels, in addition to Description/Comment text fields. The payload is unsigned and does not include optional NovelAI FEC/signature data.
- Expanded reader normalization for Easy Diffusion task envelopes, Fooocus schemes, Draw Things JSON, PixAI API / Tensor.Art TAMS handoffs, Civitai model-version resource arrays, and provider-native model identifiers.
- Added structured PixAI v2 and Tensor.Art TAMS request payloads using only explicitly captured native provider IDs. SeaArt and TensorHub / Cozy Creator remain clearly marked portable handoffs rather than unsupported claims of native one-click image import. Fooocus and Naifu also receive separately named, copyable JSON handoff chunks in the universal bundle; Naifu is not misrepresented as a verified native PNG importer.
- Added one-click **Copy JSON** actions for JSON, handoff and interoperability records in **View All Metadata…** so request payloads can be pasted into the target app/API form.
- Added regression coverage for each new export profile, XMP and NovelAI encoding, Civitai ID extraction, native provider-ID handling, handoff chunk round trips, and a real PNG+Easy Diffusion sidecar save/read round trip. Retained the v0.5.7 workflow-off exclusion and v0.5.8 long-key/value wrapping fixes.

## 0.5.8
- Fixed long metadata field names overflowing into adjacent values in **View All Metadata…**. Both grid columns can now shrink independently, and labels and values wrap long dotted, underscored, and otherwise unbroken strings within their own cells.
- Kept the viewer scrollable and width-constrained on narrow layouts; added regression coverage using long platform-handoff field names, SHA-256 keys, and long values.

## 0.5.7
- Hardened **Embed Complete Workflow: OFF** into a complete ComfyUI graph-payload exclusion. Kiri no longer writes the ComfyUI `prompt` API-graph chunk when OFF; previously only the full visual `workflow` chunk was suppressed, leaving a partial executable graph in PNG metadata.
- Suppressed recognized ComfyUI prompt/workflow aliases from passthrough metadata, removed nested workflow envelopes from custom metadata, and cleared the source canvas UUID when embedding is OFF. The opt-in ON path still writes the active prompt and full visual workflow as before.
- Preserved the Kiri-owned `kiri_metadata` node-settings record, including active upstream node inputs, so **Apply Generation Data to Node…**, prompt/negative-prompt extraction, sampler settings, resource hashes, and universal metadata exports continue to work without a ComfyUI-loadable graph chunk.
- Added regression coverage proving that OFF writes neither `prompt` nor `workflow`, does not expose an embedded workflow in Kiri's reader, and still permits generation-data import; ON continues to embed the full workflow with credential redaction.

## 0.5.6
- Added **Metadata Saving → Compatibility Export Settings…**. The compact settings dialog selects the mutually exclusive PNG `parameters` representation (A1111 / Forge / Civitai-compatible text or SwarmUI-native JSON), enables/disables InvokeAI and NovelAI metadata records, toggles the universal handoff bundle, and optionally writes a matching `.kiri.json` sidecar.
- Added a dedicated metadata-adapter module. Kiri keeps its own structured metadata as the canonical source and builds platform records from it instead of storing duplicated complete workflows.
- Default PNG exports now include portable `invokeai_metadata`, NovelAI-style `Description` / `Comment` / `Software` / `Generation time` fields, and one `kiri_interop` manifest, alongside the existing ComfyUI prompt, Kiri metadata and A1111-compatible parameters. Kiri deliberately does not fabricate InvokeAI workflow/graph records or NovelAI stealth-alpha signatures.
- Added optional SwarmUI-native `parameters` JSON. A1111/Forge/Civitai and SwarmUI use incompatible contents in the same PNG `parameters` chunk, so only one can be selected as the native value at a time. The universal manifest retains the normalized data for handoff to the other ecosystems.
- Added Civitai resource-version details to A1111-compatible parameters only when real Civitai version IDs are available from Kiri LoRA Loader sidecars. Civitai IDs are not reused as PixAI IDs.
- Added one compact handoff manifest with best-effort mappings for PixAI API, Tensor.Art, SeaArt, TensorHub / Cozy Creator, Easy Diffusion and Fooocus. Undocumented cloud-platform formats are explicitly marked as handoff/provenance data, not native one-click import guarantees.
- Added opt-in `.kiri.json` companion exports. The default remains OFF to avoid extra files; when enabled, each sidecar shares the saved PNG's base filename and contains the full Kiri record plus the portable handoff manifest. PNG metadata edits synchronize an existing Kiri sidecar when possible.
- Metadata editing now refreshes existing InvokeAI, NovelAI, SwarmUI and Kiri interchange representations when their corresponding records are present, while preserving unrelated comment/custom metadata and keeping pixel data unchanged.
- Added regression coverage for metadata adapter schemas, mutually exclusive A1111/SwarmUI writing, PixAI ID non-inference, InvokeAI/NovelAI chunks, universal handoff contents, sidecar opt-in and existing node/frontend behavior.

## 0.5.5
- Added **Apply Generation Data to Node…** to source-specific image menus when the image contains an executed graph with compatible node data.
- Added a destination-driven importer: choose the source node from the image metadata, then explicitly choose one current-workflow node of the exact same ComfyUI implementation type. Node IDs are displayed for reference only and are never used as cross-workflow identity.
- Previews the compatible same-name scalar widgets and lets users select which values to apply. Existing connections are protected: linked sockets, link arrays, object values, and unsupported widgets are never overwritten.
- Updates ordinary widgets through their native widget setter/callback and refreshes the Kiri LoRA Loader dynamic rows through its canonical `loras_json` restore hook after import.
- The metadata reader now exposes compact `generation_data` from Kiri's active graph metadata or, for other ComfyUI images, from the executed API `prompt` chunk. Optional embedded canvas workflow metadata contributes only readable titles and workflow UUID context.
- Added regression tests for different source/destination IDs, repeated same-type KSampler Advanced nodes, exact class-type matching, connected-input protection, selected-field transfer, Kiri LoRA row refresh, and Kiri/foreign ComfyUI metadata extraction.


## 0.5.4
- Reworked **Edit Metadata…** to present semantic sections instead of exposing the underlying PNG chunks as raw JSON fields. The editor now has **Prompt** (positive prompt only), **Negative Prompt**, and **Parameters** (all collected KSampler settings plus recorded image Width/Height).
- Removed the redundant raw **Kiri Metadata** editor category. Kiri's structured JSON and the original ComfyUI prompt graph remain internal metadata carriers; other custom PNG text chunks remain editable under **Additional Metadata**.
- Saving updates the structured Kiri prompt fields, sampler settings, recorded dimensions, and the interoperable A1111-style `parameters` string. The original ComfyUI graph JSON remains preserved as provenance; it is not presented as the positive prompt editor.
- Metadata inspection now prefers the edited Kiri positive/negative prompt pair over old per-node text values retained in the original execution graph, preventing stale prompt duplicates after a metadata-only edit.
- Added regression coverage for the semantic editor sections and a full PNG edit/read-back round trip, including prompt values, seed/steps/CFG, recorded dimensions, and preservation of unrelated metadata.

## 0.5.3
- Fixed prompt contamination from Kiri CheckPoint Loader selector values. Positive/negative prompt traversal now follows only text-bearing and conditioning inputs, deliberately excluding CLIP/model-loader links, so values such as `Auto (from checkpoint)` cannot be prepended to a prompt. Traversal still crosses ConditioningSetArea/conditioning chains and upstream StringMultiline/primitive text nodes. The loader bridge also recognizes `Auto (from checkpoint)` as a checkpoint-managed component selection.
- Renamed source file actions to **Open** and **Open File Location**; their operating-system behavior is unchanged.
- Added **Edit Metadata…** for PNG sources. It presents the existing embedded PNG text chunks as editable key/value rows with add/remove controls. Saving updates the original PNG in place by replacing its text metadata chunks; IDAT image data, ICC profiles, EXIF and every non-text PNG chunk are copied byte-for-byte.
- Metadata editing is intentionally limited to PNG; other image formats remain inspectable but are not rewritten. Temporary preview files cannot be edited.
- Added regression coverage for Kiri CheckPoint Loader Auto selector isolation, conditioning-node traversal, PNG text replacement, pixel preservation, and non-text chunk preservation.

## 0.5.2
- Fixed positive/negative prompt extraction for conditioning graphs that use connected string, multiline-text, primitive-text, and text-area nodes. Prompt collection now follows all active upstream links behind the sampler's positive and negative conditioning inputs and stores separate authoritative prompt strings instead of relying on generic node metadata or parsing the combined sampler-parameter text.
- Kept both **Copy Positive Prompt** and **Copy Negative Prompt** visible in every source menu; an action is disabled only when that branch has no prompt to copy. Prompt copy values are taken from the explicit branch fields before other metadata.
- Renamed the image file action to **Delete Image (Recycle Bin)**; its behavior remains a recoverable OS Recycle Bin/Trash move, never permanent deletion.
- Added visible **Save Path** and **File Name** controls. Save Path accepts a relative folder under ComfyUI/output or a complete absolute path to another folder; external files are referenced for metadata/actions with session-limited opaque tokens. File Name supports `[Seed]` / `[Gotten Seed]`, `[Steps]`, `[CFG]` / `[CFG Scale]`, `[Sampler]`, `[Scheduler]`, `[Denoise]`, `[Date]`, `[Time]`, `[Counter]`, and `[Batch]`. Existing files are never overwritten; collisions receive a numbered suffix.
- Added regression coverage for multiline string prompt connections, distinct positive/negative prompt metadata, exact filename token expansion, path containment, and non-overwriting output naming.

## 0.5.1
- Simplified each source submenu's metadata actions to **Copy Positive Prompt**, **Copy Negative Prompt**, **Copy Seed(s)**, and **Copy KSampler Settings**. All other metadata remains inspectable in a read-only **View All Metadata…** window instead of generating a long list of clipboard commands.
- Moved **Embed Complete Workflow: ON/OFF** to the top level of Kiri's right-click menu and made the Kiri menu available from the node body, including when no image is selected or previewed.
- Added execution-time metadata routing for connected generated tensors. Image A resolves to the PNG Kiri just saved when its upstream source has no existing image file; Image B receives its own active-upstream graph metadata without creating an extra saved output image.
- Retained the inline IMAGE/MASK preview update path and synchronized metadata only after execution payload processing, so VAE Decode outputs can appear in the Kiri preview and their generated metadata can be opened from the appropriate A/B source menu.
- Added Kiri loader integration: expands Kiri LoRA Loader's active `loras_json` rows into individual LoRA resources with model/CLIP strengths, SHA-256 and adjacent Civitai provenance sidecars; records trigger words and selected trigger LoRA; captures Kiri CheckPoint Loader checkpoint, encoder, and VAE selections (including Auto/checkpoint-managed selections).
- Added regression coverage for compact metadata actions, empty-node workflow toggle access, generated A/B VAE Decode metadata, and both Kiri loader metadata bridges.

## 0.5.0
- Added graph-aware metadata collection and PNG saving to ComfyUI's configured Output directory. Each explicit queue saves Kiri's primary A output; IMAGE batches become one PNG per item, and MASK inputs save as grayscale. B remains a comparison/passthrough source and is not separately saved. Tensor passthrough remains unchanged.
- Added active upstream graph traversal from Kiri's Image A input using ComfyUI's executed `PROMPT` graph. The collector follows connected inputs recursively and excludes unrelated branches. It also checks visual workflow mode-4 bypass markers, skips bypassed nodes that remain represented in a prompt, and rewrites their outgoing links through the matching connected input so the recorded active graph remains coherent.
- Added specialized extraction for sampler/guider settings, positive/negative text nodes, checkpoint/diffusion models, VAEs, text encoders, LoRAs and strengths, ControlNet, IPAdapter, CLIP Vision and upscalers. All serializable inputs for collected nodes are also retained in the structured `kiri_metadata` record for generic/custom node coverage.
- Added registered-model SHA-256 resolution and a process cache keyed by real path, size and modification time. Added conventional A1111-style `parameters` text alongside ComfyUI `prompt` and Kiri's structured JSON metadata. The `prompt` chunk is filtered to the active Image A ancestry; full visual workflow embedding is separate.
- Added a single optional `CUSTOM META` wildcard input. Its connected runtime value overrides the manually entered custom metadata. The manual multiline editor is opened from the right-click Metadata Saving submenu and does not add a permanent visible widget. Strings, numbers, booleans, lists and dictionaries are retained as JSON values; tensors and runtime objects are summarized safely.
- Added the sole metadata-saving toggle, **Embed Complete Workflow**, OFF by default. When enabled, the full visual ComfyUI workflow is embedded in addition to ordinary generation metadata. Credential-like keys are redacted from generic metadata and workflow content; ComfyUI's `--disable-metadata` option is respected.
- Preserved the v0.4.7 node identity, source-specific menus, embedded-workflow recovery, live upstream previews, A/B comparer interaction, preview input ownership, and dual IMAGE/MASK passthrough.
- Added regression tests for graph traversal, unrelated-branch filtering, custom metadata precedence, workflow opt-in, grayscale mask saving, model hashes, and hidden frontend controls.

## 0.4.7
- Added **Open Embedded Workflow** to each source-specific submenu (Image A, Image B, and Loaded Image), but only when that source image contains a structurally valid ComfyUI canvas workflow graph.
- Added embedded-workflow extraction for ComfyUI `workflow` metadata fields, including JSON-string and nested workflow envelopes. API-prompt-only JSON, InvokeAI workflow/graph records, and unrelated JSON are not misidentified as openable ComfyUI canvas workflows.
- Opens the recovered graph in a new unsaved workflow tab using `app.extensionManager.workflow.createNewTemporary()` and `openWorkflow()`. The current workflow tab and unsaved edits remain intact; the recovered graph is not automatically executed.
- Frontends without the workflow-tab API receive a clear message and are not allowed to fall back to replacing the current canvas.
- Added metadata-format and frontend runtime regression tests for workflow detection, menu visibility, source naming, tab creation, data isolation, and safe handling of unsupported frontends.


## 0.4.6
- Made the entire visible image-preview rectangle an owned Kiri interaction surface using a virtual hit-target returned by `node.getWidgetOnPos`; the virtual target is not added to `node.widgets`, so it creates no visible controls, extra widget rows, serialized values, or node-height changes.
- Clicking or dragging elsewhere inside the preview no longer starts LiteGraph's normal node-drag gesture. A simple click can still select Kiri, while a body drag is consumed without moving the node or the canvas.
- Kept the comparison handle as the only draggable control inside the preview; dragging the visible handle still updates the A/B divider through ComfyUI's CanvasPointer lifecycle.
- Added a matching handle-only / preview-surface interception path for older LiteGraph frontends, without changing node movement outside the preview.
- Added frontend runtime regression coverage for preview-body gesture capture, handle-only divider movement, preserved selection, outside-preview widget behavior, and no additional visible widgets.

## 0.4.5
- Fixed the A/B comparison interaction using ComfyUI's supported `CanvasPointer` widget API on modern frontends (`onPointerDown`, `onDrag`, `onDragEnd`, and `onDragCancel`).
- The divider now moves only when grabbing and dragging the visible center handle; clicks elsewhere in the preview no longer jump/nudge the divider and no invisible left/right arrow hit-zones are used.
- Added a circular, zoom-aware grab tolerance around the visible handle and preserved the initial pointer offset so grabbing its edge does not cause a jump.
- Added a passive `ew-resize` cursor when hovering the handle and retained a handle-only pointer-capture fallback for older LiteGraph frontends.
- Added frontend runtime regression coverage for handle-only hit testing, click-through outside the handle, smooth drag updates, pointer offset preservation, and drag cancellation.

## 0.4.4
- Added separate right-click submenus for Image A, Image B, and Loaded Image.
- Metadata reads are now source-specific: A/B follow their own connected graph input; Loaded Image continues to follow Kiri's filename selector.
- Added upstream file provenance tracing through standard Load Image nodes, Kiri-to-Kiri connections, unambiguous single-image pass-through/transform chains, and saved ComfyUI output image records.
- Each source submenu now targets its own file for Open in Default Image Viewer, Open Containing Folder, and Move to Recycle Bin, in addition to that source's individual metadata-copy actions.
- Connected sources with no recoverable file, or with multiple ambiguous ancestors, no longer silently borrow metadata/file actions from Kiri's selected fallback.
- Added safe metadata/file-action resolution for ComfyUI-managed input/output/temp folders, including subfolders, with absolute-path, traversal, and symlink-escape rejection.
- Added independent per-source metadata caches/request serials and regression tests for source-specific menus, Load Image tracing, pass-through-chain tracing, saved output records, and managed output paths.

## 0.4.3
- Reworked connected-input preview discovery to resolve the actual upstream node and output slot from each Kiri input link. Kiri now reads already-loaded upstream `node.imgs` immediately, without waiting for Kiri's own execution or a new generation.
- Added compatibility with ComfyUI's stored node preview URLs and node output image records for upstream nodes that expose previews through the frontend preview/output stores.
- The native selected-file image remains a fallback only. While either live input is connected, the native preview source is fed the connected preview or an empty image list while waiting; it can no longer show the selected file over an unavailable live source.
- Kept the Points Editor-style execution-payload preview as a fallback for upstream nodes that expose their image only when evaluated.
- Added proper visual handling for the built-in Load Image node's MASK output by deriving its inverted-alpha preview; the IMAGE/MASK tensors themselves still pass through unchanged.
- The preview updates when upstream Load Image selectors change, when links connect/disconnect, and when upstream preview records refresh. The existing wipe slider and compact preview area are retained, without A/B labels.
- Documented that ComfyUI stores uploaded image copies in its configured input folder and Kiri operates on that managed copy.

## 0.4.2
- Fixed the live preview ownership issue by feeding the connected single/combined preview image through the node's native `imgs` preview source instead of drawing a competing canvas foreground overlay. The built-in ComfyUI image-preview widget now draws the live source itself, so changing the filename selector cannot paint the selected-file image over the live comparison.
- The A/B comparison divider remains draggable, but all A/B corner labels and single-input labels have been removed to match rgthree Image Comparer.
- Added a guarded native preview override that preserves ComfyUI's asynchronously updated selected-file image array as the fallback and restores it when no live input preview is active.
- Image discovery now recursively scans the configured ComfyUI input directory and includes images in nested subfolders, using relative paths. The scan root is taken from `folder_paths.get_input_directory()`, so custom `--input-directory` locations on other drives are respected.
- Fixed Windows **Open Containing Folder** to call Explorer through `ShellExecuteW` with the complete `/select,"path"` argument, preserving spaces and alternate-drive paths.
- Added tests for nested-folder discovery/resolution and Windows Explorer argument quoting.


## 0.4.1
- Reworked live input previews to follow the KJNodes Points Editor transport pattern: the Python node converts connected tensors to JPEG/Base64 and returns separate one-element UI fields (`kiri_preview_a`, `kiri_preview_b`, and `kiri_preview_mode`), matching Points Editor’s `ui.bg_image: [base64]` contract.
- The frontend now reads the inline image data in `node.onExecuted` and displays it without writing temporary preview files or making a second `/view` request.
- Added compatibility handling for flattened and nested ComfyUI execution-message shapes.
- Kept the original A/B single-source and wipe-comparison behavior, dual IMAGE/MASK socket passthrough, and selected-file fallback.
- Bounded the display payload to a 1600-pixel maximum side; this affects only the preview, never the tensors passed downstream.

## 0.4.0

- Changed the existing two optional input slots to independent `IMAGE,MASK` multi-type sockets and made both outputs dual-type as well.
- Preserved the old input socket IDs (`image_in`, `mask_in`) for workflow connection compatibility; frontend labels are A and B, with outputs A and B.
- Connected inputs pass through to their paired outputs by object identity, without conversion. Unconnected outputs continue to use the selected file's IMAGE and MASK.
- Added immediate post-evaluation live input preview in the existing image-preview region. One connected input displays by itself; two inputs display a draggable A/B wipe comparison.
- The node is marked as an output node to keep the preview refreshing when nothing is connected downstream.
- Live preview PNGs are written to ComfyUI's temporary directory and replaced on later executions; tensor outputs remain untouched.

## 0.3.1
- Changed the preview context-menu delete action to **Move Image to Recycle Bin**.
- Removed the confirmation dialog; the action runs immediately when selected.
- Files are moved to the OS Recycle Bin/Trash using `send2trash` when installed.
- Added a Windows-native Recycle Bin fallback for portable ComfyUI environments.
- On macOS/Linux, missing `send2trash` produces an actionable error instead of permanently deleting the file.
- The image selector and metadata are refreshed after the file is moved.

## 0.3.0

- Added optional `image_in` (`IMAGE`) and `mask_in` (`MASK`) passthrough sockets. Each independently overrides its matching selected-file output; both connected inputs are returned unchanged without decoding the selected image.
- Preserved the compact node face: no additional permanent widgets, settings, or metadata panel.
- Added preview-only context-menu actions to open the selected image in the host OS default image viewer, reveal/open its containing folder, and delete the selected image after confirmation.
- Added guarded file-action API endpoints restricted to ComfyUI's input directory; path traversal and symlink escapes are rejected.
- Added image-list refresh and next-image selection after deleting the active image.
- Added NovelAI `stealth_pngcomp` alpha-channel metadata decoding and legacy metadata normalization.
- Added generic XMP/RDF field extraction (including Draw Things-style metadata).
- Expanded Easy Diffusion, Fooocus/Fooocus-MRE, Naifu, A1111-compatible, and legacy InvokeAI parsing.
- Added regression tests for metadata formats and IMAGE/MASK passthrough behavior.

## 0.2.0

- Added structured parsing for SwarmUI `sui_image_params`, `sui_extra_data`, and `sui_models` metadata.
- Added structured parsing for InvokeAI `invokeai_metadata`, `invokeai_workflow`, and `invokeai_graph` records.
- Improved InvokeAI ModelIdentifier handling to prefer the display name over the internal key; added model/LoRA hash and weight extraction.
- Added SwarmUI original prompt and generation date normalization.
- Replaced the extra entries in ComfyUI's ordinary node context menu with a preview-boundary menu override: right-clicking the displayed image opens only Kiri actions; other right-clicks remain native ComfyUI menus.
- Improved metadata response handling and fallback menu cleanup.

## 0.1.0

- Initial Load Image-style node, automatic metadata endpoint, and individual copy actions.
