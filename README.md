# Kiri Image Loader

A compact ComfyUI image loader with automatic embedded-metadata inspection, individual right-click copy actions, dual IMAGE/MASK passthrough/comparison sockets, a live in-node wipe preview, and preview-scoped file actions.

## Features

- Familiar Load Image-style frontend with the normal image selector/preview and `IMAGE` / `MASK` outputs.
- Two optional dual-type input sockets, displayed as **A** and **B**. Each accepts either `IMAGE` or `MASK` and has a matching dual-type output. The exact tensor object is passed through unchanged; no image/mask conversion occurs.
- The existing socket IDs `image_in` and `mask_in` are retained so workflows made with earlier Kiri versions keep their connections. Their visible labels are now **A** and **B**.
- Output A defaults to the selected file's IMAGE when input A is unconnected. Output B defaults to the selected file's MASK when input B is unconnected. Each incoming socket independently overrides its paired output. When both are connected, the selected file is not decoded for execution.
- The four sockets are connection points only; they add no controls, widgets, metadata panel, or node height.
- Reads metadata automatically and independently for **Image A**, **Image B**, and the node's **Loaded Image**. Connected image sources are traced back to their originating ComfyUI Load Image file or an unambiguous saved output image when possible. No queue/generation run is required for an already-loaded source.
- **Right-click directly on the rendered image preview** to open a Kiri-only context menu with separate **Image A**, **Image B**, and **Loaded Image** submenus. Right-click anywhere else on the node retains the regular ComfyUI context menu.
- Each source submenu has its own individual `Copy ...` metadata actions. No Copy All action; unavailable or ambiguous original-file metadata is reported instead of silently using the wrong image.
- Each source submenu also has its own Open in Default Image Viewer, Open Containing Folder, and immediate Recycle Bin/Trash action when a real file can be resolved.
- After moving an image to the Recycle Bin/Trash, the image selector refreshes and selects the first remaining file (or clears the preview if none remain).
- Live input preview: Kiri resolves each connected input link back to its upstream node and immediately uses an already-loaded frontend preview when available (including the built-in Load Image preview). When both are available, a draggable vertical wipe compares A (left) and B (right), with no A/B labels. Drag only the visible center handle to move the divider; clicking elsewhere in the preview does not nudge it. For upstream nodes that expose previews through ComfyUI's stored output/preview records, Kiri also reads those records. Nodes that expose no frontend preview still use the KJNodes Points Editor-style inline Base64 execution payload as a fallback; no temporary preview files are written and output tensors are never altered.
- When either live input is connected, Kiri takes ownership of the native preview source (`node.imgs` and, on newer frontends, the image-preview widget's `previewImages` argument). The selected-file image cannot paint over the live source. While no upstream preview is available yet, the preview is intentionally blank rather than misleadingly showing the selected file. Disconnecting the inputs restores the most recently selected file preview.
- ComfyUI's standard upload button copies uploaded images into its configured `input` folder. Kiri uses that ComfyUI-managed copy for metadata reading, opening, revealing, and Recycle Bin actions; it does not make another copy.
- The image selector recursively discovers supported image files under ComfyUI's configured input directory, including nested subfolders. It uses `folder_paths.get_input_directory()` (including custom `--input-directory` values), not a hard-coded Documents or portable-install path.
- The node is an output node so its preview refreshes when the workflow is queued even when neither A nor B output is connected onward.
- Unknown and custom scalar metadata fields remain individually copyable; structured workflow records also have individual raw-JSON copy actions.
- File actions are restricted to image files inside ComfyUI-managed input/output/temp directories, including nested subfolders; path traversal and symlink escapes are rejected. The node's own image selector remains restricted to the configured input directory.

## Install

1. Stop ComfyUI.
2. Back up or remove the previous `ComfyUI/custom_nodes/KiriImageLoader` folder.
3. Extract the `KiriImageLoader` folder from this ZIP into `ComfyUI/custom_nodes/`.
4. Start ComfyUI again.
5. Add **Kiri Image Loader** from `Kiri/Image`.
6. Connect any IMAGE or MASK output to input A and/or input B. Their paired outputs pass the exact connected values onward. Drag across the existing preview to move the A/B comparison divider.

On Windows, the built-in Recycle Bin fallback works without extra packages. On macOS/Linux, install the listed dependency in ComfyUI's Python environment with `python -m pip install -r requirements.txt`.

## Metadata coverage

The reader inspects embedded metadata exposed by Pillow in PNG, JPEG, WebP, TIFF, BMP, and GIF images. Coverage now includes:

- **A1111-compatible tools:** A1111, common Forge/SD.Next-style `parameters` strings, prompts, negative prompts, and generation settings.
- **SwarmUI:** `sui_image_params`, `sui_extra_data`, and `sui_models`, including model/LoRA names, weights, hashes, original prompts, and generation dates.
- **InvokeAI:** current `invokeai_metadata`, `invokeai_workflow`, and `invokeai_graph` records, plus legacy `Dream` records when present.
- **Easy Diffusion:** JSON-style prompt, negative prompt, seed, inference steps, guidance scale, sampler, model identifier, and other scalar fields.
- **Fooocus / Fooocus-MRE:** prompt/settings fields, Fooocus styles, V2 expansion, performance, sharpness, ADM guidance, adaptive CFG, and unknown fields.
- **NovelAI:** legacy metadata and the `stealth_pngcomp` alpha-channel payload used by stealth PNG metadata. The parser reads the compressed JSON payload and its prompt/settings fields; it does not validate NovelAI signatures or perform forward-error correction.
- **Draw Things and other XMP writers:** XMP/RDF XML properties are parsed for recognized generation fields and exposed as individual XMP copy actions. Unknown XMP values remain available too.
- **Naifu:** common prompt/negative-prompt and generation fields in JSON-style records.
- **ComfyUI:** `prompt` and `workflow` JSON, common sampler/checkpoint/LoRA/text-encode node inputs, and workflow widget values.
- **Generic image metadata:** EXIF text and otherwise-unrecognized scalar or structured metadata exposed by Pillow.

Compatibility depends on metadata actually being embedded in the file. Re-encoding, social sites, or chat services may strip it. ComfyUI workflows can contain many inactive branches and custom nodes; this reader surfaces common node fields and raw workflow JSON but cannot infer every custom node's semantics. XMP support is generic XML-property extraction rather than a complete implementation of every vendor's XMP schema. Video containers are not supported.

## Right-click menu behavior

The extension intercepts ComfyUI's canvas context-menu call only when the graph coordinate falls inside the currently rendered image preview rectangle. The Kiri menu contains three separate source submenus: **Image A**, **Image B**, and **Loaded Image**. Each submenu lists its own metadata-copy actions and file actions for that exact source. For connected inputs, Kiri follows the upstream graph link; pass-through/transform chains inherit metadata only when there is exactly one unambiguous image-file ancestor. Saved ComfyUI output records can also be inspected from their managed output/temp folder. If a connected tensor has no recoverable image file, preview and passthrough still work, but Kiri does not invent a filename or borrow metadata from the selected fallback image. The original ComfyUI context-menu implementation is called unchanged everywhere outside the preview rectangle.

## File-action behavior

- **Open in Default Image Viewer:** launches the exact source file using the operating system's default file association on the machine running ComfyUI.
- **Open Containing Folder:** Windows Explorer selects the exact source file; macOS Finder reveals it; Linux opens the containing directory (file selection is not standardized across Linux file managers).
- **Move Image to Recycle Bin:** immediately moves the exact source file to the operating system Recycle Bin/Trash without confirmation. When the source is a standard Load Image node in the input directory, its selector refreshes to the next available image. This is recoverable through the OS trash interface.