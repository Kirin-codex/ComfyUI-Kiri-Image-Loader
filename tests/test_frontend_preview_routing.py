from pathlib import Path
import unittest


class FrontendPreviewRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (Path(__file__).resolve().parents[1] / "web" / "kiri_image_loader.js").read_text(encoding="utf-8")

    def test_connected_inputs_resolve_the_actual_upstream_graph_link(self):
        self.assertIn("function findInputLink(node, input)", self.source)
        self.assertIn("node?.getInputLink?.(inputIndex)", self.source)
        self.assertIn("link.origin_id", self.source)
        self.assertIn("graph?.getNodeById?.(link.origin_id)", self.source)

    def test_already_loaded_upstream_image_is_read_without_kiri_execution(self):
        self.assertIn("function getSourceNodePreviewImage(sourceNode, sourceSlot = 0, visited = new Set(), depth = 0)", self.source)
        self.assertIn("Array.isArray(sourceNode.imgs)", self.source)
        self.assertIn("app.nodePreviewImages?.[id]", self.source)
        self.assertIn("sourceNode.images", self.source)
        self.assertIn("app.nodeOutputs?.[id]?.images", self.source)
        self.assertIn("function syncLivePreviewFromUpstream(node)", self.source)

    def test_upstream_preview_sync_runs_before_native_node_drawing(self):
        sync_call = self.source.index("syncLivePreviewFromUpstream(node);", self.source.index("canvas.drawNode = function"))
        native_draw = self.source.index("return originalDrawNode.call(this, node, ctx, ...args);", sync_call)
        self.assertLess(sync_call, native_draw)

    def test_connected_but_unavailable_sources_use_loaded_image_only_as_auto_final_fallback(self):
        self.assertIn("function getVisiblePreviewImages(node)", self.source)
        self.assertIn("if (imageA) return [imageA];", self.source)
        self.assertIn("if (imageB) return [imageB];", self.source)
        self.assertIn("return loaded;", self.source)
        self.assertIn('hasConnectedImageInput(node) || mode !== "auto"', self.source)

    def test_modern_preview_widget_receives_the_live_preview_argument(self):
        self.assertIn('constructorName.includes("ImagePreviewWidget")', self.source)
        self.assertIn("previewImages: images", self.source)

    def test_native_file_preview_is_preserved_as_disconnect_fallback(self):
        self.assertIn("const ownDescriptor = Object.getOwnPropertyDescriptor(node, \"imgs\")", self.source)
        self.assertIn("const readFallback = () =>", self.source)
        self.assertIn("return readFallback();", self.source)

    def test_load_image_mask_output_is_shown_as_inverted_alpha(self):
        self.assertIn("function makeLoadImageMaskPreview(image)", self.source)
        self.assertIn("const value = 255 - pixels[i + 3];", self.source)

    def test_connection_changes_clear_stale_preview_and_resolve_again(self):
        self.assertIn("const originalOnConnectionsChange = node.onConnectionsChange;", self.source)
        self.assertIn("livePreviewByNode.delete(this);", self.source)
        self.assertIn("queueMicrotask(() => {\n        syncLivePreviewFromUpstream(this);", self.source)

    def test_metadata_is_resolved_independently_for_a_b_and_loaded_image(self):
        self.assertIn("function connectedImageDescriptor(node, sourceKey)", self.source)
        self.assertIn("function traceImageFileSource(sourceNode, sourceSlot = 0, visited = new Set(), context = { remaining: MAX_IMAGE_TRACE_NODES })", self.source)
        self.assertIn('sourceMenuEntry(node, "image_a", "Image A")', self.source)
        self.assertIn('sourceMenuEntry(node, "image_b", "Image B")', self.source)
        self.assertIn('sourceMenuEntry(node, "loaded", "Loaded Image")', self.source)
        self.assertIn("function readMetadataForSource(node, sourceKey, descriptor)", self.source)

    def test_file_actions_use_each_source_descriptor_and_preserve_source_type(self):
        self.assertIn("filename: descriptor.filename", self.source)
        self.assertIn("type: descriptor.type || \"input\"", self.source)
        self.assertIn("subfolder: descriptor.subfolder || \"\"", self.source)
        self.assertIn('descriptor.type !== "temp"', self.source)

    def test_source_menus_have_individual_metadata_and_file_actions(self):
        self.assertIn("function sourceSubmenuItems(node, sourceKey, label)", self.source)
        self.assertIn('content: "Open"', self.source)
        self.assertIn('content: "Open File Location"', self.source)
        self.assertIn('content: "Delete Image (Recycle Bin)"', self.source)
        self.assertIn('content: "Edit Metadata…"', self.source)
        self.assertIn('action: "edit_metadata"', self.source)
        self.assertIn("...metadataCopyItems(data)", self.source)
        self.assertIn("has_submenu: true", self.source)

    def test_metadata_resolution_traces_saved_output_records_and_single_image_ancestors(self):
        self.assertIn("function descriptorFromOutputRecord(record, originNode = null)", self.source)
        self.assertIn("app.nodeOutputs?.[id]?.images", self.source)
        self.assertIn("function imageInputSources(node)", self.source)
        self.assertIn("This image combines multiple upstream files", self.source)

    def test_comparison_has_no_a_b_corner_labels_or_overlay_draw_hook(self):
        self.assertIn("Clean comparer-style divider: deliberately no A/B labels or badges.", self.source)
        self.assertNotIn("drawConnectedPreviewAfterNode", self.source)
        self.assertNotIn("onDrawForeground =", self.source)

    def test_subgraph_boundary_inputs_do_not_call_output_resolver_as_input_resolver(self):
        self.assertIn("function isSubgraphBoundaryNode(node)", self.source)
        self.assertIn("if (!isSubgraphBoundaryNode(node) && inputIndex >= 0)", self.source)
        self.assertIn("function resolveSubgraphOutputSource(subgraphNode, outputSlot)", self.source)
        self.assertIn("subgraphNode.resolveSubgraphOutputLink(slot)", self.source)

    def test_nested_subgraph_outputs_are_resolved_by_exact_slot_and_graph_scoped_id(self):
        self.assertIn("getSourceNodePreviewImage(sourceNode, sourceSlot = 0, visited = new Set(), depth = 0)", self.source)
        self.assertIn("getNodePreviewLookupKeys(sourceNode)", self.source)
        self.assertIn("${String(graphId)}:${String(id)}", self.source)
        self.assertIn("visited.has(sourceNode)", self.source)

    def test_canvas_preview_integration_errors_are_isolated_from_host_rendering(self):
        draw_block = self.source[self.source.index("canvas.drawNode = function"):self.source.index("if (patchedCanvases.has(canvas)", self.source.index("canvas.drawNode = function"))]
        self.assertIn("try {", draw_block)
        self.assertIn("catch (error)", draw_block)
        self.assertIn("return originalDrawNode.call(this, node, ctx, ...args);", draw_block)

    def test_large_upscaled_images_use_bounded_display_only_comparison_canvases(self):
        self.assertIn("MAX_COMPARISON_PREVIEW_PIXELS = 2_000_000", self.source)
        self.assertIn("MAX_COMPARISON_PREVIEW_EDGE = 2048", self.source)
        self.assertIn("const sourceWidth = Math.max(1, Math.round(Number(base.naturalWidth || base.width)))", self.source)


if __name__ == "__main__":
    unittest.main()
