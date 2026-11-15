import base64
import importlib
import json
import io
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
from PIL import Image, PngImagePlugin


class NodePassthroughTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.input_dir = Path(cls.temp.name)
        folder_paths = types.ModuleType("folder_paths")
        cls.temp_preview_dir = cls.input_dir / "temp"
        cls.output_dir = cls.input_dir / "output"
        cls.temp_preview_dir.mkdir()
        cls.output_dir.mkdir()
        folder_paths.get_input_directory = lambda: str(cls.input_dir)
        folder_paths.get_output_directory = lambda: str(cls.output_dir)
        folder_paths.get_annotated_filepath = lambda filename: str(cls.input_dir / filename)
        folder_paths.get_temp_directory = lambda: str(cls.temp_preview_dir)
        folder_paths.get_full_path = lambda category, filename: None

        def get_save_image_path(prefix, output_dir, width, height):
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            existing = list(Path(output_dir).glob(f"{prefix}_*_.png"))
            counters = []
            for item in existing:
                try:
                    counters.append(int(item.stem.rsplit("_", 2)[1]))
                except Exception:
                    pass
            return str(output_dir), prefix, (max(counters) + 1 if counters else 1), "", prefix

        folder_paths.get_save_image_path = get_save_image_path
        sys.modules["folder_paths"] = folder_paths

        class FakeRoutes:
            def __init__(self):
                self.handlers = {}

            def get(self, route_path):
                def register(function):
                    self.handlers[("GET", route_path)] = function
                    return function
                return register

            def post(self, route_path):
                def register(function):
                    self.handlers[("POST", route_path)] = function
                    return function
                return register

        fake_server = types.ModuleType("server")
        cls.routes = FakeRoutes()
        fake_server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=cls.routes))
        sys.modules["server"] = fake_server

        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        cls.package = importlib.import_module("KiriImageLoader")
        cls.node_class = cls.package.KiriImageLoader

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("KiriImageLoader", None)
        sys.modules.pop("folder_paths", None)
        sys.modules.pop("server", None)
        try:
            sys.path.remove(str(Path(__file__).resolve().parents[2]))
        except ValueError:
            pass
        cls.temp.cleanup()

    def load_node_with_saving(self, *args, **kwargs):
        """Legacy test helper: opt existing save-path tests into explicit saving."""
        if len(args) <= 10:
            kwargs.setdefault("save_path", str(self.output_dir))
        if len(args) <= 11:
            kwargs.setdefault("filename_template", "Kirin Image [Seed]")
        return self.node_class().load_image(*args, **kwargs)

    def setUp(self):
        rgba = Image.new("RGBA", (3, 2), (220, 30, 50, 128))
        rgba.save(self.input_dir / "disk.png")

    def test_embedded_workflow_excludes_transient_kiri_preview_snapshot(self):
        workflow = {
            "nodes": [
                {"id": 9, "type": "KiriImageLoader", "properties": {
                    "kiri_live_preview": {"a": {"data": "base64-preview"}, "b": None},
                    "keep_me": "normal node setting",
                }},
                {"id": 10, "type": "LoadImage", "properties": {"other": True}},
            ],
            "links": [],
        }
        embedded = self.package._workflow_for_embedding({"workflow": workflow}, True)
        self.assertNotIn("kiri_live_preview", embedded["nodes"][0]["properties"])
        self.assertEqual(embedded["nodes"][0]["properties"]["keep_me"], "normal node setting")
        self.assertEqual(embedded["nodes"][1]["properties"], {"other": True})
        self.assertIn("kiri_live_preview", workflow["nodes"][0]["properties"],
                      "Sanitizing PNG metadata must not mutate the live workflow.")

    def test_metadata_saver_inputs_are_hidden_and_custom_input_is_wildcard(self):
        inputs = self.node_class.INPUT_TYPES()
        self.assertEqual(inputs["required"]["save_path"][0], "STRING")
        self.assertEqual(inputs["required"]["save_path"][1]["default"], "")
        self.assertEqual(inputs["required"]["filename_template"][0], "STRING")
        self.assertEqual(inputs["required"]["filename_template"][1]["default"], "")
        self.assertIn("[Gotten Seed]", inputs["required"]["filename_template"][1]["tooltip"])
        self.assertEqual(inputs["required"]["custom_metadata_text"][0], "STRING")
        self.assertTrue(inputs["required"]["custom_metadata_text"][1]["hidden"])
        self.assertEqual(inputs["required"]["embed_workflow"][0], "BOOLEAN")
        self.assertTrue(inputs["required"]["embed_workflow"][1]["hidden"])
        self.assertEqual(inputs["required"]["metadata_export_settings"][0], "STRING")
        self.assertTrue(inputs["required"]["metadata_export_settings"][1]["hidden"])
        self.assertEqual(inputs["optional"]["custom_metadata"][0], "*")
        self.assertTrue(inputs["optional"]["custom_metadata"][1]["forceInput"])
        self.assertTrue(self.node_class.OUTPUT_NODE)

    def test_structured_metadata_editor_updates_prompt_negative_parameters_and_dimensions(self):
        path = self.output_dir / "editor-structured.png"
        kiri_record = {
            "schema_version": 1,
            "collector": "Kiri Image Loader",
            "graph": {
                "prompts": {"positive": "old positive", "negative": "old negative"},
                "samplers": [{"class_type": "KSampler", "settings": {
                    "seed": 123456, "steps": 30, "cfg": 6.5,
                    "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
                }}],
                "text_encoders_and_prompts": [
                    {"node_id": "10", "branch": "positive", "values": {"text": "old positive"}},
                    {"node_id": "11", "branch": "negative", "values": {"text": "old negative"}},
                ],
            },
        }
        comfy_prompt = {
            "10": {"class_type": "CLIPTextEncode", "inputs": {"text": "old positive"}},
            "11": {"class_type": "CLIPTextEncode", "inputs": {"text": "old negative"}},
        }
        png_info = PngImagePlugin.PngInfo()
        png_info.add_text("prompt", json.dumps(comfy_prompt))
        png_info.add_text("kiri_metadata", json.dumps(kiri_record))
        png_info.add_text("parameters", "old positive\\nNegative prompt: old negative\\nSteps: 30, Seed: 123456")
        png_info.add_text("comment", "keep this custom note")
        Image.new("RGB", (1024, 768), (10, 20, 30)).save(path, pnginfo=png_info)

        original_entries = [
            {"keyword": "prompt", "value": json.dumps(comfy_prompt)},
            {"keyword": "kiri_metadata", "value": json.dumps(kiri_record)},
            {"keyword": "parameters", "value": "old positive\\nNegative prompt: old negative\\nSteps: 30, Seed: 123456"},
            {"keyword": "comment", "value": "keep this custom note"},
        ]
        editor_values = {
            "positive_prompt": "new positive prompt",
            "negative_prompt": "new negative prompt",
            "parameters": [
                {"kind": "sampler", "sampler_index": 0, "key": "seed", "value": "987654"},
                {"kind": "sampler", "sampler_index": 0, "key": "steps", "value": "42"},
                {"kind": "sampler", "sampler_index": 0, "key": "cfg", "value": "7.25"},
            ],
            "image_width": "1536",
            "image_height": "1024",
        }
        self.package._rewrite_png_text_metadata(str(path), original_entries, editor_values)

        with Image.open(path) as image:
            self.assertEqual(image.size, (1024, 768), "Editing recorded dimensions must not resize/re-encode the image.")
            edited_kiri = json.loads(image.info["kiri_metadata"])
            self.assertEqual(edited_kiri["graph"]["prompts"], {
                "positive": "new positive prompt", "negative": "new negative prompt",
            })
            settings = edited_kiri["graph"]["samplers"][0]["settings"]
            self.assertEqual(settings["seed"], 987654)
            self.assertEqual(settings["steps"], 42)
            self.assertEqual(settings["cfg"], 7.25)
            self.assertEqual(edited_kiri["graph"]["image_dimensions"], {"width": 1536, "height": 1024})
            self.assertIn("new positive prompt", image.info["parameters"])
            self.assertIn("Negative prompt: new negative prompt", image.info["parameters"])
            self.assertIn("Size: 1536x1024", image.info["parameters"])
            self.assertEqual(image.info["comment"], "keep this custom note")

        inspected = self.package.inspect_image(str(path))
        values = {item["label"]: item["value"] for item in inspected["copy_items"]}
        self.assertEqual(values["Positive Prompt"], "new positive prompt")
        self.assertEqual(values["Negative Prompt"], "new negative prompt")
        self.assertNotIn("old positive", [item["value"] for item in inspected["copy_items"] if item["category"] == "Prompts"])

        # Explicitly clearing either prompt must not cause the original
        # per-node text values in the retained graph to reappear as fallbacks.
        self.package._rewrite_png_text_metadata(str(path), inspected["editable_metadata"], {
            "positive_prompt": "", "negative_prompt": "", "parameters": [],
            "image_width": "1536", "image_height": "1024",
        })
        cleared = self.package.inspect_image(str(path))
        self.assertFalse([item for item in cleared["copy_items"] if item["category"] == "Prompts"])

    def test_kiri_saves_primary_image_and_preserves_tensor_passthrough(self):
        image_value = torch.rand((1, 6, 8, 3))
        result = self.load_node_with_saving("disk.png", image_value, None, "save-basic", embed_workflow=True)
        output_a, _output_b, _output_c = result["result"]
        self.assertIs(output_a, image_value)
        saved_record = result["ui"]["kiri_saved_files"][0][0]
        saved_path = self.output_dir / saved_record["filename"]
        self.assertTrue(saved_path.is_file(), "Kiri should save its primary A output to ComfyUI output.")
        with Image.open(saved_path) as image:
            self.assertEqual(image.size, (8, 6))
            self.assertIn("kiri_metadata", image.info)
        self.assertIn("kiri_saved_files", result["ui"])

    def test_default_metadata_exports_include_invokeai_novelai_and_universal_bundle(self):
        tensor = torch.rand((1, 6, 8, 3))
        result = self.load_node_with_saving(
            "disk.png", image_in=tensor, unique_id="interop-default", prompt={}, extra_pnginfo={}, embed_workflow=True,
        )
        filename = result["ui"]["kiri_saved_files"][0][0]["filename"]
        with Image.open(self.output_dir / filename) as image:
            self.assertIn("parameters", image.info)
            self.assertIn("Size: 8x6", image.info["parameters"])
            self.assertIn("invokeai_metadata", image.info)
            self.assertIn("Comment", image.info)
            self.assertIn("Software", image.info)
            self.assertIn("kiri_interop", image.info)
            invoke = json.loads(image.info["invokeai_metadata"])
            self.assertEqual((invoke["width"], invoke["height"]), (8, 6))
            bundle = json.loads(image.info["kiri_interop"])
            self.assertIn("pixai_api", bundle["platform_handoff"])

    def test_swarmui_profile_owns_parameters_chunk_without_overwriting_native_metadata(self):
        tensor = torch.rand((1, 6, 8, 3))
        settings = json.dumps({"parameters_format": "swarmui"})
        result = self.load_node_with_saving(
            "disk.png", image_in=tensor, unique_id="interop-swarm", prompt={}, extra_pnginfo={}, embed_workflow=True,
            metadata_export_settings=settings,
        )
        filename = result["ui"]["kiri_saved_files"][0][0]["filename"]
        with Image.open(self.output_dir / filename) as image:
            envelope = json.loads(image.info["parameters"])
            self.assertIn("sui_image_params", envelope)
            self.assertEqual(envelope["sui_image_params"]["width"], "8")
            self.assertIn("invokeai_metadata", image.info)
            self.assertIn("kiri_interop", image.info)

    def test_interchange_sidecar_is_opt_in_and_uses_matching_output_stem(self):
        tensor = torch.rand((1, 6, 8, 3))
        settings = json.dumps({"write_sidecar": True})
        result = self.load_node_with_saving(
            "disk.png", image_in=tensor, unique_id="interop-sidecar", prompt={}, extra_pnginfo={}, embed_workflow=True,
            metadata_export_settings=settings,
        )
        file_record = result["ui"]["kiri_saved_files"][0][0]
        image_path = self.output_dir / file_record["filename"]
        sidecar_path = self.output_dir / file_record["metadata_sidecar"]
        self.assertEqual(sidecar_path.name, image_path.stem + ".kiri.json")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        self.assertEqual(sidecar["schema"], "kiri-universal-generation-metadata")

    def test_metadata_editor_refreshes_export_adapters_and_preserves_case_distinct_comments(self):
        path = self.output_dir / "adapter-edit.png"
        Image.new("RGB", (32, 24), (12, 34, 56)).save(path)
        kiri_record = {"schema_version": 1, "collector": "Kiri Image Loader", "graph": {
            "prompts": {"positive": "old positive", "negative": "old negative"},
            "samplers": [{"class_type": "KSampler", "settings": {"seed": 10, "steps": 20, "cfg": 6.0, "sampler_name": "euler"}}],
            "model_resources": [],
            "image_dimensions": {"width": 32, "height": 24},
        }}
        invoke = {"metadata_version": "1.0.0", "positive_prompt": "old positive", "negative_prompt": "old negative", "seed": 10, "width": 32, "height": 24, "external_note": "preserve"}
        novel_comment = {"prompt": "old positive", "uc": "old negative", "seed": 10, "custom_field": "preserve"}
        interop = {"schema": "kiri-universal-generation-metadata", "generation": {"positive_prompt": "old positive"}}
        entries = [
            {"keyword": "kiri_metadata", "value": json.dumps(kiri_record)},
            {"keyword": "parameters", "value": "old positive\nNegative prompt: old negative\nSteps: 20, Seed: 10"},
            {"keyword": "invokeai_metadata", "value": json.dumps(invoke)},
            {"keyword": "Description", "value": "old positive"},
            {"keyword": "Comment", "value": json.dumps(novel_comment)},
            {"keyword": "comment", "value": "ordinary user note"},
            {"keyword": "kiri_interop", "value": json.dumps(interop)},
        ]
        sidecar = path.with_name(path.stem + ".kiri.json")
        sidecar.write_text(json.dumps({"schema": "kiri-universal-generation-metadata", "kiri_metadata": kiri_record, "adapters": interop}), encoding="utf-8")
        self.package._rewrite_png_text_metadata(str(path), entries, {
            "positive_prompt": "new positive", "negative_prompt": "new negative",
            "parameters": [
                {"kind": "sampler", "sampler_index": 0, "key": "seed", "value": "999"},
                {"kind": "sampler", "sampler_index": 0, "key": "steps", "value": "42"},
            ],
            "image_width": "32", "image_height": "24",
        })
        with Image.open(path) as image:
            self.assertEqual(image.info["Description"], "new positive")
            self.assertEqual(json.loads(image.info["Comment"])["prompt"], "new positive")
            self.assertEqual(json.loads(image.info["Comment"])["custom_field"], "preserve")
            self.assertEqual(image.info["comment"], "ordinary user note")
            updated_invoke = json.loads(image.info["invokeai_metadata"])
            self.assertEqual(updated_invoke["positive_prompt"], "new positive")
            self.assertEqual(updated_invoke["seed"], 999)
            self.assertEqual(updated_invoke["external_note"], "preserve")
            self.assertEqual(json.loads(image.info["kiri_interop"])["generation"]["positive_prompt"], "new positive")
        updated_sidecar = json.loads(sidecar.read_text(encoding="utf-8"))
        self.assertEqual(updated_sidecar["kiri_metadata"]["graph"]["prompts"]["negative"], "new negative")
        self.assertEqual(updated_sidecar["adapters"]["generation"]["positive_prompt"], "new positive")

    def test_graph_aware_metadata_collects_active_models_loras_prompts_and_sampler(self):
        image_value = torch.rand((1, 8, 9, 3))
        prompt = {
            "100": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "base.safetensors"}},
            "101": {"class_type": "LoraLoader", "inputs": {"model": ["100", 0], "clip": ["100", 1], "lora_name": "style.safetensors", "strength_model": 0.75, "strength_clip": 0.6}},
            "102": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["101", 1], "text": "beautiful positive prompt"}},
            "103": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["101", 1], "text": "bad negative prompt"}},
            "104": {"class_type": "KSampler", "inputs": {"model": ["101", 0], "positive": ["102", 0], "negative": ["103", 0], "latent_image": ["105", 0], "seed": 123456, "steps": 32, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.85}},
            "105": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 768, "batch_size": 1}},
            "106": {"class_type": "VAEDecode", "inputs": {"samples": ["104", 0], "vae": ["100", 2]}},
            "999": {"class_type": "KiriImageLoader", "inputs": {"image": "disk.png", "image_in": ["106", 0]}},
            # This node is not reachable from A and must not enter Kiri's graph record.
            "777": {"class_type": "KSampler", "inputs": {"seed": 999, "steps": 1, "cfg": 1}},
        }
        workflow = {"nodes": [{"id": 100, "type": "CheckpointLoaderSimple", "title": "Base Model"}, {"id": 777, "type": "KSampler", "title": "Unrelated Sampler"}], "links": [[1, 1, 0, 2, 0, "MODEL"]]}
        result = self.load_node_with_saving(
            "disk.png", image_value, None, "999", "manual artist metadata", True, None,
            prompt, {"workflow": workflow}, None,
        )
        self.assertIs(result["result"][0], image_value)
        saved = sorted(self.output_dir.glob("Kirin Image*.png"), key=lambda p: p.stat().st_mtime_ns)[-1]
        with Image.open(saved) as image:
            info = dict(image.info)
        self.assertIn("prompt", info, "The explicit ON setting preserves the active ComfyUI execution prompt.")
        self.assertIn("kiri_metadata", info)
        self.assertIn("parameters", info)
        self.assertIn("workflow", info, "The explicit ON setting preserves the full visual canvas workflow.")
        metadata = __import__("json").loads(info["kiri_metadata"])
        nodes = metadata["graph"]["active_upstream_nodes"]
        node_ids = {str(item["id"]) for item in nodes}
        self.assertIn("100", node_ids)
        self.assertIn("101", node_ids)
        self.assertIn("104", node_ids)
        self.assertNotIn("777", node_ids)
        self.assertNotIn("999", node_ids, "The Kiri saver itself is not an upstream generation node.")
        resources = metadata["graph"]["model_resources"]
        resource_names = [item.get("name") for item in resources]
        self.assertIn("base.safetensors", resource_names)
        self.assertIn("style.safetensors", resource_names)
        self.assertIn("Embedded in base.safetensors", resource_names)
        text_nodes = metadata["graph"]["text_encoders_and_prompts"]
        branches = {item["values"].get("text"): item["branch"] for item in text_nodes}
        self.assertEqual(branches["beautiful positive prompt"], "positive")
        self.assertEqual(branches["bad negative prompt"], "negative")
        self.assertEqual(metadata["custom_metadata"]["value"], "manual artist metadata")
        self.assertIn("Steps: 32", info["parameters"])
        self.assertIn("Seed: 123456", info["parameters"])

    def test_kiri_loader_bridge_records_checkpoint_selection_lora_strengths_and_civitai_sidecar(self):
        import json
        import folder_paths

        image_value = torch.rand((1, 6, 7, 3))
        lora_path = self.input_dir / "kiri-bridge.safetensors"
        lora_path.write_bytes(b"test lora bytes for metadata bridge")
        (self.input_dir / "kiri-bridge.cminfo.json").write_text(json.dumps({
            "CivitaiModelId": 178169,
            "CivitaiVersionId": 1136800,
            "ModelName": "Bridge Character",
            "VersionName": "SDXL Main",
            "CreatorUsername": "ExampleCreator",
            "BaseModel": "SDXL",
            "TrainedWords": ["bridge_character", "blue eyes"],
            "ModelUrl": "https://civitai.com/models/178169",
        }), encoding="utf-8")
        original_get_full_path = folder_paths.get_full_path
        folder_paths.get_full_path = lambda category, filename: str(lora_path) if category == "loras" and filename == "kiri-bridge.safetensors" else None
        try:
            prompt = {
                "100": {"class_type": "UniversalCheckpointLoader", "inputs": {
                    "text_encoder": "Auto", "text_encoder_2": "Auto", "text_encoder_3": "Auto",
                    "checkpoint": "bridge-base.safetensors", "vae": "Auto", "vae_2": "Auto",
                }},
                "101": {"class_type": "KeryulaPowerLoraLoader", "inputs": {
                    "model": ["100", 0], "clip": ["100", 1],
                    "loras_json": json.dumps([
                        {"on": True, "lora": "kiri-bridge.safetensors", "strength_model": 0.72, "strength_clip": 0.43},
                        {"on": False, "lora": "disabled.safetensors", "strength_model": 1.0, "strength_clip": 1.0},
                    ]),
                    "trigger_lora": "kiri-bridge.safetensors", "trigger_words": "bridge_character, blue eyes",
                }},
                "102": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["101", 1], "text": "bridge positive prompt"}},
                "103": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["101", 1], "text": "bridge negative prompt"}},
                "104": {"class_type": "KSampler", "inputs": {
                    "model": ["101", 0], "positive": ["102", 0], "negative": ["103", 0],
                    "latent_image": ["105", 0], "seed": 7654321, "steps": 30, "cfg": 5.8,
                    "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0,
                }},
                "105": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
                "106": {"class_type": "VAEDecode", "inputs": {"samples": ["104", 0], "vae": ["100", 2]}},
                "999": {"class_type": "KiriImageLoader", "inputs": {"image": "disk.png", "image_in": ["106", 0]}},
            }
            result = self.load_node_with_saving("disk.png", image_value, None, "kiri-bridge", "", False, None, prompt, {}, None)
        finally:
            folder_paths.get_full_path = original_get_full_path

        runtime = json.loads(result["ui"]["kiri_runtime_metadata_a"][0])
        graph = runtime["graph"]
        loras = [item for item in graph["model_resources"] if item.get("type") == "LoRA"]
        self.assertEqual(len(loras), 1, "Disabled Kiri LoRA rows should not appear as used resources.")
        self.assertEqual(loras[0]["name"], "kiri-bridge.safetensors")
        self.assertEqual(loras[0]["strength_model"], 0.72)
        self.assertEqual(loras[0]["strength_clip"], 0.43)
        self.assertEqual(loras[0]["civitai_provenance"]["model_id"], 178169)
        self.assertEqual(loras[0]["civitai_provenance"]["version_id"], 1136800)
        self.assertEqual(loras[0]["civitai_provenance"]["trained_words"], ["bridge_character", "blue eyes"])
        kiri_details = {str(item["node_id"]): item for item in graph["kiri_loader_details"]}
        self.assertEqual(kiri_details["100"]["checkpoint_or_model"], "bridge-base.safetensors")
        self.assertEqual(kiri_details["100"]["text_encoders"][0]["selection"], "Auto")
        self.assertEqual(kiri_details["100"]["text_encoders"][0]["source"], "auto / checkpoint-managed")
        self.assertEqual(kiri_details["100"]["vaes"][0]["source"], "auto / checkpoint-managed")
        self.assertEqual(kiri_details["101"]["trigger_words"], "bridge_character, blue eyes")
        self.assertEqual(graph["lora_trigger_words"][0]["value"], "bridge_character, blue eyes")
        self.assertNotIn("_kiri_loader_details", result["ui"]["kiri_runtime_metadata_a"][0], "Internal collector fields must not leak into the executed prompt.")

    def test_vae_decode_outputs_are_saved_with_metadata_and_b_has_independent_graph_metadata(self):
        import json

        image_a = torch.rand((1, 6, 7, 3))
        image_b = torch.rand((1, 6, 7, 3))
        prompt = {
            "10": {"class_type": "CLIPTextEncode", "inputs": {"text": "generated A prompt"}},
            "11": {"class_type": "KSampler", "inputs": {"positive": ["10", 0], "seed": 1001, "steps": 28, "cfg": 6}},
            "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0]}},
            "20": {"class_type": "CLIPTextEncode", "inputs": {"text": "generated B prompt"}},
            "21": {"class_type": "KSampler", "inputs": {"positive": ["20", 0], "seed": 2002, "steps": 36, "cfg": 4}},
            "22": {"class_type": "VAEDecode", "inputs": {"samples": ["21", 0]}},
            "999": {"class_type": "KiriImageLoader", "inputs": {
                "image": "disk.png", "image_in": ["12", 0], "mask_in": ["22", 0],
            }},
        }
        result = self.load_node_with_saving("disk.png", image_a, image_b, "vae-decode-outputs", prompt=prompt)
        self.assertIs(result["result"][0], image_a)
        self.assertIs(result["result"][1], image_b)
        self.assertEqual(result["ui"]["kiri_preview_mode"], ["compare"])
        self.assertIn("kiri_saved_files", result["ui"])
        metadata_a = json.loads(result["ui"]["kiri_runtime_metadata_a"][0])
        metadata_b = json.loads(result["ui"]["kiri_runtime_metadata_b"][0])
        self.assertEqual(metadata_a["metadata_scope"], "active upstream graph contributing to image_in")
        self.assertEqual(metadata_b["metadata_scope"], "active upstream graph contributing to mask_in")
        self.assertEqual(metadata_a["graph"]["samplers"][0]["settings"]["seed"], 1001)
        self.assertEqual(metadata_b["graph"]["samplers"][0]["settings"]["seed"], 2002)
        self.assertEqual(metadata_a["graph"]["text_encoders_and_prompts"][0]["values"]["text"], "generated A prompt")
        self.assertEqual(metadata_b["graph"]["text_encoders_and_prompts"][0]["values"]["text"], "generated B prompt")
        self.assertEqual(len(result["ui"]["kiri_saved_files"][0]), 1, "Only A should be written as the primary output file.")

    def test_multiline_string_nodes_feed_distinct_positive_and_negative_prompt_metadata_and_filename_tokens(self):
        import json

        image_value = torch.rand((1, 6, 7, 3))
        positive_text = "Positive from StringMultiline: cinematic portrait, warm light"
        negative_text = "Negative from StringMultiline: blurry, malformed hands"
        prompt = {
            "1": {"class_type": "UniversalCheckpointLoader", "inputs": {
                "checkpoint": "test-checkpoint.safetensors",
                "text_encoder": "Auto (from checkpoint)",
            }},
            "10": {"class_type": "StringMultiline", "inputs": {"text": positive_text}},
            "11": {"class_type": "StringMultiline", "inputs": {"value": negative_text}},
            "12": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 0], "text": ["10", 0]}},
            "13": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 0], "text": ["11", 0]}},
            "14": {"class_type": "ConditioningSetArea", "inputs": {"conditioning": ["12", 0], "width": 512, "height": 512}},
            "15": {"class_type": "ConditioningSetArea", "inputs": {"conditioning": ["13", 0], "width": 512, "height": 512}},
            "16": {"class_type": "KSampler", "inputs": {
                "positive": ["14", 0], "negative": ["15", 0], "seed": 998877,
                "steps": 31, "cfg": 5.5, "sampler_name": "dpmpp_2m", "scheduler": "karras",
            }},
            "17": {"class_type": "VAEDecode", "inputs": {"samples": ["16", 0]}},
            "999": {"class_type": "KiriImageLoader", "inputs": {
                "image": "disk.png", "image_in": ["17", 0],
            }},
        }
        result = self.load_node_with_saving(
            "disk.png", image_value, None, "multiline-prompt-test", prompt=prompt,
            save_path="Kiri Images/Prompt Tests", filename_template="Kirin Image [Gotten Seed]",
        )
        saved_record = result["ui"]["kiri_saved_files"][0][0]
        self.assertEqual(saved_record["filename"], "Kirin Image 998877.png")
        self.assertEqual(saved_record["subfolder"], "Kiri Images/Prompt Tests")
        saved_path = self.output_dir / "Kiri Images" / "Prompt Tests" / saved_record["filename"]
        self.assertTrue(saved_path.is_file())

        runtime = json.loads(result["ui"]["kiri_runtime_metadata_a"][0])
        self.assertEqual(runtime["graph"]["prompts"]["positive"], positive_text)
        self.assertEqual(runtime["graph"]["prompts"]["negative"], negative_text)
        self.assertNotIn("Auto (from checkpoint)", runtime["graph"]["prompts"]["positive"])
        self.assertNotIn("Auto (from checkpoint)", runtime["graph"]["prompts"]["negative"])
        checkpoint_details = next(item for item in runtime["graph"]["kiri_loader_details"] if item["node_id"] == "1")
        self.assertEqual(checkpoint_details["text_encoders"][0]["selection"], "Auto (from checkpoint)")
        self.assertEqual(checkpoint_details["text_encoders"][0]["source"], "auto / checkpoint-managed")
        self.assertIn(positive_text, runtime["parameters"])
        self.assertIn("Negative prompt: " + negative_text, runtime["parameters"])
        self.assertIn("Steps: 31", runtime["parameters"])
        self.assertNotIn("998877", runtime["graph"]["prompts"]["positive"])
        self.assertNotIn("Steps", runtime["graph"]["prompts"]["positive"])

        inspected = self.package.inspect_image(str(saved_path))
        copied_prompts = {item["label"]: item["value"] for item in inspected["copy_items"] if item["label"] in {"Positive Prompt", "Negative Prompt"}}
        self.assertEqual(copied_prompts["Positive Prompt"], positive_text)
        self.assertEqual(copied_prompts["Negative Prompt"], negative_text)

        # A second run with the same seed never overwrites the first image.
        second = self.load_node_with_saving(
            "disk.png", image_value, None, "multiline-prompt-test-2", prompt=prompt,
            save_path="Kiri Images/Prompt Tests", filename_template="Kirin Image [Gotten Seed]",
        )
        second_record = second["ui"]["kiri_saved_files"][0][0]
        self.assertNotEqual(second_record["filename"], saved_record["filename"])
        self.assertTrue((self.output_dir / "Kiri Images" / "Prompt Tests" / second_record["filename"]).is_file())

    def test_png_metadata_editor_replaces_text_without_changing_pixels_or_binary_chunks(self):
        import json
        import struct
        from PIL import PngImagePlugin

        path = self.output_dir / "editable-metadata.png"
        pixels = Image.new("RGB", (11, 9), (17, 83, 149))
        info = PngImagePlugin.PngInfo()
        info.add_text("prompt", "old positive")
        info.add_text("parameters", "Steps: 20")
        info.add_itxt("workflow", json.dumps({"nodes": [{"id": 1}]}))
        icc = b"Kiri ICC profile fixture" * 4
        pixels.save(path, pnginfo=info, icc_profile=icc)

        def chunks_without_text(target):
            raw = target.read_bytes()
            cursor = 8
            chunks = []
            while cursor + 12 <= len(raw):
                length = struct.unpack(">I", raw[cursor:cursor + 4])[0]
                end = cursor + 12 + length
                chunk_type = raw[cursor + 4:cursor + 8]
                if chunk_type not in {b"tEXt", b"zTXt", b"iTXt"}:
                    chunks.append(raw[cursor:end])
                cursor = end
                if chunk_type == b"IEND":
                    break
            return chunks

        original_binary_chunks = chunks_without_text(path)
        with Image.open(path) as image:
            original_pixels = image.copy()

        self.assertEqual(self.package._rewrite_png_text_metadata(str(path), [
            {"keyword": "prompt", "value": "new positive\nsecond line"},
            {"keyword": "negative_prompt", "value": "new negative"},
            {"keyword": "kiri_note", "value": "edited internally"},
        ]), 3)

        with Image.open(path) as image:
            self.assertEqual(image.tobytes(), original_pixels.tobytes())
            self.assertEqual(image.size, original_pixels.size)
            self.assertEqual(image.info.get("icc_profile"), icc)
        self.assertEqual(chunks_without_text(path), original_binary_chunks)
        edited = self.package.inspect_image(str(path))
        values = {item["label"]: item["value"] for item in edited["copy_items"]}
        self.assertEqual(values["Positive Prompt"], "new positive\nsecond line")
        self.assertEqual(values["Negative Prompt"], "new negative")
        self.assertIn("Metadata · kiri_note", values)
        self.assertNotIn("Steps: 20", values.values())
        self.assertEqual(edited["editable_metadata"][0]["keyword"], "prompt")

        jpeg_path = self.output_dir / "not-editable.jpg"
        Image.new("RGB", (4, 4), (10, 20, 30)).save(jpeg_path)
        with self.assertRaisesRegex(ValueError, "PNG images only"):
            self.package._rewrite_png_text_metadata(str(jpeg_path), [])

    def test_absolute_save_path_outside_output_uses_registered_file_token(self):
        import json

        target = self.input_dir / "custom-output-location"
        image_value = torch.rand((1, 4, 5, 3))
        prompt = {
            "1": {"class_type": "KSampler", "inputs": {"seed": 654321, "steps": 20}},
            "2": {"class_type": "VAEDecode", "inputs": {"samples": ["1", 0]}},
            "999": {"class_type": "KiriImageLoader", "inputs": {"image": "disk.png", "image_in": ["2", 0]}},
        }
        result = self.load_node_with_saving(
            "disk.png", image_value, None, "external-save-path-test", prompt=prompt,
            save_path=str(target), filename_template="Kirin Image [Gotten Seed]",
        )
        record = result["ui"]["kiri_saved_files"][0][0]
        self.assertEqual(record["type"], "kiri_external")
        self.assertEqual(record["filename"], "Kirin Image 654321.png")
        self.assertTrue(record["path_token"])
        saved_path = target / record["filename"]
        self.assertTrue(saved_path.is_file())
        self.assertEqual(self.package._resolve_external_output_path(record["path_token"]), str(saved_path.resolve()))
        self.package._forget_external_output_path(record["path_token"])
        with self.assertRaisesRegex(FileNotFoundError, "reference is no longer available"):
            self.package._resolve_external_output_path(record["path_token"])

    def test_clipboard_image_route_serves_managed_and_tokenized_external_images(self):
        import asyncio
        from types import SimpleNamespace

        route = self.routes.handlers[("GET", "/kiri-image-loader/image")]
        managed_request = SimpleNamespace(query={"filename": "disk.png", "type": "input", "subfolder": ""})
        managed_response = asyncio.run(route(managed_request))
        self.assertEqual(managed_response.status, 200)
        self.assertEqual(Path(managed_response._path), (self.input_dir / "disk.png").resolve())

        blocked_request = SimpleNamespace(query={"filename": "../outside.png", "type": "input", "subfolder": ""})
        blocked_response = asyncio.run(route(blocked_request))
        self.assertEqual(blocked_response.status, 404, "Clipboard serving must not permit paths outside managed ComfyUI roots.")

        external_path = self.output_dir / "clipboard-external.png"
        Image.new("RGB", (5, 4), (20, 40, 60)).save(external_path)
        token = self.package._register_external_output_path(str(external_path))
        external_request = SimpleNamespace(query={"filename": external_path.name, "type": "kiri_external", "path_token": token})
        external_response = asyncio.run(route(external_request))
        self.assertEqual(external_response.status, 200)
        self.assertEqual(Path(external_response._path), external_path.resolve())
        self.package._forget_external_output_path(token)

    def test_save_path_rejects_parent_directory_escape(self):
        image_value = torch.rand((1, 4, 5, 3))
        with self.assertRaisesRegex(ValueError, "inside ComfyUI/output"):
            self.package._safe_output_directory("../outside-output")

    def test_connected_custom_metadata_overrides_manual_text(self):
        image_value = torch.rand((1, 4, 5, 3))
        connected = {"artist": "connected artist", "rating": 5, "tags": ["one", "two"]}
        result = self.load_node_with_saving(
            "disk.png", image_value, None, "custom-override", "manual text", True,
            connected, {}, {}, None,
        )
        saved = sorted(self.output_dir.glob("Kirin Image*.png"), key=lambda p: p.stat().st_mtime_ns)[-1]
        with Image.open(saved) as image:
            metadata = __import__("json").loads(image.info["kiri_metadata"])
            custom = __import__("json").loads(image.info["kiri_custom_metadata"])
        self.assertEqual(metadata["custom_metadata"]["source"], "connected input")
        self.assertEqual(metadata["custom_metadata"]["value"], connected)
        self.assertEqual(custom["value"], connected)
        self.assertNotEqual(metadata["custom_metadata"]["value"], "manual text")

    def test_manual_custom_metadata_parses_json_but_preserves_freeform_text(self):
        image_value = torch.rand((1, 4, 5, 3))
        self.load_node_with_saving("disk.png", image_value, None, "custom-json", '{"artist":"Keryula","rating":5}', True, None, {}, {}, None)
        saved = sorted(self.output_dir.glob("Kirin Image*.png"), key=lambda p: p.stat().st_mtime_ns)[-1]
        with Image.open(saved) as image:
            metadata = __import__("json").loads(image.info["kiri_metadata"])
            custom = __import__("json").loads(image.info["kiri_custom_metadata"])
        self.assertEqual(metadata["custom_metadata"]["source"], "manual editor (JSON)")
        self.assertEqual(metadata["custom_metadata"]["value"], {"artist": "Keryula", "rating": 5})
        self.assertEqual(custom["value"], {"artist": "Keryula", "rating": 5})

    def test_clip_vision_loader_is_not_mislabeled_as_text_encoder(self):
        classification = self.package._classify_model_resource("CLIPVisionLoader", "clip_name")
        self.assertEqual(classification[0], "CLIP Vision")

    def test_workflow_embedding_is_opt_in_and_redacts_credentials(self):
        image_value = torch.rand((1, 4, 5, 3))
        workflow = {
            "nodes": [{"id": 1, "type": "LoadImage", "widgets_values": ["image.png"],
                       "properties": {"api_key": "do-not-embed", "safe_label": "kept"}}],
            "links": [],
        }
        api_prompt = {
            "1": {"class_type": "KSampler", "inputs": {"seed": 123, "steps": 20}},
        }
        self.load_node_with_saving(
            "disk.png", image_value, None, "workflow-on", "", True, None,
            api_prompt, {"workflow": workflow}, None,
        )
        saved = sorted(self.output_dir.glob("Kirin Image*.png"), key=lambda p: p.stat().st_mtime_ns)[-1]
        with Image.open(saved) as image:
            self.assertIn("workflow", image.info)
            self.assertIn("prompt", image.info, "The explicit ON setting keeps the active ComfyUI execution prompt.")
            embedded = __import__("json").loads(image.info["workflow"])
        properties = embedded["nodes"][0]["properties"]
        self.assertEqual(properties["api_key"], "[REDACTED]")
        self.assertEqual(properties["safe_label"], "kept")

    def test_workflow_off_with_platform_metadata_disabled_is_plain_text_only(self):
        image_value = torch.rand((1, 4, 5, 3))
        api_prompt = {
            "10": {"class_type": "KSampler", "inputs": {
                "seed": 123456, "steps": 28, "cfg": 6.5,
                "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
                "positive": ["12", 0], "negative": ["13", 0],
            }},
            "12": {"class_type": "CLIPTextEncode", "inputs": {"text": "strict positive prompt"}},
            "13": {"class_type": "CLIPTextEncode", "inputs": {"text": "strict negative prompt"}},
            "11": {"class_type": "KiriImageLoader", "inputs": {"image_in": ["10", 0]}},
        }
        canvas_workflow = {
            "id": "source-workflow-uuid",
            "last_node_id": 11,
            "last_link_id": 1,
            "nodes": [
                {"id": 10, "type": "KSampler", "title": "My Custom Sampler Name"},
                {"id": 12, "type": "CLIPTextEncode", "title": "My Custom Positive Text Node"},
                {"id": 13, "type": "CLIPTextEncode", "title": "My Custom Negative Text Node"},
                {"id": 11, "type": "KiriImageLoader", "title": "Kiri Image Loader"},
            ],
            "links": [[1, 10, 0, 11, 0, "IMAGE"]],
        }
        extra_pnginfo = {
            "workflow": canvas_workflow,
            "ComfyUI_Workflow": canvas_workflow,
            "prompt": api_prompt,
            "additional_metadata": {
                "workflow_json": __import__("json").dumps(canvas_workflow),
                "artist_note": "must not be serialized in strict OFF mode",
            },
        }
        custom_metadata = {
            "artist": "Keryula",
            "workflow": canvas_workflow,
            "rating": 5,
        }
        export_settings = __import__("json").dumps({
            "parameters_format": "fooocus",
            "invokeai": True,
            "novelai": True,
            "universal_bundle": True,
            "platform_metadata": False,
            "write_sidecar": True,
            "write_easy_diffusion_task": True,
        })
        result = self.load_node_with_saving(
            "disk.png", image_value, None, "11", "manual artist metadata", False,
            custom_metadata, api_prompt, extra_pnginfo, None,
            metadata_export_settings=export_settings,
        )
        file_record = result["ui"]["kiri_saved_files"][0][0]
        saved = self.output_dir / file_record["filename"]
        with Image.open(saved) as image:
            info = dict(image.info)
            self.assertEqual(image.mode, "RGB", "NovelAI stealth alpha payload must not be written in text-only OFF mode.")
            self.assertEqual(image.getexif(), {}, "Draw Things EXIF JSON must not be written in text-only OFF mode.")

        # Exact allow-list: no JSON chunk of any app/platform type can remain.
        self.assertEqual(set(info), {"parameters", "Kiri Nodes Used", "Kiri Metadata Mode"})
        self.assertIn("strict positive prompt", info["parameters"])
        self.assertIn("Negative prompt: strict negative prompt", info["parameters"])
        self.assertIn("Steps: 28", info["parameters"])
        self.assertIn("Seed: 123456", info["parameters"])
        self.assertIn("Size: 5x4", info["parameters"])
        self.assertIn("My Custom Sampler Name", info["Kiri Nodes Used"])
        self.assertIn("My Custom Positive Text Node", info["Kiri Nodes Used"])
        self.assertIn("My Custom Negative Text Node", info["Kiri Nodes Used"])
        self.assertNotIn("10", info["Kiri Nodes Used"], "Node IDs must not be included in the human-readable inventory.")
        self.assertNotIn("123456", info["Kiri Nodes Used"], "Node inputs must not be included in the human-readable inventory.")
        self.assertEqual(info["Kiri Metadata Mode"], "ComfyUI Workflow OFF; Other-Platform Metadata OFF; no ComfyUI graph data")
        for key, value in info.items():
            try:
                __import__("json").loads(value)
            except (TypeError, ValueError):
                continue
            self.fail(f"Text-only OFF metadata must not contain JSON, but {key!r} parsed as JSON.")

        self.assertNotIn("metadata_sidecar", file_record)
        self.assertNotIn("easy_diffusion_task", file_record)
        self.assertFalse(saved.with_suffix(".kiri.json").exists())
        self.assertFalse(saved.with_suffix(".easy-diffusion.json").exists())

        inspected = self.package.inspect_image(str(saved))
        self.assertIsNone(inspected["embedded_workflow"])
        self.assertIsNone(inspected["generation_data"], "No node IDs/inputs remain to reconstruct a graph or run Apply Generation Data.")
        node_item = next(item for item in inspected["copy_items"] if item["label"] == "Nodes Used")
        self.assertEqual(node_item["category"], "ComfyUI Nodes")
        self.assertEqual(node_item["value"], info["Kiri Nodes Used"])

    def test_workflow_off_can_export_other_platform_parameters_without_any_comfyui_graph_json(self):
        image_value = torch.rand((1, 4, 5, 3))
        api_prompt = {
            "10": {"class_type": "KSampler", "inputs": {
                "seed": 123456, "steps": 28, "cfg": 6.5,
                "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
                "positive": ["12", 0], "negative": ["13", 0],
            }},
            "12": {"class_type": "CLIPTextEncode", "inputs": {"text": "platform positive"}},
            "13": {"class_type": "CLIPTextEncode", "inputs": {"text": "platform negative"}},
            "11": {"class_type": "KiriImageLoader", "inputs": {"image_in": ["10", 0]}},
        }
        canvas_workflow = {
            "id": "do-not-embed-this-canvas-id",
            "nodes": [
                {"id": 10, "type": "KSampler", "title": "Custom Sampler"},
                {"id": 12, "type": "CLIPTextEncode", "title": "Custom Positive"},
                {"id": 13, "type": "CLIPTextEncode", "title": "Custom Negative"},
                {"id": 11, "type": "KiriImageLoader", "title": "Kiri Image Loader"},
            ],
            "links": [[1, 10, 0, 11, 0, "IMAGE"]],
        }
        settings = json.dumps({
            "parameters_format": "fooocus",
            "save_prompts": True,
            "save_node_names": True,
            "platform_metadata": True,
            "invokeai": True,
            "novelai": False,
            "universal_bundle": True,
            "write_sidecar": True,
            "write_easy_diffusion_task": True,
        })
        result = self.load_node_with_saving(
            "disk.png", image_value, None, "workflow-off-platform-on", "manual custom metadata", False,
            {"private_custom": "must not be embedded as Kiri graph metadata"}, api_prompt,
            {"workflow": canvas_workflow, "prompt": api_prompt}, None,
            metadata_export_settings=settings,
        )
        file_record = result["ui"]["kiri_saved_files"][0][0]
        saved = self.output_dir / file_record["filename"]
        with Image.open(saved) as image:
            info = dict(image.info)

        self.assertNotIn("prompt", info)
        self.assertNotIn("workflow", info)
        self.assertNotIn("kiri_metadata", info)
        self.assertNotIn("kiri_custom_metadata", info)
        self.assertIn("parameters", info)
        self.assertIn("fooocus_scheme", info)
        self.assertIn("invokeai_metadata", info)
        self.assertIn("kiri_interop", info)
        self.assertIn("pixai_api", info)
        self.assertIn("tensor_art_tams", info)
        self.assertIn("drawthings_user_comment", info)
        self.assertIn("XML:com.adobe.xmp", info)
        self.assertIn("easy_diffusion_task", info)
        self.assertIn("Kiri Nodes Used", info)
        self.assertNotIn("10", info["Kiri Nodes Used"])
        self.assertIn("Custom Sampler", info["Kiri Nodes Used"])

        forbidden = {"node_id", "class_type", "inputs", "input", "nodes", "links", "workflow", "widgets_values", "active_upstream_nodes", "root_node_id", "source_workflow_id"}
        def check_no_comfy_graph(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden.intersection(value.keys()), f"Platform metadata contains ComfyUI node fields: {forbidden.intersection(value.keys())}")
                for child in value.values(): check_no_comfy_graph(child)
            elif isinstance(value, list):
                for child in value: check_no_comfy_graph(child)

        for key, value in info.items():
            if key in {"parameters", "Kiri Nodes Used", "Kiri Metadata Mode", "Description", "Software", "XML:com.adobe.xmp"}:
                continue
            try:
                payload = json.loads(value)
            except (TypeError, ValueError):
                continue
            check_no_comfy_graph(payload)

        inspected = self.package.inspect_image(str(saved))
        self.assertIsNone(inspected["embedded_workflow"])
        self.assertIsNone(inspected["generation_data"])
        sidecar_path = self.output_dir / file_record["metadata_sidecar"]
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        check_no_comfy_graph(sidecar)
        task_path = self.output_dir / file_record["easy_diffusion_task"]
        task = json.loads(task_path.read_text(encoding="utf-8"))
        check_no_comfy_graph(task)

    def test_prompt_and_node_name_toggles_apply_to_a_and_b_runtime_metadata_and_saved_output(self):
        import json

        image_a = torch.rand((1, 6, 7, 3))
        image_b = torch.rand((1, 6, 7, 3))
        prompt = {
            "10": {"class_type": "KSampler", "inputs": {
                "positive": ["12", 0], "negative": ["13", 0],
                "seed": 1001, "steps": 28, "cfg": 6,
            }},
            "12": {"class_type": "CLIPTextEncode", "inputs": {"text": "must-not-save-positive"}},
            "13": {"class_type": "CLIPTextEncode", "inputs": {"text": "must-not-save-negative"}},
            "20": {"class_type": "KSampler", "inputs": {
                "positive": ["22", 0], "negative": ["23", 0],
                "seed": 2002, "steps": 36, "cfg": 4,
            }},
            "22": {"class_type": "CLIPTextEncode", "inputs": {"text": "must-not-save-B-positive"}},
            "23": {"class_type": "CLIPTextEncode", "inputs": {"text": "must-not-save-B-negative"}},
            "999": {"class_type": "KiriImageLoader", "inputs": {"image_in": ["10", 0], "mask_in": ["20", 0]}},
        }
        settings = json.dumps({
            "parameters_format": "fooocus",
            "save_prompts": False,
            "save_node_names": False,
            "platform_metadata": True,
            "novelai": False,
        })
        result = self.load_node_with_saving(
            "disk.png", image_a, image_b, "prompt-node-toggles", prompt=prompt,
            metadata_export_settings=settings,
        )
        runtime_b = json.loads(result["ui"]["kiri_runtime_metadata_b"][0])
        self.assertNotIn("positive", runtime_b["graph"]["prompts"])
        self.assertNotIn("negative", runtime_b["graph"]["prompts"])
        self.assertFalse(any(item.get("branch") in {"positive", "negative", "shared"} for item in runtime_b["graph"]["text_encoders_and_prompts"]))

        file_record = result["ui"]["kiri_saved_files"][0][0]
        with Image.open(self.output_dir / file_record["filename"]) as image:
            info = dict(image.info)
        self.assertNotIn("Kiri Nodes Used", info)
        self.assertNotIn("must-not-save-positive", info.get("parameters", ""))
        self.assertNotIn("must-not-save-negative", info.get("parameters", ""))
        fooocus = json.loads(info["parameters"])
        self.assertEqual(fooocus["prompt"], "")
        self.assertEqual(fooocus["negative_prompt"], "")

    def test_independent_prompt_controls_preserve_sampler_checkpoint_and_lora_metadata(self):
        import json

        graph = {
            "prompts": {"positive": "private positive", "negative": "public negative"},
            "samplers": [{"class_type": "KSampler", "settings": {
                "seed": 987654321, "steps": 32, "cfg": 6.5,
                "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0,
            }}],
            "model_resources": [
                {"type": "Checkpoint / Model", "name": "base.safetensors", "sha256": "a" * 64},
                {"type": "LoRA", "name": "style.safetensors", "strength_model": 0.75, "strength_clip": 0.5, "sha256": "b" * 64},
            ],
            "text_encoders_and_prompts": [
                {"node_id": "12", "branch": "positive", "values": {"text": "private positive"}},
                {"node_id": "13", "branch": "negative", "values": {"text": "public negative"}},
            ],
            "active_upstream_nodes": [
                {"id": "12", "class_type": "CLIPTextEncode", "inputs": {"text": "private positive"}},
                {"id": "13", "class_type": "CLIPTextEncode", "inputs": {"text": "public negative"}},
                {"id": "10", "class_type": "KSampler", "inputs": {"steps": 32, "cfg": 6.5}},
            ],
        }
        record = {"graph": graph, "parameters": "old parameters"}
        prefs = self.package.normalize_export_settings({
            "save_positive_prompt": False,
            "save_negative_prompt": True,
            "save_node_names": True,
        })
        saved = self.package._apply_metadata_save_preferences(record, prefs, preserve_comfy_graph=True)
        saved_graph = saved["graph"]

        self.assertNotIn("positive", saved_graph["prompts"])
        self.assertEqual(saved_graph["prompts"]["negative"], "public negative")
        self.assertEqual(saved_graph["samplers"][0]["settings"]["steps"], 32)
        self.assertEqual(saved_graph["samplers"][0]["settings"]["cfg"], 6.5)
        self.assertEqual(saved_graph["model_resources"][0]["name"], "base.safetensors")
        self.assertEqual(saved_graph["model_resources"][1]["name"], "style.safetensors")
        self.assertEqual(saved_graph["model_resources"][1]["strength_model"], 0.75)
        self.assertEqual(saved_graph["active_upstream_nodes"][0]["inputs"]["text"], "")
        self.assertEqual(saved_graph["active_upstream_nodes"][1]["inputs"]["text"], "public negative")

        params = saved["parameters"]
        self.assertTrue(params.startswith("\nNegative prompt: public negative\nSteps:"))
        self.assertIn("CFG scale: 6.5", params)
        self.assertIn("Seed: 987654321", params)
        self.assertIn("Model: base.safetensors", params)
        self.assertIn("LoRA 1: style.safetensors [model=0.75; clip=0.5]", params)
        self.assertNotIn("private positive", params)

        # Reverse the privacy choice: preserve positive and independently blank negative.
        reverse = self.package._apply_metadata_save_preferences(record, self.package.normalize_export_settings({
            "save_positive_prompt": True, "save_negative_prompt": False,
        }), preserve_comfy_graph=True)
        self.assertEqual(reverse["graph"]["prompts"]["positive"], "private positive")
        self.assertNotIn("negative", reverse["graph"]["prompts"])
        self.assertTrue(reverse["parameters"].startswith("private positive\nNegative prompt: \nSteps:"), repr(reverse["parameters"]))
        self.assertIn("Model: base.safetensors", reverse["parameters"])
        self.assertIn("LoRA 1: style.safetensors", reverse["parameters"])

    def test_prompt_off_saved_png_keeps_sampler_models_and_loras_readable(self):
        import json

        prompt = {
            "10": {"class_type": "KSampler", "inputs": {
                "model": ["21", 0], "positive": ["12", 0], "negative": ["13", 0],
                "seed": 987654321, "steps": 32, "cfg": 6.5,
                "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0,
            }},
            "12": {"class_type": "CLIPTextEncode", "inputs": {"text": "must-not-save-positive", "clip": ["21", 1]}},
            "13": {"class_type": "CLIPTextEncode", "inputs": {"text": "keep-this-negative", "clip": ["21", 1]}},
            "20": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "base.safetensors"}},
            "21": {"class_type": "LoraLoader", "inputs": {
                "model": ["20", 0], "clip": ["20", 1], "lora_name": "style.safetensors",
                "strength_model": 0.75, "strength_clip": 0.5,
            }},
            "999": {"class_type": "KiriImageLoader", "inputs": {"image_in": ["10", 0]}},
        }
        settings = json.dumps({
            "save_positive_prompt": False,
            "save_negative_prompt": True,
            "save_node_names": False,
            "platform_metadata": False,
        })
        # This test exercises metadata independently of empty/bypassed links.
        # Supply a real A tensor so the saver creates a new PNG rather than
        # correctly skipping a connected-but-empty generation branch.
        image_value = torch.rand((1, 6, 7, 3), dtype=torch.float32)
        self.load_node_with_saving(
            "disk.png", image_value, None, "999", prompt=prompt,
            metadata_export_settings=settings,
        )
        saved_path = sorted(self.output_dir.glob("Kirin Image*.png"), key=lambda p: p.stat().st_mtime_ns)[-1]
        with Image.open(saved_path) as image:
            parameters = image.info["parameters"]
        self.assertNotIn("must-not-save-positive", parameters)
        self.assertIn("Negative prompt: keep-this-negative", parameters)
        self.assertIn("Steps: 32", parameters)
        self.assertIn("CFG scale: 6.5", parameters)
        self.assertIn("Seed: 987654321", parameters)
        self.assertIn("Sampler: dpmpp_2m", parameters)
        self.assertIn("Scheduler: karras", parameters)
        self.assertIn("Model: base.safetensors", parameters)
        self.assertIn("LoRA 1: style.safetensors [model=0.75; clip=0.5]", parameters)

        # The same string must be categorized as settings/resources, not one giant positive prompt.
        inspected = self.package.inspect_image(str(saved_path))
        values = {item["label"]: item["value"] for item in inspected["copy_items"]}
        self.assertNotIn("Positive Prompt", values)
        self.assertEqual(values["Negative Prompt"], "keep-this-negative")
        self.assertEqual(values["Steps"], "32")
        self.assertEqual(values["CFG"], "6.5")
        self.assertEqual(values["Sampler"], "dpmpp_2m")
        self.assertEqual(values["Scheduler"], "karras")
        self.assertEqual(values["Seed"], "987654321")
        self.assertEqual(values["Checkpoint"], "base.safetensors")
        self.assertEqual(values["LoRA 1 Name"], "style.safetensors")
        self.assertEqual(values["LoRA 1 Strength"], "0.75")
        self.assertEqual(values["LoRA 1 CLIP Strength"], "0.5")

    def test_independent_prompt_preferences_flow_into_platform_profiles_without_losing_resources(self):
        import json

        record = {
            "graph": {
                "prompts": {"positive": "private positive", "negative": "public negative"},
                "samplers": [{"settings": {"seed": 77, "steps": 24, "cfg": 5.5, "sampler_name": "euler", "scheduler": "normal"}}],
                "model_resources": [
                    {"type": "Checkpoint / Model", "name": "platform-base.safetensors"},
                    {"type": "LoRA", "name": "platform-style.safetensors", "strength_model": 0.65, "strength_clip": 0.4},
                ],
            }
        }
        prefs = self.package.normalize_export_settings({
            "parameters_format": "fooocus",
            "save_positive_prompt": False,
            "save_negative_prompt": True,
        })
        filtered = self.package._platform_export_record(
            self.package._apply_metadata_save_preferences(record, prefs, preserve_comfy_graph=False)
        )
        chunks = self.package.build_export_chunks(filtered, 1024, 768, prefs)
        fooocus = json.loads(chunks["parameters"])
        self.assertEqual(fooocus["prompt"], "")
        self.assertEqual(fooocus["negative_prompt"], "public negative")
        self.assertEqual(fooocus["steps"], 24)
        self.assertEqual(fooocus["base_model"], "platform-base")
        self.assertEqual(fooocus["lora_combined_1"], "platform-style : 0.65")

        reverse_prefs = self.package.normalize_export_settings({
            "parameters_format": "fooocus",
            "save_positive_prompt": True,
            "save_negative_prompt": False,
        })
        reverse_filtered = self.package._platform_export_record(
            self.package._apply_metadata_save_preferences(record, reverse_prefs, preserve_comfy_graph=False)
        )
        reverse_chunks = self.package.build_export_chunks(reverse_filtered, 1024, 768, reverse_prefs)
        reverse_fooocus = json.loads(reverse_chunks["parameters"])
        self.assertEqual(reverse_fooocus["prompt"], "private positive")
        self.assertEqual(reverse_fooocus["negative_prompt"], "")
        self.assertEqual(reverse_fooocus["base_model"], "platform-base")
        self.assertEqual(reverse_fooocus["lora_combined_1"], "platform-style : 0.65")

    def test_workflow_on_positive_prompt_off_keeps_negative_prompt_and_graph_settings(self):
        import json

        image_value = torch.rand((1, 6, 7, 3))
        prompt = {
            "10": {"class_type": "KSampler", "inputs": {"positive": ["12", 0], "negative": ["13", 0], "steps": 32, "seed": 987}},
            "12": {"class_type": "CLIPTextEncode", "inputs": {"text": "private-positive-text"}},
            "13": {"class_type": "CLIPTextEncode", "inputs": {"text": "retained-negative-text"}},
            "99": {"class_type": "KiriImageLoader", "inputs": {"image_in": ["10", 0]}},
        }
        workflow = {"nodes": [
            {"id": 10, "type": "KSampler", "widgets_values": [32, 987]},
            {"id": 12, "type": "CLIPTextEncode", "widgets_values": ["private-positive-text"]},
            {"id": 13, "type": "CLIPTextEncode", "widgets_values": ["retained-negative-text"]},
        ], "links": []}
        settings = json.dumps({
            "save_positive_prompt": False, "save_negative_prompt": True,
            "platform_metadata": False,
        })
        self.load_node_with_saving(
            "disk.png", image_value, None, "workflow-on-positive-off", "", True, None,
            prompt, {"workflow": workflow}, None, metadata_export_settings=settings,
        )
        saved = sorted(self.output_dir.glob("Kirin Image*.png"), key=lambda p: p.stat().st_mtime_ns)[-1]
        with Image.open(saved) as image:
            api = json.loads(image.info["prompt"])
            canvas = json.loads(image.info["workflow"])
            record = json.loads(image.info["kiri_metadata"])
            parameters = image.info["parameters"]
        self.assertEqual(api["12"]["inputs"]["text"], "")
        self.assertEqual(api["13"]["inputs"]["text"], "retained-negative-text")
        self.assertEqual(canvas["nodes"][1]["widgets_values"], [""])
        self.assertEqual(canvas["nodes"][2]["widgets_values"], ["retained-negative-text"])
        self.assertNotIn("positive", record["graph"]["prompts"])
        self.assertEqual(record["graph"]["prompts"]["negative"], "retained-negative-text")
        self.assertNotIn("private-positive-text", parameters)
        self.assertIn("Negative prompt: retained-negative-text", parameters)
        self.assertIn("Steps: 32", parameters)

    def test_workflow_on_negative_prompt_off_keeps_positive_prompt_and_graph_settings(self):
        import json

        image_value = torch.rand((1, 6, 7, 3))
        prompt = {
            "10": {"class_type": "KSampler", "inputs": {"positive": ["12", 0], "negative": ["13", 0], "steps": 28, "seed": 123}},
            "12": {"class_type": "CLIPTextEncode", "inputs": {"text": "retained-positive-text"}},
            "13": {"class_type": "CLIPTextEncode", "inputs": {"text": "private-negative-text"}},
            "99": {"class_type": "KiriImageLoader", "inputs": {"image_in": ["10", 0]}},
        }
        workflow = {"nodes": [
            {"id": 10, "type": "KSampler", "widgets_values": [28, 123]},
            {"id": 12, "type": "CLIPTextEncode", "widgets_values": ["retained-positive-text"]},
            {"id": 13, "type": "CLIPTextEncode", "widgets_values": ["private-negative-text"]},
        ], "links": []}
        settings = json.dumps({
            "save_positive_prompt": True, "save_negative_prompt": False,
            "platform_metadata": False,
        })
        self.load_node_with_saving(
            "disk.png", image_value, None, "workflow-on-negative-off", "", True, None,
            prompt, {"workflow": workflow}, None, metadata_export_settings=settings,
        )
        saved = sorted(self.output_dir.glob("Kirin Image*.png"), key=lambda p: p.stat().st_mtime_ns)[-1]
        with Image.open(saved) as image:
            api = json.loads(image.info["prompt"])
            canvas = json.loads(image.info["workflow"])
            record = json.loads(image.info["kiri_metadata"])
            parameters = image.info["parameters"]
        self.assertEqual(api["12"]["inputs"]["text"], "retained-positive-text")
        self.assertEqual(api["13"]["inputs"]["text"], "")
        self.assertEqual(canvas["nodes"][1]["widgets_values"], ["retained-positive-text"])
        self.assertEqual(canvas["nodes"][2]["widgets_values"], [""])
        self.assertEqual(record["graph"]["prompts"]["positive"], "retained-positive-text")
        self.assertNotIn("negative", record["graph"]["prompts"])
        self.assertIn("retained-positive-text", parameters)
        self.assertNotIn("private-negative-text", parameters)
        self.assertIn("Steps: 28", parameters)

    def test_text_only_metadata_editor_does_not_reintroduce_json(self):
        entries = [
            {"keyword": "parameters", "value": "old positive\nNegative prompt: old negative\nSteps: 20, Seed: 10"},
            {"keyword": "Kiri Nodes Used", "value": "Custom Sampler\nCustom Prompt Node"},
            {"keyword": "Kiri Metadata Mode", "value": "ComfyUI Workflow OFF; Other-Platform Metadata OFF; no ComfyUI graph data"},
            {"keyword": "comment", "value": "must be removed rather than serialized in strict mode"},
        ]
        updated = self.package._apply_structured_editor_values(entries, {
            "positive_prompt": "new positive",
            "negative_prompt": "new negative",
            "parameters": [
                {"kind": "sampler", "sampler_index": 0, "key": "steps", "label": "Steps", "value": "42"},
                {"kind": "sampler", "sampler_index": 0, "key": "seed", "label": "Seed", "value": "999"},
            ],
            "image_width": "32", "image_height": "24",
        })
        self.assertEqual({item["keyword"] for item in updated}, {"parameters", "Kiri Nodes Used", "Kiri Metadata Mode"})
        values = {item["keyword"]: item["value"] for item in updated}
        self.assertIn("new positive", values["parameters"])
        self.assertIn("Negative prompt: new negative", values["parameters"])
        self.assertIn("Steps: 42", values["parameters"])
        self.assertIn("Seed: 999", values["parameters"])
        self.assertIn("Size: 32x24", values["parameters"])
        self.assertEqual(values["Kiri Nodes Used"], "Custom Sampler\nCustom Prompt Node")
        self.assertEqual(values["Kiri Metadata Mode"], "ComfyUI Workflow OFF; Other-Platform Metadata OFF; no ComfyUI graph data")
        self.assertTrue(all("json" not in item["keyword"].lower() for item in updated))

    def test_mask_primary_input_is_saved_as_grayscale_and_passthrough_is_unchanged(self):
        mask = torch.rand((2, 5, 6))
        result = self.load_node_with_saving("disk.png", mask, None, "save-mask")
        self.assertIs(result["result"][0], mask)
        saved = sorted(self.output_dir.glob("Kirin Image*.png"), key=lambda p: p.stat().st_mtime_ns)[-1]
        with Image.open(saved) as image:
            self.assertEqual(image.mode, "L")
            self.assertEqual(image.size, (6, 5))

    def test_optional_socket_types_are_exposed(self):
        inputs = self.node_class.INPUT_TYPES()
        self.assertIn("image_in", inputs["optional"])
        self.assertEqual(inputs["optional"]["image_in"][0], "IMAGE,MASK")
        self.assertIn("mask_in", inputs["optional"])
        self.assertEqual(inputs["optional"]["mask_in"][0], "IMAGE,MASK")
        self.assertEqual(self.node_class.RETURN_TYPES, ("IMAGE,MASK", "IMAGE,MASK", "IMAGE"))
        self.assertEqual(self.node_class.RETURN_NAMES, ("A", "B", "C"))

    def test_both_connected_inputs_pass_through_exactly(self):
        image_value = torch.rand((1, 5, 7, 3))
        mask_value = torch.rand((1, 5, 7))
        result = self.load_node_with_saving("does-not-exist.png", image_value, mask_value, "node-both")
        output_a, output_b, output_c = result["result"]
        self.assertIs(output_a, image_value)
        self.assertIs(output_b, mask_value)
        payload = result["ui"]
        self.assertEqual(payload["kiri_preview_mode"], ["compare"])
        self.assertEqual(payload["kiri_preview_a_kind"], ["IMAGE"])
        self.assertEqual(payload["kiri_preview_b_kind"], ["MASK"])
        decoded_a = Image.open(io.BytesIO(base64.b64decode(payload["kiri_preview_a"][0])))
        decoded_b = Image.open(io.BytesIO(base64.b64decode(payload["kiri_preview_b"][0])))
        self.assertEqual(decoded_a.size, (7, 5))
        self.assertEqual(decoded_b.size, (7, 5))
        self.assertEqual(decoded_a.format, "JPEG")
        self.assertEqual(decoded_b.format, "JPEG")
        self.assertEqual(list(self.temp_preview_dir.iterdir()), [], "Live previews should not be written to disk.")

    def test_image_input_overrides_only_image_output(self):
        image_value = torch.rand((1, 2, 3, 3))
        result = self.load_node_with_saving("disk.png", image_value, None, "node-a")
        output_a, output_b, output_c = result["result"]
        self.assertIs(output_a, image_value)
        self.assertEqual(tuple(output_b.shape), (1, 2, 3, 3))
        self.assertTrue(torch.equal(output_b, output_c), "Unconnected B defaults to the selected Loaded Image, not its alpha mask.")
        payload = result["ui"]
        self.assertEqual(payload["kiri_preview_mode"], ["single"])
        self.assertTrue(payload["kiri_preview_a"][0], "Inline base64 image data must be present in the execution UI payload.")
        self.assertEqual(payload["kiri_preview_a_kind"], ["IMAGE"])
        self.assertNotIn("kiri_preview_b", payload)

    def test_image_socket_can_passthrough_a_mask_without_conversion(self):
        mask_value = torch.rand((1, 4, 6))
        result = self.load_node_with_saving("disk.png", mask_value, None, "mask-on-a")
        output_a, _, _ = result["result"]
        self.assertIs(output_a, mask_value)
        self.assertEqual(result["ui"]["kiri_preview_a_kind"], ["MASK"])

    def test_mask_socket_can_passthrough_an_image_without_conversion(self):
        image_value = torch.rand((1, 4, 6, 3))
        result = self.load_node_with_saving("disk.png", None, image_value, "image-on-b")
        _, output_b, _ = result["result"]
        self.assertIs(output_b, image_value)
        self.assertEqual(result["ui"]["kiri_preview_b_kind"], ["IMAGE"])

    def test_both_connected_inputs_do_not_require_selected_file(self):
        image_value = torch.rand((1, 5, 7, 3))
        mask_value = torch.rand((1, 5, 7))
        result = self.load_node_with_saving("missing-file.png", image_value, mask_value, "no-file")
        output_a, output_b, output_c = result["result"]
        self.assertIs(output_a, image_value)
        self.assertIs(output_b, mask_value)

    def test_mask_input_overrides_only_mask_output(self):
        mask_value = torch.rand((1, 4, 6))
        result = self.load_node_with_saving("disk.png", None, mask_value, "node-mask")
        output_a, output_b, output_c = result["result"]
        self.assertIs(output_b, mask_value)
        self.assertEqual(tuple(output_a.shape), (1, 2, 3, 3))
        payload = result["ui"]
        self.assertEqual(payload["kiri_preview_mode"], ["single"])
        self.assertNotIn("kiri_preview_a", payload)
        self.assertEqual(payload["kiri_preview_b_kind"], ["MASK"])


    def test_zero_sized_connected_inputs_return_neutral_outputs_without_saving(self):
        empty_image = torch.empty((0, 2, 3, 3), dtype=torch.float32)
        empty_mask = torch.empty((0, 2, 3), dtype=torch.float32)
        result = self.load_node_with_saving("disk.png", empty_image, empty_mask, "empty-both-connected")
        output_a, output_b, output_c = result["result"]
        self.assertEqual(tuple(output_a.shape), (1, 2, 3, 3), "Placeholders should retain spatial dimensions from empty connected IMAGE tensors.")
        self.assertEqual(tuple(output_b.shape), (1, 2, 3), "Placeholders should retain spatial dimensions from empty connected MASK tensors.")
        self.assertTrue(torch.isfinite(output_a).all())
        self.assertTrue(torch.isfinite(output_b).all())
        self.assertEqual(result["ui"]["kiri_preview_mode"], ["file"])
        self.assertNotIn("kiri_preview_a_kind", result["ui"], "Neutral placeholders must not masquerade as real previews.")
        self.assertNotIn("kiri_preview_b_kind", result["ui"], "Neutral placeholders must not masquerade as real previews.")
        self.assertIn("kiri_save_skipped", result["ui"], "An empty connected A must not save the node's unrelated selected file.")
        self.assertNotIn("kiri_saved_files", result["ui"])
        self.assertNotIn("kiri_save_error", result["ui"])

    def test_empty_vae_decode_and_empty_chained_kiri_outputs_are_safe_together(self):
        prompt = {
            "900": {"class_type": "KiriImageLoader", "inputs": {
                "image_in": ["901", 0], "mask_in": ["902", 1],
            }},
            "901": {"class_type": "KiriImageLoader", "inputs": {
                "image_in": ["903", 0],
            }},
            "902": {"class_type": "KiriImageLoader", "inputs": {
                "mask_in": ["904", 0],
            }},
            "903": {"class_type": "VAEDecode", "inputs": {}},
            "904": {"class_type": "VAEDecode", "inputs": {}},
        }
        extra_pnginfo = {"workflow": {
            "nodes": [{"id": 900, "inputs": [
                {"name": "image_in", "slot_index": 0, "link": 1},
                {"name": "mask_in", "slot_index": 1, "link": 2},
            ]}],
            "links": [
                [1, 901, 0, 900, 0, "IMAGE"],
                [2, 902, 1, 900, 1, "MASK"],
            ],
        }}
        empty_image_from_vae = torch.empty((0, 8, 9, 3), dtype=torch.float32)
        empty_mask_from_chained_kiri = torch.empty((0, 8, 9), dtype=torch.float32)
        result = self.load_node_with_saving(
            "disk.png", empty_image_from_vae, empty_mask_from_chained_kiri,
            "900", prompt=prompt, extra_pnginfo=extra_pnginfo,
        )
        output_a, output_b, output_c = result["result"]
        self.assertEqual(tuple(output_a.shape), (1, 8, 9, 3))
        self.assertEqual(tuple(output_b.shape), (1, 8, 9))
        self.assertEqual(output_a.count_nonzero().item(), 0)
        self.assertEqual(output_b.count_nonzero().item(), 0)
        self.assertEqual(result["ui"]["kiri_preview_mode"], ["file"])
        self.assertIn("kiri_save_skipped", result["ui"])
        self.assertNotIn("kiri_saved_files", result["ui"])
        self.assertNotIn("kiri_save_error", result["ui"])

    def test_empty_vae_decode_with_valid_chained_kiri_b_preserves_b(self):
        prompt = {
            "910": {"class_type": "KiriImageLoader", "inputs": {
                "image_in": ["911", 0], "mask_in": ["912", 1],
            }},
            "911": {"class_type": "VAEDecode", "inputs": {}},
            "912": {"class_type": "KiriImageLoader", "inputs": {}},
        }
        extra_pnginfo = {"workflow": {
            "nodes": [{"id": 910, "inputs": [
                {"name": "image_in", "slot_index": 0, "link": 11},
                {"name": "mask_in", "slot_index": 1, "link": 12},
            ]}],
            "links": [
                [11, 911, 0, 910, 0, "IMAGE"],
                [12, 912, 1, 910, 1, "IMAGE"],
            ],
        }}
        empty_image = torch.empty((0, 8, 9, 3), dtype=torch.float32)
        valid_kiri_b = torch.rand((1, 8, 9, 3), dtype=torch.float32)
        result = self.load_node_with_saving(
            "disk.png", empty_image, valid_kiri_b, "910",
            prompt=prompt, extra_pnginfo=extra_pnginfo,
        )
        output_a, output_b, output_c = result["result"]
        self.assertEqual(tuple(output_a.shape), (1, 8, 9, 3))
        self.assertEqual(output_a.count_nonzero().item(), 0)
        self.assertIs(output_b, valid_kiri_b)
        self.assertEqual(result["ui"]["kiri_preview_mode"], ["single"])
        self.assertIn("kiri_save_skipped", result["ui"])
        self.assertNotIn("kiri_save_error", result["ui"])

    def test_connected_empty_a_does_not_prevent_unconnected_b_loaded_image_fallback(self):
        prompt = {
            "920": {"class_type": "KiriImageLoader", "inputs": {
                "image_in": ["921", 0],
            }},
            "921": {"class_type": "VAEDecode", "inputs": {}},
        }
        empty_image = torch.empty((0, 8, 9, 3), dtype=torch.float32)
        result = self.load_node_with_saving(
            "disk.png", empty_image, None, "920", prompt=prompt,
        )
        output_a, output_b, output_c = result["result"]
        self.assertEqual(tuple(output_a.shape), (1, 8, 9, 3))
        self.assertEqual(output_a.count_nonzero().item(), 0)
        expected_image, _expected_mask = self.package._load_disk_image_and_mask(
            self.package._resolve_image_path("disk.png")
        )
        self.assertTrue(torch.equal(output_b, expected_image))
        self.assertIn("kiri_save_skipped", result["ui"])
        self.assertNotIn("kiri_saved_files", result["ui"])

    def test_empty_image_branch_does_not_discard_valid_mask_branch(self):
        empty_image = torch.empty((0, 2, 3, 3), dtype=torch.float32)
        valid_mask = torch.rand((1, 2, 3), dtype=torch.float32)
        result = self.load_node_with_saving("disk.png", empty_image, valid_mask, "empty-a-valid-b")
        output_a, output_b, output_c = result["result"]
        self.assertEqual(tuple(output_a.shape), (1, 2, 3, 3), "The empty connected IMAGE branch should receive a neutral image shaped from B.")
        self.assertTrue(torch.count_nonzero(output_a).item() == 0)
        self.assertIs(output_b, valid_mask, "A usable B input must pass through unchanged when A is empty.")
        self.assertEqual(result["ui"]["kiri_preview_mode"], ["single"])
        self.assertIn("kiri_save_skipped", result["ui"])
        self.assertNotIn("kiri_save_error", result["ui"])

    def test_empty_mask_branch_does_not_discard_valid_image_branch(self):
        valid_image = torch.rand((1, 2, 3, 3), dtype=torch.float32)
        empty_mask = torch.empty((0, 2, 3), dtype=torch.float32)
        result = self.load_node_with_saving("disk.png", valid_image, empty_mask, "valid-a-empty-b")
        output_a, output_b, output_c = result["result"]
        self.assertIs(output_a, valid_image, "A usable A input must pass through unchanged when B is empty.")
        self.assertEqual(tuple(output_b.shape), (1, 2, 3), "The empty connected MASK branch should receive a neutral mask shaped from A.")
        self.assertTrue(torch.count_nonzero(output_b).item() == 0)
        self.assertEqual(result["ui"]["kiri_preview_mode"], ["single"])
        self.assertIn("kiri_saved_files", result["ui"], "A real A image can still be saved when only B is empty.")
        self.assertNotIn("kiri_save_error", result["ui"])

    def test_none_inputs_with_workflow_links_return_typed_neutral_fallbacks(self):
        prompt = {
            "999": {"class_type": "KiriImageLoader", "inputs": {
                "image_in": ["10", 0], "mask_in": ["11", 0],
            }},
            "10": {"class_type": "VAEDecode", "inputs": {}},
            "11": {"class_type": "ImagePassthrough", "inputs": {}},
        }
        extra_pnginfo = {"workflow": {
            "nodes": [{"id": 999, "inputs": [
                {"name": "image_in", "slot_index": 0, "link": 1},
                {"name": "mask_in", "slot_index": 1, "link": 2},
            ]}],
            "links": [
                [1, 10, 0, 999, 0, "IMAGE"],
                [2, 11, 0, 999, 1, "MASK"],
            ],
        }}
        result = self.load_node_with_saving(
            "missing-input.png", None, None, "999", prompt=prompt, extra_pnginfo=extra_pnginfo,
        )
        output_a, output_b, output_c = result["result"]
        self.assertEqual(tuple(output_a.shape), (1, 64, 64, 3))
        self.assertEqual(tuple(output_b.shape), (1, 64, 64))
        self.assertTrue(torch.count_nonzero(output_a).item() == 0)
        self.assertTrue(torch.count_nonzero(output_b).item() == 0)
        self.assertEqual(result["ui"]["kiri_preview_mode"], ["file"])
        self.assertIn("kiri_save_skipped", result["ui"])
        self.assertNotIn("kiri_save_error", result["ui"])

    def test_connected_none_inputs_do_not_save_the_selected_fallback_image(self):
        prompt = {
            "997": {"class_type": "KiriImageLoader", "inputs": {
                "image_in": ["10", 0], "mask_in": ["11", 0],
            }},
            "10": {"class_type": "VAEDecode", "inputs": {}},
            "11": {"class_type": "BypassedImageSource", "inputs": {}},
        }
        extra_pnginfo = {"workflow": {
            "nodes": [{"id": 997, "inputs": [
                {"name": "image_in", "slot_index": 0, "link": 1},
                {"name": "mask_in", "slot_index": 1, "link": 2},
            ]}],
            "links": [
                [1, 10, 0, 997, 0, "IMAGE"],
                [2, 11, 0, 997, 1, "MASK"],
            ],
        }}
        result = self.load_node_with_saving(
            "disk.png", None, None, "997", prompt=prompt, extra_pnginfo=extra_pnginfo,
        )
        output_a, output_b, output_c = result["result"]
        self.assertEqual(tuple(output_a.shape), (1, 64, 64, 3))
        self.assertEqual(tuple(output_b.shape), (1, 64, 64))
        self.assertTrue(torch.count_nonzero(output_a).item() == 0)
        self.assertTrue(torch.count_nonzero(output_b).item() == 0)
        self.assertIn("kiri_save_skipped", result["ui"])
        self.assertNotIn("kiri_saved_files", result["ui"])
        self.assertNotIn("kiri_save_error", result["ui"])

    def test_workflow_socket_types_choose_correct_neutral_tensor_type(self):
        prompt = {
            "998": {"class_type": "KiriImageLoader", "inputs": {
                "image_in": ["10", 0], "mask_in": ["11", 0],
            }},
            "10": {"class_type": "EmptyMaskSource", "inputs": {}},
            "11": {"class_type": "VAEDecode", "inputs": {}},
        }
        extra_pnginfo = {"workflow": {
            "nodes": [{"id": 998, "inputs": [
                {"name": "image_in", "slot_index": 0, "link": 1},
                {"name": "mask_in", "slot_index": 1, "link": 2},
            ]}],
            "links": [
                [1, 10, 0, 998, 0, "MASK"],
                [2, 11, 0, 998, 1, "IMAGE"],
            ],
        }}
        result = self.load_node_with_saving(
            "missing-input.png", None, None, "998", prompt=prompt, extra_pnginfo=extra_pnginfo,
        )
        output_a, output_b, output_c = result["result"]
        self.assertEqual(tuple(output_a.shape), (1, 64, 64), "A follows the upstream MASK socket type, not its default label.")
        self.assertEqual(tuple(output_b.shape), (1, 64, 64, 3), "B follows the upstream IMAGE socket type, not its default label.")
        self.assertEqual(result["ui"]["kiri_preview_mode"], ["file"])

    def test_zero_dimension_tensor_with_unavailable_selected_file_returns_safe_placeholder(self):
        empty_image = torch.empty((1, 0, 6, 3), dtype=torch.float32)
        result = self.load_node_with_saving("missing-input.png", empty_image, None, "zero-height-image")
        output_a, output_b, output_c = result["result"]
        self.assertEqual(tuple(output_a.shape), (1, 64, 64, 3))
        self.assertEqual(tuple(output_b.shape), (1, 64, 64, 3))
        self.assertTrue(torch.count_nonzero(output_a).item() == 0)
        self.assertTrue(torch.isfinite(output_a).all())
        self.assertTrue(torch.isfinite(output_b).all())
        self.assertIn("kiri_save_skipped", result["ui"])
        self.assertNotIn("kiri_save_error", result["ui"])

    def test_default_file_load_returns_loaded_image_for_both_unconnected_branches(self):
        result = self.load_node_with_saving("disk.png")
        output_a, output_b, output_c = result["result"]
        self.assertEqual(tuple(output_a.shape), (1, 2, 3, 3))
        self.assertEqual(tuple(output_b.shape), (1, 2, 3, 3))
        self.assertTrue(torch.equal(output_a, output_c))
        self.assertTrue(torch.equal(output_b, output_c))
        self.assertEqual(result["ui"]["kiri_preview_mode"], ["file"])

    def test_save_target_b_saves_b_with_branch_metadata_and_no_legacy_a_alias(self):
        image_b = torch.rand((1, 6, 8, 3))
        settings = json.dumps({"save_images_from": "b"})
        result = self.node_class().load_image(
            "disk.png", None, image_b, "save-only-b", prompt={}, extra_pnginfo={}, embed_workflow=True,
            save_path=str(self.output_dir), filename_template="Branch [Counter]",
            metadata_export_settings=settings,
        )
        output_a, output_b, _output_c = result["result"]
        self.assertIs(output_b, image_b)
        self.assertIn("kiri_saved_files_b", result["ui"])
        self.assertNotIn("kiri_saved_files_a", result["ui"])
        self.assertNotIn("kiri_saved_files", result["ui"], "A legacy file alias must never mislabel a B-only save as A.")
        record = result["ui"]["kiri_saved_files_b"][0][0]
        with Image.open(self.output_dir / record["filename"]) as saved:
            self.assertEqual(saved.size, (8, 6))
            kiri_record = json.loads(saved.info["kiri_metadata"])
            self.assertEqual(kiri_record["source_input"], "B")
            self.assertEqual(kiri_record["saved_input"], "B")

    def test_save_target_both_saves_each_branch_and_uses_numbered_collision_suffix(self):
        image_a = torch.rand((1, 6, 8, 3))
        image_b = torch.rand((1, 6, 8, 3))
        settings = json.dumps({"save_images_from": "both"})
        result = self.node_class().load_image(
            "disk.png", image_a, image_b, "save-both", prompt={}, extra_pnginfo={},
            save_path=str(self.output_dir), filename_template="Same Name",
            metadata_export_settings=settings,
        )
        self.assertIn("kiri_saved_files_a", result["ui"])
        self.assertIn("kiri_saved_files_b", result["ui"])
        record_a = result["ui"]["kiri_saved_files_a"][0][0]
        record_b = result["ui"]["kiri_saved_files_b"][0][0]
        self.assertNotEqual(record_a["filename"], record_b["filename"])
        self.assertEqual(record_a["filename"], "Same Name.png")
        self.assertRegex(record_b["filename"], r"^Same Name_\d{5}_.png$")
        self.assertTrue((self.output_dir / record_a["filename"]).is_file())
        self.assertTrue((self.output_dir / record_b["filename"]).is_file())

    def test_save_target_b_skips_connected_empty_branch_without_affecting_a_or_c(self):
        empty_b = torch.empty((0, 6, 8, 3))
        prompt = {"save-empty-b": {"class_type": "KiriImageLoader", "inputs": {"mask_in": ["decode", 0]}},
                  "decode": {"class_type": "VAEDecode", "inputs": {}}}
        settings = json.dumps({"save_images_from": "b"})
        before = set(self.output_dir.glob("*.png"))
        result = self.node_class().load_image(
            "disk.png", None, empty_b, "save-empty-b", prompt=prompt,
            save_path=str(self.output_dir), filename_template="Should Not Save",
            metadata_export_settings=settings,
        )
        self.assertIn("kiri_save_skipped", result["ui"])
        self.assertNotIn("kiri_saved_files_b", result["ui"])
        self.assertEqual(set(self.output_dir.glob("*.png")), before)
        self.assertEqual(tuple(result["result"][1].shape), (1, 6, 8, 3))
        self.assertTrue(torch.count_nonzero(result["result"][1]).item() == 0)
        self.assertTrue(torch.equal(result["result"][2], self.package._load_disk_image_and_mask(self.package._resolve_image_path("disk.png"))[0]))

    def test_save_is_disabled_when_save_path_and_filename_are_blank(self):
        image_value = torch.rand((1, 4, 5, 3))
        before = set(self.output_dir.glob("*.png"))
        result = self.node_class().load_image("disk.png", image_value, None, "save-off-default")
        after = set(self.output_dir.glob("*.png"))
        self.assertIs(result["result"][0], image_value)
        self.assertEqual(len(result["result"]), 3)
        self.assertEqual(after, before, "Blank save fields must never create an output file.")
        self.assertIn("kiri_save_skipped", result["ui"])
        self.assertNotIn("kiri_saved_files", result["ui"])

    def test_save_requires_both_nonblank_path_and_filename(self):
        image_value = torch.rand((1, 4, 5, 3))
        before = set(self.output_dir.glob("*.png"))
        path_only = self.node_class().load_image(
            "disk.png", image_value, None, "save-path-only", save_path=str(self.output_dir), filename_template=""
        )
        name_only = self.node_class().load_image(
            "disk.png", image_value, None, "save-name-only", save_path="", filename_template="Kirin Image [Seed]"
        )
        self.assertEqual(set(self.output_dir.glob("*.png")), before)
        self.assertIn("kiri_save_skipped", path_only["ui"])
        self.assertIn("kiri_save_skipped", name_only["ui"])
        self.assertIs(path_only["result"][0], image_value)
        self.assertIs(name_only["result"][0], image_value)

    def test_output_c_is_loaded_image_when_a_and_b_are_connected(self):
        image_a = torch.rand((1, 5, 7, 3))
        image_b = torch.rand((1, 5, 7, 3))
        result = self.node_class().load_image("disk.png", image_a, image_b, "loaded-output-c")
        output_a, output_b, output_c = result["result"]
        self.assertIs(output_a, image_a)
        self.assertIs(output_b, image_b)
        self.assertEqual(tuple(output_c.shape), (1, 2, 3, 3))
        expected, _mask = self.package._load_disk_image_and_mask(self.package._resolve_image_path("disk.png"))
        self.assertTrue(torch.equal(output_c, expected))
        self.assertIsNot(output_c, image_a)
        self.assertIsNot(output_c, image_b)

    def test_outputs_a_and_b_both_default_to_loaded_image_when_unconnected_and_c_is_independent(self):
        result = self.node_class().load_image("disk.png", unique_id="loaded-fallback-a-b-c")
        output_a, output_b, output_c = result["result"]
        expected_image, _expected_mask = self.package._load_disk_image_and_mask(self.package._resolve_image_path("disk.png"))
        self.assertTrue(torch.equal(output_a, expected_image))
        self.assertTrue(torch.equal(output_b, expected_image))
        self.assertTrue(torch.equal(output_c, expected_image))

    def test_output_c_is_neutral_image_when_loaded_image_is_missing(self):
        result = self.node_class().load_image("missing-file.png", unique_id="missing-loaded-c")
        output_a, output_b, output_c = result["result"]
        self.assertEqual(tuple(output_c.shape), (1, 64, 64, 3))
        self.assertTrue(torch.isfinite(output_c).all())
        self.assertEqual(tuple(output_a.shape), (1, 64, 64, 3))
        self.assertEqual(tuple(output_b.shape), (1, 64, 64, 3))

    def test_kiri_output_c_metadata_tracks_loaded_image_without_a_or_b_branches(self):
        prompt = {
            "10": {"class_type": "KiriImageLoader", "inputs": {"image_in": ["20", 0], "mask_in": ["30", 1], "image": "disk.png"}},
            "20": {"class_type": "VAEDecode", "inputs": {}},
            "30": {"class_type": "LoadImage", "inputs": {"image": "other.png"}},
        }
        nodes, node_map, _root = self.package._collect_active_upstream_nodes(
            {"99": {"class_type": "Downstream", "inputs": {"image_in": ["10", 2]}}, **prompt},
            "99", "image_in", workflow=None,
        )
        self.assertEqual([item["class_type"] for item in nodes], ["KiriImageLoader"])
        self.assertEqual(node_map["10"]["inputs"], {"image": "disk.png"})

    def test_kiri_output_a_metadata_traces_only_image_a_branch(self):
        prompt = {
            "99": {"class_type": "Downstream", "inputs": {"image_in": ["10", 0]}},
            "10": {"class_type": "KiriImageLoader", "inputs": {"image_in": ["20", 0], "mask_in": ["30", 0], "image": "disk.png"}},
            "20": {"class_type": "VAEDecode", "inputs": {}},
            "30": {"class_type": "UnrelatedMaskBranch", "inputs": {}},
        }
        nodes, _map, _root = self.package._collect_active_upstream_nodes(prompt, "99", "image_in", workflow=None)
        self.assertEqual({item["class_type"] for item in nodes}, {"KiriImageLoader", "VAEDecode"})
        self.assertNotIn("UnrelatedMaskBranch", {item["class_type"] for item in nodes})
        kiri_node = next(item for item in nodes if item["class_type"] == "KiriImageLoader")
        self.assertEqual(set(kiri_node["inputs"]), {"image_in"}, "When A is connected, do not record the unused selected-file fallback.")

    def test_kiri_output_b_metadata_traces_only_image_b_branch(self):
        prompt = {
            "99": {"class_type": "Downstream", "inputs": {"image_in": ["10", 1]}},
            "10": {"class_type": "KiriImageLoader", "inputs": {"image_in": ["20", 0], "mask_in": ["30", 0], "image": "disk.png"}},
            "20": {"class_type": "UnrelatedImageBranch", "inputs": {}},
            "30": {"class_type": "VAEDecode", "inputs": {}},
        }
        nodes, _map, _root = self.package._collect_active_upstream_nodes(prompt, "99", "image_in", workflow=None)
        self.assertEqual({item["class_type"] for item in nodes}, {"KiriImageLoader", "VAEDecode"})
        self.assertNotIn("UnrelatedImageBranch", {item["class_type"] for item in nodes})
        kiri_node = next(item for item in nodes if item["class_type"] == "KiriImageLoader")
        self.assertEqual(set(kiri_node["inputs"]), {"mask_in"}, "When B is connected, do not record the unused selected-file fallback.")

    def test_input_file_discovery_includes_nested_subfolders(self):
        nested = self.input_dir / "other-drive-style-folder" / "nested"
        nested.mkdir(parents=True)
        Image.new("RGB", (2, 2), (1, 2, 3)).save(nested / "inside.webp")
        files = self.package._get_input_files()
        self.assertIn("disk.png", files)
        self.assertIn("other-drive-style-folder/nested/inside.webp", files)

    def test_resolver_opens_nested_input_path_under_configured_root(self):
        nested = self.input_dir / "subfolder"
        nested.mkdir()
        image_path = nested / "nested.png"
        Image.new("RGB", (2, 2), (1, 2, 3)).save(image_path)
        resolved = self.package._resolve_image_path("subfolder/nested.png")
        self.assertEqual(Path(resolved), image_path.resolve())

    def test_windows_explorer_select_argument_quotes_paths_with_spaces(self):
        path = r"D:\My stuff and gens\ComfyUI\input\nested folder\image file.png"
        self.assertEqual(
            self.package._windows_explorer_select_parameters(path),
            r'/select,"D:\My stuff and gens\ComfyUI\input\nested folder\image file.png"',
        )

    def test_managed_output_image_path_resolves_for_connected_generated_image(self):
        output_image = self.output_dir / "generated.png"
        Image.new("RGB", (2, 3), (10, 20, 30)).save(output_image)
        resolved = self.package._resolve_managed_image_path("generated.png", "output", "")
        self.assertEqual(Path(resolved), output_image.resolve())

    def test_managed_output_image_path_supports_subfolders(self):
        nested = self.output_dir / "batch_001"
        nested.mkdir()
        output_image = nested / "preview.webp"
        Image.new("RGB", (2, 3), (10, 20, 30)).save(output_image)
        resolved = self.package._resolve_managed_image_path("preview.webp", "output", "batch_001")
        self.assertEqual(Path(resolved), output_image.resolve())

    def test_managed_output_image_resolver_rejects_path_traversal(self):
        outside = self.input_dir / "outside.png"
        Image.new("RGB", (1, 1)).save(outside)
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.package._resolve_managed_image_path("../outside.png", "output", "")

    def test_managed_output_image_resolver_rejects_symlink_escape(self):
        outside = self.input_dir / "outside-output.png"
        Image.new("RGB", (1, 1)).save(outside)
        link_dir = self.output_dir / "external-link"
        try:
            link_dir.symlink_to(self.input_dir, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("Symlink creation is unavailable in this environment.")
        try:
            with self.assertRaises(FileNotFoundError):
                self.package._resolve_managed_image_path("outside-output.png", "output", "external-link")
        finally:
            link_dir.unlink(missing_ok=True)

    def test_managed_resolver_rejects_unknown_comfy_folder_type(self):
        with self.assertRaises(ValueError):
            self.package._resolve_managed_image_path("disk.png", "arbitrary", "")

    def test_resolver_rejects_paths_outside_input_directory(self):
        outside_name = f"kiri-outside-{os.getpid()}.png"
        outside_path = self.input_dir.parent / outside_name
        Image.new("RGB", (1, 1)).save(outside_path)
        try:
            with self.assertRaises(FileNotFoundError):
                self.package._resolve_image_path(f"../{outside_name}")
        finally:
            outside_path.unlink(missing_ok=True)

    def test_delete_action_moves_only_selected_input_file_to_trash(self):
        trash_dir = self.input_dir / "test-trash"
        trash_dir.mkdir()
        moved = []

        def fake_move_to_trash(path):
            destination = trash_dir / Path(path).name
            os.replace(path, destination)
            moved.append(destination)

        original = self.package._move_to_trash
        self.package._move_to_trash = fake_move_to_trash
        try:
            class FakeRequest:
                async def json(self):
                    return {"filename": "disk.png", "action": "delete"}

            response = __import__("asyncio").run(self.package.kiri_image_loader_file_action(FakeRequest()))
        finally:
            self.package._move_to_trash = original

        self.assertEqual(response.status, 200)
        self.assertFalse((self.input_dir / "disk.png").exists())
        self.assertEqual(len(moved), 1)
        self.assertTrue(moved[0].is_file())
        self.assertIn("Recycle Bin / Trash", response.text)


    def test_bypassed_upstream_nodes_are_skipped_and_the_matching_input_is_traced(self):
        prompt = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "base.safetensors"}},
            "2": {"class_type": "LoraLoader", "inputs": {"model": ["1", 0], "clip": ["1", 1], "lora_name": "bypassed.safetensors", "strength_model": 0.5}},
            "3": {"class_type": "KSampler", "inputs": {"model": ["2", 0], "seed": 42, "steps": 20}},
            "4": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["1", 2]}},
            "9": {"class_type": "KiriImageLoader", "inputs": {"image_in": ["4", 0]}},
        }
        workflow = {
            "nodes": [
                {"id": 1, "type": "CheckpointLoaderSimple", "mode": 0},
                {"id": 2, "type": "LoraLoader", "mode": 4},
                {"id": 3, "type": "KSampler", "mode": 0},
                {"id": 4, "type": "VAEDecode", "mode": 0},
            ],
            "links": [
                [1, 1, 0, 2, 0, "MODEL"],
                [2, 1, 1, 2, 1, "CLIP"],
                [3, 2, 0, 3, 0, "MODEL"],
                [4, 3, 0, 4, 0, "LATENT"],
                [5, 1, 2, 4, 1, "VAE"],
                [6, 4, 0, 9, 0, "IMAGE"],
            ],
        }
        active_nodes, node_map, _root_id = self.package._collect_active_upstream_nodes(
            prompt, "9", "image_in", workflow=workflow,
        )
        ids = {str(node["id"]) for node in active_nodes}
        self.assertIn("1", ids)
        self.assertIn("3", ids)
        self.assertIn("4", ids)
        self.assertNotIn("2", ids, "Mode 4 bypassed nodes must not be embedded as active contributors.")
        self.assertEqual(node_map["3"]["inputs"]["model"], ["1", 0], "A bypassed node's outgoing link should be rewritten through its matching input.")

    def test_registered_model_files_receive_full_sha256(self):
        import hashlib
        model = self.input_dir / "test-checkpoint.safetensors"
        payload = b"Kiri model hash regression fixture" * 100
        model.write_bytes(payload)
        previous = self.package.folder_paths.get_full_path
        self.package.folder_paths.get_full_path = lambda category, filename: str(model) if category == "checkpoints" and filename == model.name else None
        try:
            prompt = {
                "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": model.name}},
                "2": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["1", 2]}},
                "3": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "seed": 1}},
            }
            prompt["9"] = {"class_type": "KiriImageLoader", "inputs": {"image_in": ["2", 0]}}
            active_nodes, node_map, root_id = self.package._collect_active_upstream_nodes(
                prompt, "9", "image_in",
            )
            graph = self.package._extract_graph_metadata(active_nodes, node_map, {})
            resource = next(item for item in graph["model_resources"] if item.get("type") == "Checkpoint / Model")
            self.assertEqual(resource["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(root_id, "9")
        finally:
            self.package.folder_paths.get_full_path = previous

    def test_trash_helper_uses_send2trash_when_available(self):
        image_path = self.input_dir / "disk.png"
        called = []
        fake_send2trash = types.ModuleType("send2trash")
        fake_send2trash.send2trash = lambda path: called.append(path)
        previous = sys.modules.get("send2trash")
        sys.modules["send2trash"] = fake_send2trash
        try:
            self.package._move_to_trash(str(image_path))
        finally:
            if previous is None:
                sys.modules.pop("send2trash", None)
            else:
                sys.modules["send2trash"] = previous
        self.assertEqual(called, [str(image_path)])
        self.assertTrue(image_path.is_file(), "The trash helper must delegate rather than unlink the source itself.")


    def test_default_export_writes_native_xmp_novelai_and_easy_diffusion_task_sidecar(self):
        image_value = torch.full((1, 512, 512, 3), 0.4, dtype=torch.float32)
        prompt = {
            "10": {"class_type": "CLIPTextEncode", "inputs": {"text": "compatibility positive"}},
            "11": {"class_type": "CLIPTextEncode", "inputs": {"text": "compatibility negative"}},
            "12": {"class_type": "KSampler", "inputs": {
                "seed": 321, "steps": 26, "cfg": 6.25,
                "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
                "positive": ["10", 0], "negative": ["11", 0],
            }},
            "14": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0]}},
            "999": {"class_type": "KiriImageLoader", "inputs": {"image_in": ["14", 0]}},
        }
        result = self.load_node_with_saving(
            "disk.png", image_value, None, "999", embed_workflow=True,
            prompt=prompt,
        )
        record = result["ui"]["kiri_saved_files"][0][0]
        saved_path = self.output_dir / record["filename"]
        self.assertTrue(saved_path.is_file())
        self.assertEqual(record.get("easy_diffusion_task"), saved_path.stem + ".easy-diffusion.json")
        task_path = self.output_dir / record["easy_diffusion_task"]
        self.assertTrue(task_path.is_file(), "Easy Diffusion receives a directly loadable task JSON sidecar by default.")
        task = json.loads(task_path.read_text(encoding="utf-8"))
        self.assertEqual(task["reqBody"]["prompt"], "compatibility positive")
        self.assertEqual(task["reqBody"]["negative_prompt"], "compatibility negative")
        self.assertEqual(task["reqBody"]["seed"], 321)
        with Image.open(saved_path) as image:
            self.assertIn("kiri_metadata", image.info)
            self.assertIn("XML:com.adobe.xmp", image.info)
            self.assertIn("novelai_stealth", image.info)
        import importlib
        entries = importlib.import_module("KiriImageLoader.kiri_metadata").read_png_text_chunks(str(saved_path))
        xmp_entries = [item for item in entries if item["keyword"] == "XML:com.adobe.xmp"]
        self.assertEqual(len(xmp_entries), 1)
        self.assertEqual(xmp_entries[0]["format"], "iTXt", "Draw Things XMP must use PNG iTXt, not Latin-1 tEXt.")
        inspected = self.package.inspect_image(str(saved_path))
        items = inspected["copy_items"]
        self.assertTrue(any(item["label"] == "Positive Prompt" and item["value"] == "compatibility positive" for item in items))
        self.assertEqual(inspected["source"], "NovelAI")


if __name__ == "__main__":
    unittest.main()
