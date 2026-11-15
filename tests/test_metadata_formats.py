import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

MODULE_PATH = Path(__file__).resolve().parents[1] / "kiri_metadata.py"
_spec = importlib.util.spec_from_file_location("kiri_metadata_under_test", MODULE_PATH)
_metadata = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metadata)


class MetadataFormatTests(unittest.TestCase):
    def inspect_png(self, records):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.png"
            info = PngImagePlugin.PngInfo()
            for key, value in records.items():
                info.add_text(key, value if isinstance(value, str) else json.dumps(value))
            Image.new("RGB", (16, 16)).save(path, pnginfo=info)
            return _metadata.inspect_image(str(path))

    def item_values(self, result):
        return {item["label"]: item["value"] for item in result["copy_items"]}

    def test_swarmui_envelope(self):
        result = self.inspect_png({"parameters": {
            "sui_image_params": {
                "prompt": "swarm positive", "negativeprompt": "swarm negative",
                "seed": "1234", "steps": "28", "cfgscale": "6.5",
                "sampler": "DPM++ 2M", "scheduler": "karras", "model": "models/sdxl/base",
                "width": "1024", "height": "1024", "swarm_version": "0.9.9",
                "used_loras": [{"name": "test-lora", "strength": "0.7"}],
            },
            "sui_extra_data": {"date": "2026-11-02", "original_prompt": "dynamic prompt"},
            "sui_models": [
                {"name": "base.safetensors", "param": "model", "hash": "0x" + "a" * 64},
                {"name": "test-lora.safetensors", "param": "used_loras", "hash": "0x" + "b" * 64},
            ],
        }})
        values = self.item_values(result)
        self.assertEqual(result["source"], "SwarmUI")
        self.assertEqual(values["Positive Prompt"], "swarm positive")
        self.assertEqual(values["Negative Prompt"], "swarm negative")
        self.assertEqual(values["Original Prompt"], "dynamic prompt")
        self.assertEqual(values["Generation Date"], "2026-11-02")
        self.assertIn("0x" + "b" * 64, values["LoRA 1 Hash"])

    def test_invokeai_metadata_workflow_graph(self):
        result = self.inspect_png({
            "invokeai_metadata": {
                "positive_prompt": "invoke positive", "negative_prompt": "invoke negative",
                "seed": 2345, "steps": 32, "cfg_scale": 5.5, "width": 1024, "height": 768,
                "model": {"key": "models/sdxl", "hash": "modelhash", "name": "My Checkpoint", "base": "sdxl", "type": "main"},
                "loras": [{"model": {"key": "lora/l1", "hash": "lorahash", "name": "My LoRA", "base": "sdxl", "type": "lora"}, "weight": 0.65}],
            },
            "invokeai_workflow": {"nodes": [{"id": "node-a", "type": "denoise", "steps": 32}]},
            "invokeai_graph": {"nodes": {"1": {"id": "denoise", "type": "denoise", "steps": 32}}, "edges": []},
        })
        values = self.item_values(result)
        self.assertEqual(result["source"], "InvokeAI")
        self.assertEqual(values["Checkpoint"], "My Checkpoint")
        self.assertEqual(values["Checkpoint Hash"], "modelhash")
        self.assertEqual(values["LoRA 1 Name"], "My LoRA")
        self.assertEqual(values["LoRA 1 Strength"], "0.65")
        self.assertIn("InvokeAI Workflow JSON", values)
        self.assertIn("InvokeAI Graph JSON", values)
        self.assertTrue(any(item["category"] == "Workflow Details" for item in result["copy_items"]))

    def test_comfyui_embedded_workflow_is_exposed_for_opening(self):
        workflow = {
            "last_node_id": 2,
            "last_link_id": 1,
            "nodes": [
                {"id": 1, "type": "CheckpointLoaderSimple", "pos": [20, 20], "widgets_values": ["model.safetensors"]},
                {"id": 2, "type": "KSampler", "pos": [300, 20], "widgets_values": [42, 28]},
            ],
            "links": [[1, 1, 0, 2, 0, "MODEL"]],
            "groups": [],
            "config": {},
            "extra": {},
            "version": 0.4,
        }
        result = self.inspect_png({"workflow": workflow})
        self.assertEqual(result["embedded_workflow"], workflow)
        values = self.item_values(result)
        self.assertIn("Workflow JSON", values)

    def test_comfyui_workflow_json_string_is_exposed_for_opening(self):
        workflow = {
            "nodes": [{"id": 7, "type": "LoadImage", "widgets_values": ["reference.png"]}],
            "links": [],
            "last_node_id": 7,
            "last_link_id": 0,
        }
        result = self.inspect_png({"workflow": json.dumps(workflow)})
        self.assertEqual(result["embedded_workflow"], workflow)

    def test_api_prompt_is_not_misidentified_as_canvas_workflow(self):
        api_prompt = {
            "3": {"class_type": "KSampler", "inputs": {"seed": 123}},
            "4": {"class_type": "SaveImage", "inputs": {"filename_prefix": "test"}},
        }
        result = self.inspect_png({"prompt": api_prompt})
        self.assertIsNone(result["embedded_workflow"])

    def test_generation_data_exposes_kiri_active_nodes_for_explicit_import(self):
        result = self.inspect_png({"kiri_metadata": {
            "source_workflow_id": "workflow-source-uuid",
            "graph": {"active_upstream_nodes": [
                {"id": "48", "class_type": "KSamplerAdvanced", "title": "Refinement Pass", "inputs": {
                    "noise_seed": 12345, "steps": 24, "cfg": 5.5,
                    "positive": ["20", 0], "negative": ["21", 0],
                }},
                {"id": "20", "class_type": "CLIPTextEncode", "inputs": {"text": "positive prompt"}},
            ]},
        }})
        generation_data = result["generation_data"]
        self.assertEqual(generation_data["source"], "Kiri structured graph")
        self.assertEqual(generation_data["workflow_id"], "workflow-source-uuid")
        self.assertEqual(generation_data["nodes"][0]["id"], "48")
        self.assertEqual(generation_data["nodes"][0]["title"], "Refinement Pass")
        self.assertEqual(generation_data["nodes"][0]["inputs"]["noise_seed"], 12345)
        self.assertEqual(generation_data["nodes"][0]["inputs"]["positive"], ["20", 0])

    def test_generation_data_falls_back_to_executed_comfy_prompt_graph(self):
        api_prompt = {
            "30": {"class_type": "KSamplerAdvanced", "inputs": {"noise_seed": 7, "steps": 30}},
            "31": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello"}},
        }
        workflow = {
            "id": "canvas-workflow-uuid",
            "nodes": [{"id": 30, "type": "KSamplerAdvanced", "title": "Second Pass"}],
            "links": [], "last_node_id": 31, "last_link_id": 0,
        }
        result = self.inspect_png({"prompt": api_prompt, "workflow": workflow})
        generation_data = result["generation_data"]
        self.assertEqual(generation_data["source"], "ComfyUI executed prompt")
        self.assertEqual(generation_data["workflow_id"], "canvas-workflow-uuid")
        self.assertEqual(generation_data["nodes"][0]["title"], "Second Pass")
        self.assertEqual(generation_data["nodes"][0]["inputs"]["steps"], 30)

    def test_invokeai_workflow_is_not_misidentified_as_comfyui_workflow(self):
        invoke_workflow = {"nodes": [{"id": "denoise", "type": "denoise"}], "edges": []}
        result = self.inspect_png({"invokeai_workflow": invoke_workflow})
        self.assertIsNone(result["embedded_workflow"])

    def test_kiri_graph_metadata_is_expanded_into_individual_copy_actions(self):
        result = self.inspect_png({"kiri_metadata": {
            "collector": "Kiri Image Loader",
            "graph": {
                "samplers": [{"node_id": "4", "settings": {"seed": 123, "steps": 28, "cfg": 6.5}}],
                "text_encoders_and_prompts": [
                    {"branch": "positive", "class_type": "CLIPTextEncode", "values": {"text": "positive from graph"}},
                    {"branch": "negative", "class_type": "CLIPTextEncode", "values": {"text": "negative from graph"}},
                ],
                "model_resources": [
                    {"type": "Checkpoint / Model", "name": "base.safetensors", "sha256": "abc123"},
                    {"type": "LoRA", "name": "style.safetensors", "strength_model": 0.75, "strength_clip": 0.5},
                ],
                "active_upstream_nodes": [
                    {"id": "1", "class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 768}},
                ],
            },
            "custom_metadata": {"source": "manual editor", "value": "Artist: Keryula"},
        }})
        values = self.item_values(result)
        self.assertEqual(values["Positive Prompt"], "positive from graph")
        self.assertEqual(values["Negative Prompt"], "negative from graph")
        self.assertEqual(values["Sampler 1 · steps"], "28")
        self.assertEqual(values["Checkpoint / Model · name"], "base.safetensors")
        self.assertEqual(values["Checkpoint / Model SHA256 · base.safetensors"], "abc123")
        self.assertEqual(values["LoRA · style.safetensors · strength_model"], "0.75")
        self.assertEqual(values["Custom Metadata"], "Artist: Keryula")
        self.assertIn("Kiri Metadata JSON", values)
        self.assertIn("Active Upstream Nodes JSON", values)

    def test_a1111_parameters(self):
        result = self.inspect_png({"parameters": "A detailed portrait\nNegative prompt: blurry\nSteps: 30, Sampler: Euler a, CFG scale: 7, Seed: 42, Size: 768x1024"})
        values = self.item_values(result)
        self.assertEqual(values["Positive Prompt"], "A detailed portrait")
        self.assertEqual(values["Negative Prompt"], "blurry")
        self.assertEqual(values["Steps"], "30")
        self.assertEqual(values["Seed"], "42")

    def test_settings_only_a1111_parameters_do_not_become_positive_prompt(self):
        parameters = (
            "\nNegative prompt:"
            "\nSteps: 32, Sampler: dpmpp_2m, Scheduler: karras, CFG scale: 6.5, Seed: 987654321, "
            "Model: base.safetensors, Model hash: " + "a" * 64 + ", "
            "LoRA 1: style.safetensors [model=0.75; clip=0.5], LoRA 1 hash: " + "b" * 64 + ", "
            "Size: 1024x768"
        )
        result = self.inspect_png({"parameters": parameters})
        values = self.item_values(result)
        self.assertNotIn("Positive Prompt", values)
        self.assertNotIn("Negative Prompt", values)
        self.assertEqual(values["Steps"], "32")
        self.assertEqual(values["Sampler"], "dpmpp_2m")
        self.assertEqual(values["Scheduler"], "karras")
        self.assertEqual(values["CFG"], "6.5")
        self.assertEqual(values["Seed"], "987654321")
        self.assertEqual(values["Checkpoint"], "base.safetensors")
        self.assertEqual(values["Checkpoint Hash"], "a" * 64)
        self.assertEqual(values["LoRA 1 Name"], "style.safetensors")
        self.assertEqual(values["LoRA 1 Strength"], "0.75")
        self.assertEqual(values["LoRA 1 CLIP Strength"], "0.5")
        self.assertEqual(values["LoRA 1 Hash"], "b" * 64)
        self.assertEqual(values["Width"], "1024")
        self.assertEqual(values["Height"], "768")

    def test_easy_diffusion_json_metadata(self):
        result = self.inspect_png({"parameters": {
            "prompt": "easy diffusion positive", "negative_prompt": "easy diffusion negative",
            "seed": 77, "num_inference_steps": 24, "guidance_scale": 6.25,
            "sampler_name": "DPM++ 2M Karras", "model_id": "sdxl-checkpoint",
            "width": 1024, "height": 768,
        }})
        values = self.item_values(result)
        self.assertEqual(values["Positive Prompt"], "easy diffusion positive")
        self.assertEqual(values["Negative Prompt"], "easy diffusion negative")
        self.assertEqual(values["Steps"], "24")
        self.assertEqual(values["Guidance Scale"], "6.25")
        self.assertEqual(values["Sampler"], "DPM++ 2M Karras")
        self.assertEqual(values["Model ID"], "sdxl-checkpoint")

    def test_fooocus_mre_extended_parameters(self):
        result = self.inspect_png({"parameters": {
            "prompt": "fooocus positive", "negative_prompt": "fooocus negative",
            "seed": 88, "steps": 30, "cfg_scale": 4.5,
            "fooocus_v2_expansion": "expanded prompt", "fooocus_styles": ["Fooocus V2", "Cinematic"],
            "fooocus_performance": "Quality", "fooocus_sharpness": 2,
            "adm_guidance": [1.5, 0.8, 0.3], "adaptive_cfg": 7,
        }})
        values = self.item_values(result)
        self.assertEqual(values["Positive Prompt"], "fooocus positive")
        self.assertEqual(values["Fooocus V2 Expansion"], "expanded prompt")
        self.assertEqual(values["Fooocus Performance"], "Quality")
        self.assertIn("Fooocus Metadata JSON", values)

    def test_draw_things_xmp_fields(self):
        xmp = '''<?xpacket begin=""?><x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:k="https://kiri.test/ns/">
          <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
            <rdf:Description k:prompt="draw things prompt" k:negativePrompt="draw things negative"
              k:steps="21" k:cfgScale="5.2" k:sampler="DPM++ 2M" k:seed="9901" />
          </rdf:RDF></x:xmpmeta><?xpacket end="w"?>'''
        result = self.inspect_png({"XML:com.adobe.xmp": xmp})
        values = self.item_values(result)
        self.assertEqual(values["Positive Prompt"], "draw things prompt")
        self.assertEqual(values["Negative Prompt"], "draw things negative")
        self.assertEqual(values["Steps"], "21")
        self.assertEqual(values["CFG"], "5.2")
        self.assertIn("XMP Metadata XML", values)

    def test_fooocus_and_naifu_handoff_chunks_are_readable(self):
        fooocus = {
            "prompt": "fooocus handoff prompt", "negative_prompt": "fooocus handoff negative",
            "resolution": "1024x768", "seed": 321, "steps": 28,
        }
        naifu = {
            "prompt": "naifu handoff prompt", "negative_prompt": "naifu handoff negative",
            "width": 768, "height": 1024, "seed": 654, "steps": 24, "scale": 6.0,
        }
        result = self.inspect_png({
            "fooocus_handoff": fooocus,
            "naifu_handoff": naifu,
        })
        values = self.item_values(result)
        self.assertEqual(values["Positive Prompt"], "naifu handoff prompt")
        self.assertEqual(values["Negative Prompt"], "naifu handoff negative")
        self.assertIn("Fooocus Metadata JSON", values)
        self.assertIn("Naifu Metadata JSON", values)
        # One primary source label is reported per image; both provider records
        # must still be retained independently in the viewer copy list.
        self.assertIn("Fooocus / Fooocus-MRE", result["source"])
        self.assertTrue(any(item["label"].startswith("Naifu ·") for item in result["copy_items"]))

    def test_novelai_stealth_alpha_metadata(self):
        metadata = {
            "Description": "NovelAI stealth prompt",
            "Comment": json.dumps({
                "prompt": "NovelAI positive", "uc": "NovelAI negative", "seed": 42,
                "steps": 28, "scale": 5.5, "sampler": "k_euler_ancestral",
                "sm": True, "sm_dyn": False, "cfg_rescale": 0.2,
            }),
        }
        compressed = gzip.compress(json.dumps(metadata, separators=(",", ":")).encode("utf-8"))
        encoded = b"stealth_pngcomp" + (len(compressed) * 8).to_bytes(4, "big") + compressed
        bits = np.unpackbits(np.frombuffer(encoded, dtype=np.uint8))
        width = height = 64
        alpha_stream = np.full(width * height, 254, dtype=np.uint8)
        alpha_stream[:len(bits)] |= bits
        alpha = alpha_stream.reshape((width, height)).T
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[..., :3] = 128
        rgba[..., 3] = alpha
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "novelai-stealth.png"
            Image.fromarray(rgba, "RGBA").save(path)
            result = _metadata.inspect_image(str(path))
        values = self.item_values(result)
        self.assertEqual(result["source"], "NovelAI")
        self.assertEqual(values["Positive Prompt"], "NovelAI positive")
        self.assertEqual(values["Negative Prompt"], "NovelAI negative")
        self.assertEqual(values["Seed"], "42")
        self.assertEqual(values["SMEA"], "True")


    def test_novelai_legacy_metadata(self):
        result = self.inspect_png({
            "Software": "NovelAI",
            "Description": "legacy NovelAI description prompt",
            "Comment": {"prompt": "legacy positive", "uc": "legacy negative", "seed": 123, "steps": 28, "scale": 5.0},
        })
        values = self.item_values(result)
        self.assertEqual(result["source"], "NovelAI")
        self.assertEqual(values["Positive Prompt"], "legacy positive")
        self.assertEqual(values["Negative Prompt"], "legacy negative")
        self.assertEqual(values["Seed"], "123")

    def test_naifu_json_metadata(self):
        result = self.inspect_png({
            "Software": "Naifu",
            "Comment": {"prompt": "naifu positive", "uc": "naifu negative", "seed": 998, "steps": 20, "scale": 7.0},
        })
        values = self.item_values(result)
        self.assertEqual(result["source"], "Naifu")
        self.assertEqual(values["Positive Prompt"], "naifu positive")
        self.assertEqual(values["Negative Prompt"], "naifu negative")
        self.assertEqual(values["CFG"], "7.0")


    def test_draw_things_xmp_usercomment_json_round_trip(self):
        xmp = '''<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:exif="http://ns.adobe.com/exif/1.0/">
          <rdf:RDF><rdf:Description rdf:about="">
            <dc:description><rdf:Alt><rdf:li xml:lang="x-default">Draw Things positive
Negative prompt: Draw Things negative
Steps: 28, Sampler: Euler, Guidance Scale: 6.5, Seed: 998, Size: 1024x768</rdf:li></rdf:Alt></dc:description>
            <exif:UserComment><rdf:Alt><rdf:li xml:lang="x-default">{"c":"Draw Things positive","uc":"Draw Things negative","model":"test-model","sampler":"Euler","scale":6.5,"seed":998,"size":"1024x768","steps":28,"lora":[{"model":"style-lora","weight":0.75}]}</rdf:li></rdf:Alt></exif:UserComment>
          </rdf:Description></rdf:RDF></x:xmpmeta>'''
        result = self.inspect_png({"XML:com.adobe.xmp": xmp})
        values = self.item_values(result)
        self.assertEqual(values["Positive Prompt"], "Draw Things positive")
        self.assertEqual(values["Negative Prompt"], "Draw Things negative")
        self.assertEqual(values["Seed"], "998")
        self.assertEqual(values["LoRA 1 Name"], "style-lora")
        self.assertEqual(values["LoRA 1 Strength"], "0.75")
        self.assertIn("Draw Things UserComment JSON", values)

    def test_easy_diffusion_drag_drop_task_envelope_is_normalized(self):
        result = self.inspect_png({"easy_diffusion_task": {
            "numOutputsTotal": 1,
            "reqBody": {
                "prompt": "easy positive", "negative_prompt": "easy negative",
                "width": 1024, "height": 768, "seed": 71,
                "num_inference_steps": 26, "guidance_scale": 6.0,
                "sampler_name": "k_euler", "scheduler_name": "karras",
                "use_stable_diffusion_model": "my-model.safetensors",
                "use_vae_model": "my-vae.safetensors",
            },
        }})
        values = self.item_values(result)
        self.assertEqual(values["Positive Prompt"], "easy positive")
        self.assertEqual(values["Negative Prompt"], "easy negative")
        self.assertEqual(values["Steps"], "26")
        self.assertEqual(values["Guidance Scale"], "6.0")
        self.assertEqual(values["Sampler"], "k_euler")
        self.assertEqual(values["Checkpoint"], "my-model.safetensors")
        self.assertEqual(values["VAE"], "my-vae.safetensors")

    def test_pixai_and_tensorart_handoff_chunks_are_readable_parameter_records(self):
        pixai = {
            "modelVersionId": "pixai-version-7", "prompt": "pixai positive",
            "negativePrompt": "pixai negative", "width": 1024, "height": 768,
            "seed": 42, "sampling": {"steps": 28, "cfgScale": 6.5, "sampler": "Euler"},
            "loras": [{"modelVersionId": "pixai-lora-3", "weight": 0.7}],
        }
        tensor = {"requestId": "kiri-test", "stages": [
            {"type": "INPUT_INITIALIZE", "inputInitialize": {"count": 1, "seed": 42}},
            {"type": "DIFFUSION", "diffusion": {
                "sdModel": "ta-model-3", "sdVae": "ta-vae-2",
                "width": 1024, "height": 768,
                "prompts": [{"text": "tensor positive", "weight": 1.0}],
                "negativePrompts": [{"text": "tensor negative", "weight": 1.0}],
                "steps": 28, "cfgScale": 6.5, "sampler": "Euler",
                "lora": {"items": [{"loraModel": "ta-lora-9", "weight": 0.7}]},
            }},
        ]}
        result = self.inspect_png({"pixai_api": pixai, "tensor_art_tams": tensor})
        items = result["copy_items"]
        rendered = [f"{item['label']}: {item['value']}" for item in items]
        self.assertTrue(any("pixai positive" in value for value in rendered))
        self.assertTrue(any("pixai negative" in value for value in rendered))
        self.assertTrue(any("tensor positive" in value for value in rendered))
        self.assertTrue(any("tensor negative" in value for value in rendered))
        self.assertTrue(any(item["label"] == "Steps" and item["value"] == "28" for item in items))
        self.assertTrue(any(item["label"] == "Model Version ID" and item["value"] == "pixai-version-7" for item in items))
        self.assertTrue(any(item["label"] == "Platform Model ID" and item["value"] == "ta-model-3" for item in items))
        self.assertTrue(any(item["label"] == "Platform VAE ID" and item["value"] == "ta-vae-2" for item in items))

    def test_civitai_parameter_resource_block_exposes_provider_ids(self):
        parameters = "positive prompt\nNegative prompt: negative prompt\nSteps: 30, Sampler: Euler, CFG scale: 7, Seed: 42, Size: 768x1024, Civitai resources: [{\"type\":\"lora\",\"weight\":0.8,\"modelVersionId\":12345},{\"type\":\"checkpoint\",\"modelVersionId\":98765}]"
        result = self.inspect_png({"parameters": parameters})
        values = self.item_values(result)
        self.assertEqual(values["LoRA 1 Model Version ID"], "12345")
        self.assertEqual(values["Checkpoint Model Version ID"], "98765")

    def test_fooocus_native_scheme_json_is_detected_and_parsed(self):
        result = self.inspect_png({
            "parameters": {"prompt": "foo positive", "negative_prompt": "foo negative", "resolution": "1024x768", "steps": 30, "guidance_scale": 6.0, "base_model": "sdxl", "lora_combined_1": "style : 0.7"},
            "fooocus_scheme": "fooocus",
        })
        values = self.item_values(result)
        self.assertEqual(values["Positive Prompt"], "foo positive")
        self.assertEqual(values["Negative Prompt"], "foo negative")
        self.assertEqual(values["Steps"], "30")
        self.assertEqual(values["Fooocus · lora_combined_1"], "style : 0.7")


    def test_draw_things_exif_usercomment_json_is_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "draw-things-exif.png"
            exif = Image.Exif()
            exif[0x9286] = json.dumps({
                "c": "EXIF positive", "uc": "EXIF negative", "model": "draw-model",
                "sampler": "Euler", "scale": 5.5, "seed": 606,
                "size": "768x1024", "steps": 24,
                "lora": [{"model": "draw-lora", "weight": 0.6}],
            })
            Image.new("RGB", (24, 32)).save(path, exif=exif)
            result = _metadata.inspect_image(str(path))
        values = self.item_values(result)
        self.assertEqual(values["Positive Prompt"], "EXIF positive")
        self.assertEqual(values["Negative Prompt"], "EXIF negative")
        self.assertEqual(values["Seed"], "606")
        self.assertEqual(values["LoRA 1 Name"], "draw-lora")


if __name__ == "__main__":
    unittest.main()
