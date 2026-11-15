import importlib.util
import json
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "metadata_adapters.py"
_spec = importlib.util.spec_from_file_location("kiri_metadata_adapters_under_test", MODULE_PATH)
adapters = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adapters)


class MetadataAdapterTests(unittest.TestCase):
    def setUp(self):
        self.record = {
            "created_utc": "2026-11-04T00:00:00+00:00",
            "graph": {
                "prompts": {"positive": "1girl, sitting", "negative": "blurry, low quality"},
                "samplers": [{"class_type": "KSampler", "settings": {
                    "seed": 987654321, "steps": 32, "cfg": 6.5,
                    "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0,
                }}],
                "model_resources": [
                    {"type": "Checkpoint / Model", "name": "base.safetensors", "sha256": "a" * 64},
                    {"type": "LoRA", "name": "style.safetensors", "strength_model": 0.75, "strength_clip": 0.5,
                     "sha256": "b" * 64, "civitai_provenance": {"model_id": 123, "version_id": 456}},
                ],
            },
        }

    def test_adapter_release_version_matches_package_version(self):
        self.assertEqual(adapters.VERSION, "0.5.19")

    def test_swarmui_native_parameters_envelope(self):
        prefs = dict(adapters.DEFAULT_EXPORT_SETTINGS, parameters_format="swarmui")
        chunks = adapters.build_export_chunks(self.record, 1024, 768, prefs)
        envelope = json.loads(chunks["parameters"])
        params = envelope["sui_image_params"]
        self.assertEqual(params["prompt"], "1girl, sitting")
        self.assertEqual(params["negativeprompt"], "blurry, low quality")
        self.assertEqual(params["seed"], "987654321")
        self.assertEqual(params["width"], "1024")
        self.assertEqual(params["height"], "768")
        self.assertEqual(params["used_loras"][0]["strength"], "0.75")
        self.assertNotIn("hash", envelope["sui_models"][1], "A full-file SHA-256 is not SwarmUI's tensor hash.")
        self.assertEqual(envelope["sui_extra_data"]["original_prompt"], "1girl, sitting")

    def test_invokeai_portable_record_has_no_fake_workflow_or_hash(self):
        data = adapters.build_invokeai_metadata(self.record, 1024, 768)
        self.assertEqual(data["positive_prompt"], "1girl, sitting")
        self.assertEqual(data["negative_prompt"], "blurry, low quality")
        self.assertEqual(data["seed"], 987654321)
        self.assertNotIn("model", data, "A filename is not an InvokeAI catalog ModelIdentifier.")
        self.assertEqual(data["kiri_model_name_hints"], ["base.safetensors"])
        self.assertNotIn("app_version", data, "Kiri must not impersonate the host InvokeAI version.")
        self.assertNotIn("invokeai_workflow", data)
        self.assertNotIn("invokeai_graph", data)

    def test_novelai_description_and_comment_schema(self):
        description, comment = adapters.build_novelai_comment(self.record)
        self.assertEqual(description, "1girl, sitting")
        self.assertEqual(comment["prompt"], "1girl, sitting")
        self.assertEqual(comment["uc"], "blurry, low quality")
        self.assertEqual(comment["seed"], 987654321)
        self.assertEqual(comment["steps"], 32)

    def test_interop_manifest_includes_civitai_and_handoff_targets_without_fabricated_pixai_model_id(self):
        data = adapters.build_interchange_manifest(self.record, 1024, 768)
        self.assertEqual(data["schema"], "kiri-universal-generation-metadata")
        self.assertEqual(data["civitai_resources"][0]["modelVersionId"], "456")
        self.assertNotIn("modelVersionId", data["platform_handoff"]["pixai_api"])
        self.assertIn("tensor_art", data["platform_handoff"])
        self.assertIn("seaart", data["platform_handoff"])
        self.assertIn("tensorhub_cozy_creator", data["platform_handoff"])
        self.assertEqual(data["platform_handoff"]["easy_diffusion"]["reqBody"]["width"], 1024)

    def test_default_exports_verified_native_records_and_one_bundle(self):
        chunks = adapters.build_export_chunks(self.record, 1024, 768)
        self.assertIn("invokeai_metadata", chunks)
        self.assertIn("Comment", chunks)
        self.assertIn("Description", chunks)
        self.assertIn("kiri_interop", chunks)
        self.assertNotIn("parameters", chunks, "The saver writes the A1111 parameters field from Kiri's canonical text.")
        self.assertNotIn("invokeai_workflow", chunks)
        self.assertNotIn("invokeai_graph", chunks)

    def test_export_settings_normalize_and_sidecar_is_opt_in(self):
        self.assertTrue(adapters.normalize_export_settings(None)["write_easy_diffusion_task"])
        self.assertFalse(adapters.normalize_export_settings(None)["write_sidecar"])
        defaults = adapters.normalize_export_settings(None)
        self.assertTrue(defaults["save_positive_prompt"])
        self.assertTrue(defaults["save_negative_prompt"])
        self.assertTrue(defaults["save_node_names"])
        legacy = adapters.normalize_export_settings('{"save_prompts":false}')
        self.assertFalse(legacy["save_positive_prompt"])
        self.assertFalse(legacy["save_negative_prompt"])
        mixed = adapters.normalize_export_settings('{"save_prompts":false,"save_positive_prompt":true}')
        self.assertTrue(mixed["save_positive_prompt"])
        self.assertFalse(mixed["save_negative_prompt"])
        self.assertTrue(defaults["platform_metadata"])
        settings = adapters.normalize_export_settings('{"parameters_format":"not-real","write_sidecar":true}')
        self.assertEqual(settings["parameters_format"], "a1111")
        self.assertTrue(settings["write_sidecar"])
        sidecar = adapters.build_sidecar(self.record, 1024, 768, settings)
        self.assertEqual(sidecar["schema_version"], 1)
        self.assertIn("adapters", sidecar)


    def test_fooocus_native_parameters_profile(self):
        prefs = dict(adapters.DEFAULT_EXPORT_SETTINGS, parameters_format="fooocus")
        chunks = adapters.build_export_chunks(self.record, 1024, 768, prefs)
        data = json.loads(chunks["parameters"])
        self.assertEqual(chunks["fooocus_scheme"], "fooocus")
        self.assertEqual(data["prompt"], "1girl, sitting")
        self.assertEqual(data["negative_prompt"], "blurry, low quality")
        self.assertEqual(data["resolution"], "1024x768")
        self.assertEqual(data["base_model"], "base")
        self.assertEqual(data["lora_combined_1"], "style : 0.75")

    def test_export_profiles_honor_independent_prompt_privacy_without_dropping_models(self):
        prefs = dict(adapters.DEFAULT_EXPORT_SETTINGS, parameters_format="fooocus", save_positive_prompt=False, save_negative_prompt=True)
        chunks = adapters.build_export_chunks(self.record, 1024, 768, prefs)
        fooocus = json.loads(chunks["parameters"])
        self.assertEqual(fooocus["prompt"], "")
        self.assertEqual(fooocus["negative_prompt"], "blurry, low quality")
        self.assertEqual(fooocus["steps"], 32)
        self.assertEqual(fooocus["base_model"], "base")
        self.assertEqual(fooocus["lora_combined_1"], "style : 0.75")

        reverse = dict(adapters.DEFAULT_EXPORT_SETTINGS, parameters_format="fooocus", save_positive_prompt=True, save_negative_prompt=False)
        reverse_fooocus = json.loads(adapters.build_export_chunks(self.record, 1024, 768, reverse)["parameters"])
        self.assertEqual(reverse_fooocus["prompt"], "1girl, sitting")
        self.assertEqual(reverse_fooocus["negative_prompt"], "")
        self.assertEqual(reverse_fooocus["base_model"], "base")
        self.assertEqual(reverse_fooocus["lora_combined_1"], "style : 0.75")

    def test_easy_diffusion_json_task_has_native_reqbody_shape(self):
        request = adapters.build_easy_diffusion_request(self.record, 1024, 768)
        self.assertEqual(request["numOutputsTotal"], 1)
        body = request["reqBody"]
        self.assertEqual(body["prompt"], "1girl, sitting")
        self.assertEqual(body["negative_prompt"], "blurry, low quality")
        self.assertEqual(body["width"], 1024)
        self.assertEqual(body["height"], 768)
        self.assertEqual(body["num_inference_steps"], 32)
        self.assertEqual(body["guidance_scale"], 6.5)
        self.assertEqual(body["use_stable_diffusion_model"], "base.safetensors")

    def test_draw_things_xmp_and_user_comment_are_both_generated(self):
        chunks = adapters.build_export_chunks(self.record, 1024, 768)
        xmp = chunks["XML:com.adobe.xmp"]
        self.assertIn("dc:description", xmp)
        self.assertIn("exif:UserComment", xmp)
        self.assertIn("1girl, sitting", xmp)
        user_comment = json.loads(chunks["drawthings_user_comment"])
        self.assertEqual(user_comment["c"], "1girl, sitting")
        self.assertEqual(user_comment["uc"], "blurry, low quality")
        self.assertEqual(user_comment["size"], "1024x768")

    def test_pixai_and_tensorart_use_only_explicit_provider_native_ids(self):
        record = json.loads(json.dumps(self.record))
        record["graph"]["model_resources"][0]["pixai_provenance"] = {"model_version_id": "pixai-v-44"}
        record["graph"]["model_resources"][0]["tensor_art_provenance"] = {"model_id": "tensor-model-12"}
        record["graph"]["model_resources"][1]["pixai_provenance"] = {"model_version_id": "pixai-lora-8"}
        record["graph"]["model_resources"][1]["tensor_art_provenance"] = {"model_id": "tensor-lora-9"}
        manifest = adapters.build_interchange_manifest(record, 1024, 768)
        pixai = manifest["platform_handoff"]["pixai_api"]
        self.assertEqual(pixai["modelVersionId"], "pixai-v-44")
        self.assertEqual(pixai["loras"][0]["modelVersionId"], "pixai-lora-8")
        tensor = manifest["platform_handoff"]["tensor_art"]
        diffusion = tensor["stages"][1]["diffusion"]
        self.assertEqual(diffusion["sdModel"], "tensor-model-12")
        self.assertEqual(diffusion["lora"]["items"][0]["loraModel"], "tensor-lora-9")

    def test_novelai_stealth_payload_round_trips_and_preserves_visible_pixels(self):
        from PIL import Image
        record = self.record
        original = Image.new("RGB", (512, 512), (110, 80, 150))
        embedded = adapters.embed_novelai_stealth(original, record)
        self.assertEqual(embedded.size, original.size)
        self.assertEqual(embedded.convert("RGB").getpixel((0, 0)), original.getpixel((0, 0)))
        # Validate the byte-level framing without depending on UI or a live NovelAI service.
        payload = adapters.build_novelai_stealth_payload(record)
        self.assertTrue(payload.startswith(b"stealth_pngcomp"))
        self.assertGreater(len(payload), 24)


    def test_universal_bundle_has_copyable_fooocus_and_naifu_handoff_chunks(self):
        chunks = adapters.build_export_chunks(self.record, 1024, 768)
        fooocus = json.loads(chunks["fooocus_handoff"])
        naifu = json.loads(chunks["naifu_handoff"])
        self.assertEqual(fooocus["prompt"], "1girl, sitting")
        self.assertEqual(fooocus["resolution"], "1024x768")
        self.assertEqual(naifu["prompt"], "1girl, sitting")
        self.assertEqual(naifu["negative_prompt"], "blurry, low quality")
        self.assertEqual(naifu["width"], 1024)
        self.assertIn("native import schema is not verified", naifu["_compatibility"])

    def test_swarmui_uses_only_explicit_tensor_hash(self):
        record = json.loads(json.dumps(self.record))
        record["graph"]["model_resources"][1]["swarmui_tensor_hash"] = "c" * 64
        envelope = adapters.build_swarmui_payload(record, 1024, 768)
        self.assertEqual(envelope["sui_models"][1]["hash"], "0x" + "c" * 64)


if __name__ == "__main__":
    unittest.main()
