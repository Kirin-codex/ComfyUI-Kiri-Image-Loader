"""Generator metadata writers used by Kiri Image Loader.

The Kiri record remains canonical. Adapters emit only fields whose external
format is sufficiently documented. Generic site handoff profiles are stored
inside one `kiri_interop` manifest instead of pretending to be native import
formats for platforms that do not document a PNG metadata contract.
"""
from __future__ import annotations

import gzip
import json
import uuid
from datetime import datetime, timezone
from typing import Any

VERSION = "0.5.19"
DEFAULT_EXPORT_SETTINGS = {
    "parameters_format": "a1111",  # a1111, swarmui, fooocus, or easy_diffusion; one owns PNG `parameters`
    # These three metadata controls are intentionally separate from the
    # complete ComfyUI workflow switch. Platform exports contain parameter
    # values only; they never contain ComfyUI node objects or graph links.
    "save_positive_prompt": True,
    "save_negative_prompt": True,
    "save_node_names": True,
    "platform_metadata": True,
    "invokeai": True,
    "novelai": True,
    "universal_bundle": True,
    "write_sidecar": False,
    "write_easy_diffusion_task": True,
    # Which dual-image branch (or both) is written when Save Path and File
    # Name are populated. Defaults to A to preserve the historical saver.
    "save_images_from": "a",
}


def normalize_export_settings(value: Any = None) -> dict:
    settings = dict(DEFAULT_EXPORT_SETTINGS)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            value = None
    if isinstance(value, dict):
        fmt = str(value.get("parameters_format", settings["parameters_format"])).lower()
        settings["parameters_format"] = fmt if fmt in {"a1111", "swarmui", "fooocus", "easy_diffusion"} else "a1111"
        # Backward compatibility: v0.5.11 used one combined prompt switch.
        # Explicit v0.5.12 positive/negative values always take precedence.
        legacy_prompts = value.get("save_prompts") if "save_prompts" in value else None
        for key in ("save_positive_prompt", "save_negative_prompt"):
            if key in value:
                settings[key] = bool(value[key])
            elif legacy_prompts is not None:
                settings[key] = bool(legacy_prompts)
        for key in (
            "save_node_names", "platform_metadata", "invokeai", "novelai",
            "universal_bundle", "write_sidecar", "write_easy_diffusion_task",
        ):
            if key in value:
                settings[key] = bool(value[key])
        save_images_from = str(value.get("save_images_from", settings["save_images_from"])).strip().lower()
        settings["save_images_from"] = save_images_from if save_images_from in {"a", "b", "both"} else "a"
    return settings


def _graph(record: dict) -> dict:
    value = record.get("graph") if isinstance(record, dict) else None
    return value if isinstance(value, dict) else {}


def _prompts(record: dict) -> tuple[str, str]:
    prompts = _graph(record).get("prompts", {})
    if not isinstance(prompts, dict):
        prompts = {}
    return str(prompts.get("positive") or ""), str(prompts.get("negative") or "")


def _samplers(record: dict) -> list[dict]:
    items = _graph(record).get("samplers", [])
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _first_settings(record: dict) -> dict:
    for sampler in _samplers(record):
        values = sampler.get("settings")
        if isinstance(values, dict) and values:
            return values
    return {}


def _first(values: dict, *names: str, default=None):
    for name in names:
        if name in values and values[name] is not None and str(values[name]).strip() != "":
            return values[name]
    return default


def _resources(record: dict) -> list[dict]:
    values = _graph(record).get("model_resources", [])
    return [item for item in values if isinstance(item, dict)] if isinstance(values, list) else []


def _civitai_provenance(resource: dict) -> dict:
    value = resource.get("civitai_provenance")
    return value if isinstance(value, dict) else {}


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_json_clean(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _apply_prompt_export_preferences(record: dict, settings: dict) -> dict:
    """Return a safe copy with only explicitly disabled prompt branches removed."""
    safe = _json_clean(record if isinstance(record, dict) else {})
    graph = safe.get("graph") if isinstance(safe, dict) else None
    if not isinstance(graph, dict):
        return safe
    save_positive = bool(settings.get("save_positive_prompt", True))
    save_negative = bool(settings.get("save_negative_prompt", True))
    prompts = graph.get("prompts")
    if isinstance(prompts, dict):
        if not save_positive:
            prompts.pop("positive", None)
        if not save_negative:
            prompts.pop("negative", None)
    text_items = graph.get("text_encoders_and_prompts")
    if isinstance(text_items, list):
        kept = []
        for item in text_items:
            if not isinstance(item, dict):
                kept.append(item)
                continue
            branch = item.get("branch")
            if branch == "positive" and not save_positive:
                continue
            if branch == "negative" and not save_negative:
                continue
            if branch == "shared" and not (save_positive or save_negative):
                continue
            kept.append(item)
        graph["text_encoders_and_prompts"] = kept
    return safe


def _resource_model_name(resources: list[dict], *types: str) -> str | None:
    wanted = {value.lower() for value in types}
    for resource in resources:
        if str(resource.get("type", "")).lower() in wanted and resource.get("name"):
            return str(resource["name"])
    return None


def _sha256(resource: dict) -> str | None:
    value = str(resource.get("sha256") or "").strip().lower()
    if len(value) == 64 and all(ch in "0123456789abcdef" for ch in value):
        return value
    return None


def _provider_id(resource: dict, provider: str, *keys: str) -> str | None:
    """Return only an explicitly recorded ID owned by the named provider.

    Civitai IDs, filenames and hashes are deliberately never converted into
    PixAI/Tensor.Art IDs: those identifiers belong to different catalogues.
    """
    provider_data = resource.get(f"{provider}_provenance")
    if not isinstance(provider_data, dict):
        provider_data = resource.get("provider_ids") if isinstance(resource.get("provider_ids"), dict) else {}
    nested = provider_data.get(provider) if isinstance(provider_data.get(provider), dict) else provider_data
    for key in keys:
        value = nested.get(key) if isinstance(nested, dict) else None
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _first_resource(resources: list[dict], *types: str) -> dict | None:
    wanted = {str(value).lower() for value in types}
    return next((item for item in resources if str(item.get("type", "")).lower() in wanted and item.get("name")), None)


def _filename_stem(value: Any) -> str:
    text = str(value or "").replace("\\", "/").rsplit("/", 1)[-1]
    return text.rsplit(".", 1)[0] if "." in text else text


def build_fooocus_payload(record: dict, width: int, height: int) -> dict:
    """Build Fooocus' native JSON parameter scheme consumed by Load Parameters."""
    positive, negative = _prompts(record)
    settings = _first_settings(record)
    resources = _resources(record)
    checkpoint = _first_resource(resources, "Checkpoint / Model")
    vae = _first_resource(resources, "VAE")
    refiner = _first_resource(resources, "Refiner Model", "Refiner")
    result: dict[str, Any] = {
        "prompt": positive,
        "negative_prompt": negative,
        "full_prompt": positive,
        "full_negative_prompt": negative,
        "steps": _first(settings, "steps"),
        "resolution": f"{int(width)}x{int(height)}",
    }
    mapping = {
        "seed": ("seed", "noise_seed"),
        "guidance_scale": ("cfg", "cfg_scale", "guidance_scale"),
        "sampler": ("sampler_name", "sampler"),
        "scheduler": ("scheduler",),
        "sharpness": ("sharpness",),
        "performance": ("performance",),
        "styles": ("styles", "fooocus_styles"),
        "adaptive_cfg": ("adaptive_cfg",),
        "clip_skip": ("clip_skip",),
        "overwrite_switch": ("overwrite_switch",),
        "refiner_switch": ("refiner_switch", "refinerswitchat"),
        "refiner_swap_method": ("refiner_swap_method",),
        "adm_guidance": ("adm_guidance", "adm_guidance_values"),
        "freeu": ("freeu",),
        "inpaint_engine_version": ("inpaint_engine_version",),
        "inpaint_method": ("inpaint_method",),
    }
    for output_key, candidates in mapping.items():
        value = _first(settings, *candidates)
        if value is not None:
            if isinstance(value, tuple):
                value = list(value)
            result[output_key] = value
    if checkpoint:
        result["base_model"] = _filename_stem(checkpoint.get("name"))
        if _sha256(checkpoint):
            result["base_model_hash"] = _sha256(checkpoint)
    if vae:
        result["vae"] = _filename_stem(vae.get("name"))
    if refiner:
        result["refiner_model"] = _filename_stem(refiner.get("name"))
        if _sha256(refiner):
            result["refiner_model_hash"] = _sha256(refiner)
    lora_resources = [item for item in resources if item.get("type") == "LoRA" and item.get("name")]
    for index, item in enumerate(lora_resources, 1):
        weight = item.get("strength_model", 1.0)
        result[f"lora_combined_{index}"] = f"{_filename_stem(item['name'])} : {weight}"
    if lora_resources:
        # Keep a second explicit array for third-party tooling; Fooocus itself
        # consumes lora_combined_N above.
        result["loras"] = [[_filename_stem(x["name"]), x.get("strength_model", 1.0)] for x in lora_resources]
    return _json_clean({k: v for k, v in result.items() if v is not None})


def build_draw_things_metadata(record: dict, width: int, height: int) -> dict:
    """Build the compact Draw Things EXIF UserComment JSON structure."""
    positive, negative = _prompts(record)
    settings = _first_settings(record)
    resources = _resources(record)
    checkpoint = _first_resource(resources, "Checkpoint / Model")
    result: dict[str, Any] = {
        "c": positive, "uc": negative, "size": f"{int(width)}x{int(height)}",
    }
    mapping = {
        "sampler": ("sampler_name", "sampler"),
        "scale": ("cfg", "cfg_scale", "guidance_scale"),
        "seed": ("seed", "noise_seed"),
        "steps": ("steps",),
        "strength": ("denoise", "strength"),
        "shift": ("shift",),
        "seed_mode": ("seed_mode",),
    }
    for output_key, candidates in mapping.items():
        value = _first(settings, *candidates)
        if value is not None:
            result[output_key] = value
    if checkpoint:
        result["model"] = _filename_stem(checkpoint.get("name"))
    loras = []
    for item in resources:
        if item.get("type") == "LoRA" and item.get("name"):
            loras.append({"model": _filename_stem(item["name"]), "weight": item.get("strength_model", 1.0)})
    if loras:
        result["lora"] = loras
    result["generated_by"] = f"Kiri Image Loader {VERSION}"
    return _json_clean(result)


def _xml_escape(value: Any) -> str:
    return (str(value).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def build_draw_things_xmp(record: dict, width: int, height: int) -> str:
    """Build the Draw Things-compatible XMP payload used in PNG iTXt."""
    metadata = build_draw_things_metadata(record, width, height)
    positive, negative = _prompts(record)
    settings = _first_settings(record)
    lines = []
    if positive:
        lines.append(positive)
    if negative:
        lines.append(f"Negative prompt: {negative}")
    parameters = []
    for label, candidates in (
        ("Steps", ("steps",)), ("Sampler", ("sampler_name", "sampler")),
        ("Guidance Scale", ("cfg", "cfg_scale", "guidance_scale")),
        ("Seed", ("seed", "noise_seed")),
    ):
        value = _first(settings, *candidates)
        if value is not None:
            parameters.append(f"{label}: {value}")
    parameters.append(f"Size: {int(width)}x{int(height)}")
    if metadata.get("model"):
        parameters.append(f"Model: {metadata['model']}")
    if metadata.get("strength") is not None:
        parameters.append(f"Strength: {metadata['strength']}")
    for index, lora in enumerate(metadata.get("lora", []), 1):
        parameters.append(f"LoRA {index} Model: {lora['model']}, LoRA {index} Weight: {lora.get('weight', 1.0)}")
    if parameters:
        lines.append(", ".join(parameters))
    description = _xml_escape("\n".join(lines))
    user_comment = _xml_escape(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")))
    producer = _xml_escape(f"Kiri Image Loader {VERSION}")
    return "\n".join([
        '<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 6.0.0">',
        '  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">',
        '    <rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:exif="http://ns.adobe.com/exif/1.0/">',
        '      <dc:description><rdf:Alt><rdf:li xml:lang="x-default">' + description + '</rdf:li></rdf:Alt></dc:description>',
        '      <xmp:CreatorTool>' + producer + '</xmp:CreatorTool>',
        '      <exif:UserComment><rdf:Alt><rdf:li xml:lang="x-default">' + user_comment + '</rdf:li></rdf:Alt></exif:UserComment>',
        '    </rdf:Description>',
        '  </rdf:RDF>',
        '</x:xmpmeta>',
    ])


def build_tensorart_request(record: dict, width: int, height: int) -> dict:
    """Build a documented TAMS v1 DIFFUSION stage (model IDs stay provider-native)."""
    positive, negative = _prompts(record)
    settings = _first_settings(record)
    resources = _resources(record)
    diffusion: dict[str, Any] = {
        "width": int(width), "height": int(height),
        "prompts": ([{"text": positive, "weight": 1.0}] if positive else []),
        "negativePrompts": ([{"text": negative, "weight": 1.0}] if negative else []),
    }
    for output_key, candidates in {
        "sampler": ("sampler_name", "sampler"),
        "steps": ("steps",), "cfgScale": ("cfg", "cfg_scale"),
        "clipSkip": ("clip_skip",), "denoisingStrength": ("denoise",),
        "scheduleName": ("scheduler",),
    }.items():
        value = _first(settings, *candidates)
        if value is not None:
            diffusion[output_key] = value
    checkpoint = _first_resource(resources, "Checkpoint / Model")
    if checkpoint:
        model_id = _provider_id(checkpoint, "tensor_art", "model_id", "modelId", "sdModel")
        if model_id:
            diffusion["sdModel"] = model_id
        # If no provider-native model ID exists, leave sdModel unset and
        # explain the required selection in the handoff notes below.
    vae = _first_resource(resources, "VAE")
    if vae:
        vae_id = _provider_id(vae, "tensor_art", "model_id", "modelId", "sdVae")
        if vae_id:
            diffusion["sdVae"] = vae_id
    lora_items = []
    for item in resources:
        if item.get("type") != "LoRA" or not item.get("name"):
            continue
        model_id = _provider_id(item, "tensor_art", "model_id", "modelId", "loraModel")
        if model_id:
            lora_items.append({"loraModel": model_id, "weight": item.get("strength_model", 1.0)})
    if lora_items:
        diffusion["lora"] = {"items": lora_items}
    diffusion = _json_clean(diffusion)
    # Request id is unique for each generated image. TAMS requires an
    # INPUT_INITIALIZE stage before DIFFUSION. Seed is copied if recorded.
    seed = _first(settings, "seed", "noise_seed")
    initialize = {"count": 1}
    if seed is not None:
        try:
            seed_value = int(seed)
            if 0 <= seed_value <= 4294967295:
                initialize["seed"] = seed_value
        except (TypeError, ValueError):
            pass
    stages = [{"type": "INPUT_INITIALIZE", "inputInitialize": initialize}, {"type": "DIFFUSION", "diffusion": diffusion}]
    return {"requestId": f"kiri-{uuid.uuid4().hex}", "stages": stages}


def build_swarmui_payload(record: dict, width: int, height: int) -> dict:
    """Build SwarmUI's documented JSON envelope for PNG `parameters`."""
    positive, negative = _prompts(record)
    settings = _first_settings(record)
    resources = _resources(record)
    image_params = {
        "prompt": positive,
        "negativeprompt": negative,
        "width": str(int(width)),
        "height": str(int(height)),
    }
    aliases = {
        "seed": ("seed", "noise_seed"),
        "steps": ("steps",),
        "cfgscale": ("cfg", "cfg_scale"),
        "sampler": ("sampler_name", "sampler"),
        "scheduler": ("scheduler",),
        "denoise": ("denoise",),
    }
    for output_key, candidates in aliases.items():
        value = _first(settings, *candidates)
        if value is not None:
            image_params[output_key] = str(value)
    model = _resource_model_name(resources, "Checkpoint / Model")
    if model:
        image_params["model"] = model
    loras = []
    models = []
    for resource in resources:
        kind = str(resource.get("type", ""))
        name = str(resource.get("name") or "")
        if not name:
            continue
        if kind == "LoRA":
            item = {"name": name}
            if resource.get("strength_model") is not None:
                item["strength"] = str(resource["strength_model"])
            if resource.get("strength_clip") is not None:
                item["clip_strength"] = str(resource["strength_clip"])
            loras.append(item)
            # SwarmUI's hash is a tensor-only hash, not a normal whole-file
            # SHA-256. Only forward a value explicitly recorded in that format.
            tensor_hash = str(resource.get("swarmui_tensor_hash") or resource.get("tensor_hash") or "").strip().lower()
            valid_tensor_hash = tensor_hash.removeprefix("0x")
            item_model = {"name": name, "param": "used_loras"}
            if len(valid_tensor_hash) == 64 and all(ch in "0123456789abcdef" for ch in valid_tensor_hash):
                item_model["hash"] = "0x" + valid_tensor_hash
            models.append(item_model)
        elif kind in {"Checkpoint / Model", "VAE", "Text Encoder"}:
            tensor_hash = str(resource.get("swarmui_tensor_hash") or resource.get("tensor_hash") or "").strip().lower()
            valid_tensor_hash = tensor_hash.removeprefix("0x")
            item_model = {"name": name, "param": {"Checkpoint / Model": "model", "VAE": "vae", "Text Encoder": "clip"}[kind]}
            if len(valid_tensor_hash) == 64 and all(ch in "0123456789abcdef" for ch in valid_tensor_hash):
                item_model["hash"] = "0x" + valid_tensor_hash
            models.append(item_model)
    if loras:
        image_params["used_loras"] = loras
    envelope = {"sui_image_params": image_params}
    if models:
        envelope["sui_models"] = models
    envelope["sui_extra_data"] = {
        "created_by": "Kiri Image Loader",
        "kiri_adapter_version": VERSION,
        "created_utc": record.get("created_utc"),
        "date": record.get("created_utc"),
        "original_prompt": positive,
        "original_negativeprompt": negative,
    }
    return _json_clean(envelope)


def build_invokeai_metadata(record: dict, width: int, height: int) -> dict:
    """Build a valid InvokeAI core-metadata record without inventing catalog IDs.

    InvokeAI's ModelIdentifier is installation-catalog based and contains a
    BLAKE3 content hash. A Kiri SHA-256 or a filename is not a valid substitute.
    Model records are included only when a source explicitly supplies a complete
    InvokeAI ModelIdentifier; portable name hints remain informational.
    """
    positive, negative = _prompts(record)
    settings = _first_settings(record)
    resources = _resources(record)
    result = {
        "metadata_version": "1.0.0",
        # app_version is reserved by InvokeAI for an InvokeAI release number.
        # Record Kiri as producer metadata instead of impersonating the host app.
        "kiri_producer": f"Kiri Image Loader {VERSION}",
        "positive_prompt": positive,
        "negative_prompt": negative,
        "width": int(width),
        "height": int(height),
    }
    mapping = {
        "seed": ("seed", "noise_seed"),
        "steps": ("steps",),
        "cfg_scale": ("cfg", "cfg_scale", "guidance_scale"),
        "scheduler": ("scheduler", "scheduler_name"),
        "strength": ("denoise", "strength"),
    }
    for key, candidates in mapping.items():
        value = _first(settings, *candidates)
        if value is not None:
            result[key] = value

    def exact_identifier(resource: dict) -> dict | None:
        candidate = resource.get("invokeai_model_identifier")
        if not isinstance(candidate, dict):
            return None
        required = ("key", "hash", "name", "base", "type")
        if not all(candidate.get(field) not in (None, "") for field in required):
            return None
        # Avoid leaking arbitrary sidecar fields into the typed identifier.
        return {field: candidate[field] for field in required}

    checkpoint = _first_resource(resources, "Checkpoint / Model")
    if checkpoint:
        identifier = exact_identifier(checkpoint)
        if identifier:
            result["model"] = identifier
        else:
            result["kiri_model_name_hints"] = [str(checkpoint["name"])]
    vae = _first_resource(resources, "VAE")
    if vae:
        identifier = exact_identifier(vae)
        if identifier:
            result["vae"] = identifier
        else:
            result.setdefault("kiri_model_name_hints", []).append(str(vae["name"]))
    loras = []
    for item in resources:
        if item.get("type") != "LoRA" or not item.get("name"):
            continue
        identifier = exact_identifier(item)
        if identifier:
            loras.append({"model": identifier, "weight": item.get("strength_model", 1.0)})
        else:
            result.setdefault("kiri_lora_name_hints", []).append({
                "name": str(item["name"]), "weight": item.get("strength_model", 1.0),
            })
    if loras:
        result["loras"] = loras
    return _json_clean(result)


def build_novelai_comment(record: dict) -> tuple[str, dict]:
    """Build portable NovelAI-style Description + Comment fields, not stealth alpha."""
    positive, negative = _prompts(record)
    settings = _first_settings(record)
    comment = {"prompt": positive, "uc": negative}
    for key, candidates in {
        "seed": ("seed", "noise_seed"),
        "steps": ("steps",),
        "scale": ("cfg", "cfg_scale"),
        "sampler": ("sampler_name", "sampler"),
    }.items():
        value = _first(settings, *candidates)
        if value is not None:
            comment[key] = value
    return positive, _json_clean(comment)


def build_easy_diffusion_request(record: dict, width: int, height: int) -> dict:
    positive, negative = _prompts(record)
    settings = _first_settings(record)
    resources = _resources(record)
    req_body = {
        "prompt": positive,
        "negative_prompt": negative,
        "width": int(width),
        "height": int(height),
    }
    aliases = {
        "seed": ("seed", "noise_seed"),
        "num_inference_steps": ("steps",),
        "guidance_scale": ("cfg", "cfg_scale"),
        "sampler_name": ("sampler_name", "sampler"),
        "scheduler_name": ("scheduler",),
        "prompt_strength": ("denoise",),
        "clip_skip": ("clip_skip",),
    }
    for key, candidates in aliases.items():
        value = _first(settings, *candidates)
        if value is not None:
            req_body[key] = value
    checkpoint = _resource_model_name(resources, "Checkpoint / Model")
    if checkpoint:
        req_body["use_stable_diffusion_model"] = checkpoint
    vae = _resource_model_name(resources, "VAE")
    if vae:
        req_body["use_vae_model"] = vae
    lora_resources = [item for item in resources if item.get("type") == "LoRA" and item.get("name")]
    if lora_resources:
        req_body["use_lora_model"] = [str(item["name"]) for item in lora_resources]
        strengths = [item.get("strength_model", 1.0) for item in lora_resources]
        # Easy Diffusion currently exposes one shared LoRA alpha rather than
        # per-LoRA weights. Only emit it if the Kiri weights agree.
        if len({str(value) for value in strengths}) == 1:
            try:
                req_body["lora_alpha"] = round(float(strengths[0]) * 100)
            except (TypeError, ValueError):
                pass
    return {"numOutputsTotal": 1, "reqBody": _json_clean(req_body)}



def build_easy_diffusion_text(record: dict, width: int, height: int) -> str:
    """Build Easy Diffusion's documented drop-in TXT task format."""
    positive, negative = _prompts(record)
    settings = _first_settings(record)
    lines = [positive, f"Width: {int(width)}", f"Height: {int(height)}"]
    seed = _first(settings, "seed", "noise_seed")
    lines.append(f"Seed: {seed if seed is not None else -1}")
    aliases = (
        ("Steps", ("steps",)), ("Guidance Scale", ("cfg", "cfg_scale")),
        ("Prompt Strength", ("denoise",)), ("Sampler", ("sampler_name", "sampler")),
        ("Scheduler", ("scheduler",)),
    )
    for label, candidates in aliases:
        value = _first(settings, *candidates)
        if value is not None:
            lines.append(f"{label}: {value}")
    if negative:
        lines.append(f"Negative Prompt: {negative}")
    resources = _resources(record)
    checkpoint = _resource_model_name(resources, "Checkpoint / Model")
    vae = _resource_model_name(resources, "VAE")
    if checkpoint:
        lines.append(f"Stable Diffusion model: {checkpoint}")
    if vae:
        lines.append(f"VAE: {vae}")
    loras = [item.get("name") for item in resources if item.get("type") == "LoRA" and item.get("name")]
    if loras:
        lines.append("LoRAs: " + ", ".join(str(x) for x in loras))
    return "\n".join(lines)


def build_novelai_stealth_metadata(record: dict) -> dict:
    """Build the gzip-JSON payload used by NovelAI stealth_pngcomp alpha metadata."""
    description, comment = build_novelai_comment(record)
    return {"Description": description, "Comment": json.dumps(comment, ensure_ascii=False, separators=(",", ":"))}


def build_novelai_stealth_payload(record: dict) -> bytes:
    """Serialize NovelAI stealth_pngcomp header + gzip payload for alpha LSBs."""
    raw = json.dumps(build_novelai_stealth_metadata(record), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    compressed = gzip.compress(raw, mtime=0)
    return b"stealth_pngcomp" + (len(compressed) * 8).to_bytes(4, "big") + compressed


def embed_novelai_stealth(image, record: dict):
    """Return an RGBA copy with a NovelAI-compatible stealth payload in alpha LSBs.

    Existing alpha values are preserved to within one least-significant bit. If
    the image is too small for the payload, the original image is returned.
    """
    try:
        import numpy as np
        from PIL import Image
        payload = build_novelai_stealth_payload(record)
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
        width, height = image.size
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
        if bits.size > width * height:
            return image
        alpha_order = rgba[..., 3].T.reshape(-1).copy()
        alpha_order[:bits.size] = (alpha_order[:bits.size] & 0xFE) | bits
        rgba[..., 3] = alpha_order.reshape((width, height)).T
        return Image.fromarray(rgba, "RGBA")
    except Exception:
        return image

def build_interchange_manifest(record: dict, width: int, height: int) -> dict:
    """Portable handoff manifest for apps with no verified native PNG writer contract."""
    positive, negative = _prompts(record)
    settings = _first_settings(record)
    resources = _resources(record)
    checkpoint = _first_resource(resources, "Checkpoint / Model")
    loras = []
    for item in resources:
        if item.get("type") == "LoRA":
            loras.append({
                "name": item.get("name"),
                "strength_model": item.get("strength_model"),
                "strength_clip": item.get("strength_clip"),
                "sha256": _sha256(item),
                "civitai": _civitai_provenance(item) or None,
            })
    civitai_resources = []
    for item in resources:
        prov = _civitai_provenance(item)
        if prov.get("version_id"):
            civitai_resources.append({
                "type": {
                    "LoRA": "lora", "Checkpoint / Model": "checkpoint",
                    "VAE": "vae", "Text Encoder": "textualinversion",
                }.get(str(item.get("type")), "model"),
                "modelVersionId": str(prov["version_id"]),
                "modelId": str(prov["model_id"]) if prov.get("model_id") else None,
                "name": item.get("name"),
            })
    # PixAI v2 API: modelVersionId is mandatory, so only include it when a
    # PixAI-native ID was explicitly captured. A Civitai model version is not
    # interchangeable with a PixAI model version.
    pixai_model = _first_resource(resources, "Checkpoint / Model")
    pixai_version = _provider_id(pixai_model or {}, "pixai", "model_version_id", "modelVersionId", "version_id") if pixai_model else None
    pixai_loras = []
    for item in resources:
        if item.get("type") == "LoRA":
            version_id = _provider_id(item, "pixai", "model_version_id", "modelVersionId", "version_id")
            if version_id:
                pixai_loras.append({"modelVersionId": version_id, "weight": item.get("strength_model", 1.0)})
    pixai = {
        "prompt": positive, "negativePrompt": negative,
        "width": int(width), "height": int(height),
    }
    seed = _first(settings, "seed", "noise_seed")
    if seed is not None:
        pixai["seed"] = seed
    sampling = {}
    for output_key, candidates in {
        "steps": ("steps",), "cfgScale": ("cfg", "cfg_scale"),
        "sampler": ("sampler_name", "sampler"), "scheduler": ("scheduler",),
    }.items():
        value = _first(settings, *candidates)
        if value is not None:
            sampling[output_key] = value
    if sampling:
        pixai["sampling"] = sampling
    if pixai_version:
        pixai["modelVersionId"] = pixai_version
    if pixai_loras:
        pixai["loras"] = pixai_loras

    # Tensor.Art TAMS API v1 workflow with the documented stage structure.
    tensor_art = build_tensorart_request(record, width, height)
    lora_items = [
        item for item in resources
        if item.get("type") == "LoRA"
        and _provider_id(item, "tensor_art", "model_id", "modelId", "loraModel")
    ]
    seaart = {
        "prompt": positive, "negative_prompt": negative,
        "width": int(width), "height": int(height),
        "seed": seed, "steps": _first(settings, "steps"),
        "cfg_scale": _first(settings, "cfg", "cfg_scale"),
        "sampler": _first(settings, "sampler_name", "sampler"),
        "scheduler": _first(settings, "scheduler"),
        "loras": loras,
        "_compatibility": "Portable SeaArt UI handoff only. No stable public image-to-parameter-import contract was found; values must be entered or adapted in the target UI."
    }
    tensorhub = {
        "prompt": positive, "negative_prompt": negative,
        "width": int(width), "height": int(height),
        "seed": seed, "steps": _first(settings, "steps"),
        "cfg_scale": _first(settings, "cfg", "cfg_scale"),
        "sampler": _first(settings, "sampler_name", "sampler"),
        "scheduler": _first(settings, "scheduler"),
        "loras": loras,
        "_compatibility": "Portable TensorHub / Cozy Creator UI handoff only; no stable public image-to-parameter-import contract is verified.",
    }
    # Naifu can consume the conventional prompt/generation fields, but a
    # separate native PNG import contract is not asserted. This is a copyable
    # parameter handoff for compatible local builds.
    naifu = {
        "prompt": positive, "negative_prompt": negative,
        "width": int(width), "height": int(height),
        "seed": seed, "steps": _first(settings, "steps"),
        "scale": _first(settings, "cfg", "cfg_scale"),
        "sampler": _first(settings, "sampler_name", "sampler"),
        "scheduler": _first(settings, "scheduler"),
        "model": checkpoint.get("name") if checkpoint else None,
        "loras": loras,
        "_compatibility": "Portable Naifu-style parameter handoff; a separate native import schema is not verified.",
    }
    fooocus_handoff = build_fooocus_payload(record, width, height)
    tensor_art_notes = []
    if pixai_version is None:
        tensor_art_notes.append("PixAI: select/copy the destination model's PixAI-native modelVersionId before API submission.")
    if checkpoint and not _provider_id(checkpoint, "tensor_art", "model_id", "modelId", "sdModel"):
        tensor_art_notes.append(f"Tensor.Art: select the matching Tensor.Art model for {checkpoint.get('name')}; sdModel needs a Tensor.Art-native model ID.")
    if any(item.get("type") == "LoRA" for item in resources) and not lora_items:
        tensor_art_notes.append("Tensor.Art: LoRA filenames and weights are preserved in the manifest, but the request needs Tensor.Art-native LoRA IDs.")
    if any(item.get("type") == "LoRA" for item in resources) and not pixai_loras:
        tensor_art_notes.append("PixAI: LoRA filenames and weights are preserved in the manifest, but the request needs PixAI-native LoRA version IDs.")
    tensor_art_notes.append("Never substitute Civitai IDs or local filenames for PixAI/Tensor.Art catalogue IDs.")
    tensor_art_notes.append("SeaArt and TensorHub / Cozy Creator do not expose a stable documented image-to-parameter import format here; their objects are editable handoff templates.")

    provider_ids = {}
    for item in resources:
        name = str(item.get("name") or "")
        if not name:
            continue
        for provider, keys in {
            "pixai": ("model_version_id", "modelVersionId", "version_id"),
            "tensor_art": ("model_id", "modelId", "sdModel", "loraModel", "sdVae"),
        }.items():
            provider_data = item.get(f"{provider}_provenance")
            if isinstance(provider_data, dict):
                for key in keys:
                    if provider_data.get(key) not in (None, ""):
                        provider_ids.setdefault(provider, []).append({
                            "resource_name": name, "resource_type": item.get("type"),
                            "field": key, "id": str(provider_data[key]),
                        })
                        break
    return _json_clean({
        "schema": "kiri-universal-generation-metadata",
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "producer": f"Kiri Image Loader {VERSION}",
        "generation": {
            "positive_prompt": positive, "negative_prompt": negative,
            "width": int(width), "height": int(height),
            "sampler_settings": settings,
        },
        "resources": resources,
        "provider_native_ids": provider_ids,
        "civitai_resources": civitai_resources,
        "civitai_meta": {
            "prompt": positive, "negativePrompt": negative,
            "Size": f"{int(width)}x{int(height)}",
            "seed": _first(settings, "seed", "noise_seed"),
            "steps": _first(settings, "steps"),
            "sampler": _first(settings, "sampler_name", "sampler"),
            "cfgScale": _first(settings, "cfg", "cfg_scale"),
            "resources": civitai_resources,
            "civitaiResources": civitai_resources,
        },
        "compatibility_notes": tensor_art_notes,
        "platform_handoff": {
            "pixai_api": pixai,
            "tensor_art": tensor_art,
            "seaart": seaart,
            "tensorhub_cozy_creator": tensorhub,
            "easy_diffusion": build_easy_diffusion_request(record, width, height),
            "fooocus": fooocus_handoff,
            "naifu": naifu,
        },
    })


def build_export_chunks(record: dict, width: int, height: int, settings: Any = None) -> dict[str, str]:
    """Return distinct PNG text chunks for the selected compatibility profile."""
    prefs = normalize_export_settings(settings)
    record = _apply_prompt_export_preferences(record, prefs)
    chunks: dict[str, str] = {}
    if prefs["parameters_format"] == "swarmui":
        chunks["parameters"] = json.dumps(build_swarmui_payload(record, width, height), ensure_ascii=False, separators=(",", ":"))
    elif prefs["parameters_format"] == "fooocus":
        chunks["parameters"] = json.dumps(build_fooocus_payload(record, width, height), ensure_ascii=False, separators=(",", ":"))
        chunks["fooocus_scheme"] = "fooocus"
    elif prefs["parameters_format"] == "easy_diffusion":
        chunks["parameters"] = json.dumps(build_easy_diffusion_request(record, width, height), ensure_ascii=False, separators=(",", ":"))
    # In the A1111 profile, the caller writes the standard `parameters` text.
    if prefs["invokeai"]:
        chunks["invokeai_metadata"] = json.dumps(build_invokeai_metadata(record, width, height), ensure_ascii=False, separators=(",", ":"))
    if prefs["novelai"]:
        description, comment = build_novelai_comment(record)
        if description:
            chunks["Description"] = description
        chunks["Comment"] = json.dumps(comment, ensure_ascii=False, separators=(",", ":"))
        chunks["Software"] = f"Kiri Image Loader {VERSION}"
        if record.get("created_utc"):
            chunks["Generation time"] = str(record["created_utc"])
        chunks["novelai_stealth"] = json.dumps(build_novelai_stealth_metadata(record), ensure_ascii=False, separators=(",", ":"))
    # Draw Things reads this JSON from EXIF UserComment. It is independent of
    # the PNG `parameters` profile, so it can coexist with A1111/SwarmUI/Fooocus.
    chunks["drawthings_user_comment"] = json.dumps(build_draw_things_metadata(record, width, height), ensure_ascii=False, separators=(",", ":"))
    chunks["XML:com.adobe.xmp"] = build_draw_things_xmp(record, width, height)
    # Easy Diffusion documents a task JSON envelope for restoring all fields
    # from an embedded image. Keep a named copy even when another profile owns
    # the conventional `parameters` key.
    easy_diffusion_request = build_easy_diffusion_request(record, width, height)
    chunks["sd_metadata"] = json.dumps(easy_diffusion_request, ensure_ascii=False, separators=(",", ":"))
    chunks["easy_diffusion_task"] = json.dumps(easy_diffusion_request, ensure_ascii=False, separators=(",", ":"))
    chunks["easy_diffusion_text"] = build_easy_diffusion_text(record, width, height)
    # Separate, clearly named chunks make handoff requests easy to find/copy in
    # Kiri's metadata viewer. They are not claimed as site-native PNG keys.
    if prefs["universal_bundle"]:
        manifest = build_interchange_manifest(record, width, height)
        chunks["kiri_interop"] = json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))
        handoff = manifest.get("platform_handoff", {})
        for source_key, chunk_key in (("pixai_api", "pixai_api"), ("tensor_art", "tensor_art_tams"),
                                      ("seaart", "seaart_handoff"), ("tensorhub_cozy_creator", "tensorhub_cozy_creator_handoff"),
                                      ("fooocus", "fooocus_handoff"), ("naifu", "naifu_handoff")):
            if source_key in handoff:
                chunks[chunk_key] = json.dumps(handoff[source_key], ensure_ascii=False, separators=(",", ":"))
        if manifest.get("civitai_meta"):
            chunks["civitai_meta"] = json.dumps(manifest["civitai_meta"], ensure_ascii=False, separators=(",", ":"))
    return chunks


def build_sidecar(record: dict, width: int, height: int, settings: Any = None) -> dict:
    """Build a complete portable .kiri.json sidecar when explicitly enabled."""
    prefs = normalize_export_settings(settings)
    record = _apply_prompt_export_preferences(record, prefs)
    return {
        "schema": "kiri-universal-generation-metadata",
        "schema_version": 1,
        "producer": f"Kiri Image Loader {VERSION}",
        "kiri_metadata": _json_clean(record),
        "adapters": _json_clean(build_interchange_manifest(record, width, height)),
        "export_settings": prefs,
    }
