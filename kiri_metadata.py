"""Metadata extraction for Kiri Image Loader.

The extractor deliberately preserves raw embedded metadata while also normalizing
common generation parameters into individual copy actions. It is dependency-light:
Pillow is already part of ComfyUI's standard image stack.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import re
import struct
import zlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from PIL import Image, ExifTags
import numpy as np

SUPPORTED_METADATA_FORMATS = {"PNG", "JPEG", "WEBP", "TIFF", "BMP", "GIF"}


def _json_safe(value: Any, depth: int = 0) -> Any:
    if depth > 12:
        return "<max depth>"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        for encoding in ("utf-8", "utf-16", "latin-1"):
            try:
                decoded = value.decode(encoding).strip("\x00\ufeff ")
                if decoded:
                    return decoded
            except Exception:
                pass
        return value.hex()
    if isinstance(value, dict):
        return {str(k): _json_safe(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v, depth + 1) for v in value]
    try:
        return str(value)
    except Exception:
        return repr(value)


def _as_dict(value: Any) -> dict | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, bytes):
        value = _json_safe(value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                parsed = json.loads(text)
                return parsed if isinstance(parsed, dict) else {"items": parsed}
            except Exception:
                return None
    return None


def _norm_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), ensure_ascii=False, indent=2)
    return str(value)


def _split_a1111_parameters(text: str) -> tuple[str, str, str, dict]:
    """Parse A1111-style prompt block without splitting commas inside quotes."""
    if not text:
        return "", "", "", {}

    steps_match = re.search(r"(?:^|\n)\s*Steps\s*:", text, flags=re.IGNORECASE)
    settings_start = steps_match.start() if steps_match else -1
    prompt_part = text[:settings_start] if settings_start >= 0 else text
    settings_text = text[settings_start:].strip() if settings_start >= 0 else ""

    negative = ""
    negative_match = re.search(r"(?:^|\n)\s*Negative prompt\s*:", prompt_part, flags=re.IGNORECASE)
    if negative_match:
        positive = prompt_part[:negative_match.start()].strip()
        negative = prompt_part[negative_match.end():].strip()
    else:
        positive = prompt_part.strip()

    pairs: dict[str, str] = {}
    if settings_text:
        try:
            fields = next(csv.reader([settings_text], skipinitialspace=True))
        except Exception:
            fields = settings_text.split(",")
        # CSV parsing is quote-aware but leaves key/value boundaries intact.
        pending = []
        for token in fields:
            if ":" in token:
                if pending:
                    joined = ",".join(pending).strip()
                    if ":" in joined:
                        key, value = joined.split(":", 1)
                        pairs[key.strip()] = value.strip().strip('"')
                pending = [token]
            elif pending:
                pending.append(token)
        if pending:
            joined = ",".join(pending).strip()
            if ":" in joined:
                key, value = joined.split(":", 1)
                pairs[key.strip()] = value.strip().strip('"')

    # Quoted field values (e.g. Model name) are handled more reliably by a
    # regex fallback over the original settings text.
    for match in re.finditer(r'(?:^|,\s*)([^:,]+):\s*("(?:[^"\\]|\\.)*"|[^,]+)', settings_text):
        key = match.group(1).strip()
        value = match.group(2).strip().strip('"')
        pairs.setdefault(key, value)

    return positive, negative, settings_text, pairs


def _decode_metadata_text(value: Any) -> str:
    if isinstance(value, bytes):
        # EXIF UserComment prefixes: 8-byte ASCII / UNICODE / JIS markers.
        if value.startswith(b"ASCII\x00\x00\x00"):
            value = value[8:]
        elif value.startswith(b"UNICODE\x00"):
            value = value[8:]
            for encoding in ("utf-16-be", "utf-16-le", "utf-16"):
                try:
                    return value.decode(encoding).strip("\x00\ufeff ")
                except Exception:
                    pass
        for encoding in ("utf-8", "utf-16", "latin-1"):
            try:
                return value.decode(encoding).strip("\x00\ufeff ")
            except Exception:
                continue
        return value.hex()
    return str(value) if value is not None else ""


def _read_exif(image: Image.Image) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        exif = image.getexif()
        for tag_id, value in exif.items():
            tag = ExifTags.TAGS.get(tag_id, str(tag_id))
            result[str(tag)] = _json_safe(value)
    except Exception:
        pass
    return result


def _extract_novelai_stealth(image: Image.Image) -> dict | None:
    """Decode NovelAI's documented alpha-channel stealth_pngcomp payload.

    This reads the metadata payload only; it does not attempt signature checks or
    forward-error correction. The bit ordering follows NovelAI's public
    nai_meta.py extraction algorithm.
    """
    try:
        if "A" not in image.getbands():
            return None
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        alpha_bits = np.bitwise_and(rgba[..., 3].T.reshape(-1), 1)
        byte_aligned = alpha_bits[: (alpha_bits.size // 8) * 8]
        packed = np.packbits(byte_aligned.reshape(-1, 8), axis=1).reshape(-1).tobytes()

        magic = b"stealth_pngcomp"
        header_end = len(magic) + 4
        if len(packed) < header_end or packed[:len(magic)] != magic:
            return None

        payload_bits = int.from_bytes(packed[len(magic):header_end], "big")
        payload_length = payload_bits // 8
        if payload_length <= 0 or payload_length > len(packed) - header_end:
            return None

        compressed = packed[header_end:header_end + payload_length]
        # Bound decompression defensively; metadata should be small and is read
        # from an image-controlled payload.
        with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb") as stream:
            decoded = stream.read(16 * 1024 * 1024 + 1)
        if len(decoded) > 16 * 1024 * 1024:
            return None
        value = json.loads(decoded.decode("utf-8"))
        if isinstance(value, dict):
            comment = value.get("Comment")
            if isinstance(comment, str):
                try:
                    value["Comment"] = json.loads(comment)
                except Exception:
                    pass
            return value
    except Exception:
        return None
    return None


def _xml_local_name(value: str) -> str:
    return str(value).rsplit("}", 1)[-1].split(":")[-1]


def _parse_xmp_metadata(value: str, put) -> bool:
    """Read XMP/RDF properties, including nested rdf:Alt/rdf:li values.

    XMP properties such as dc:description and exif:UserComment commonly wrap
    their text in RDF containers. Reading only leaf element names loses the
    property name (the leaf is simply ``rdf:li``), so walk properties directly
    under each rdf:Description and flatten their descendants.
    """
    try:
        root = ET.fromstring(value)
    except Exception:
        return False

    extracted: dict[str, list[str]] = {}
    for description in (item for item in root.iter() if _xml_local_name(item.tag).lower() == "description"):
        # Attributes on rdf:Description can themselves carry XMP properties.
        for key, attr_value in description.attrib.items():
            if _xml_local_name(key).lower() in {"about", "lang"}:
                continue
            text_value = str(attr_value).strip()
            if text_value:
                extracted.setdefault(_xml_local_name(key), []).append(text_value)
        for prop in list(description):
            prop_name = _xml_local_name(prop.tag)
            leaf_values = []
            for child in prop.iter():
                if not list(child):
                    leaf_text = (child.text or "").strip()
                    if leaf_text:
                        leaf_values.append(leaf_text)
            if leaf_values:
                extracted.setdefault(prop_name, []).append("\n".join(leaf_values))

    # Some compact/legacy XMP snippets omit rdf:Description and put properties
    # directly in the RDF tree. Preserve those recognized leaves as fallback.
    if not extracted:
        for element in root.iter():
            if list(element):
                continue
            local = _xml_local_name(element.tag).lower()
            text_value = (element.text or "").strip()
            if text_value and local not in {"li", "alt", "seq", "bag"}:
                extracted.setdefault(_xml_local_name(element.tag), []).append(text_value)

    if not extracted:
        return False

    for key, values in extracted.items():
        for index, field_value in enumerate(values, start=1):
            field_name = key if index == 1 else f"{key} {index}"
            normalized = _norm_key(key)
            if normalized == "usercomment":
                user_comment = _as_dict(field_value)
                if user_comment:
                    _collect_known_from_dict(user_comment, put)
                    put("Draw Things UserComment JSON", user_comment, "Raw Metadata")
                else:
                    put(f"XMP · {field_name}", field_value, "Image Metadata")
            elif normalized == "description":
                _parse_settings_string(field_value, put)
                put("Draw Things Description", field_value, "Raw Metadata")
            else:
                _collect_known_from_dict({key: field_value}, put)
                if normalized not in _XMP_KNOWN_KEYS:
                    put(f"XMP · {field_name}", field_value, "Image Metadata")
    put("XMP Metadata XML", value, "Raw Metadata")
    return True


_XMP_KNOWN_KEYS = {
    "prompt", "positiveprompt", "negativeprompt", "uc", "c", "seed", "steps",
    "cfg", "cfgscale", "scale", "guidancescale", "sampler", "samplername", "scheduler",
    "model", "modelname", "width", "height", "size", "lora", "loras",
    "clipskip", "strength", "noise",
}


def _parse_novelai(data: dict, put):
    """Normalize NovelAI legacy and v4 metadata while retaining unknown fields."""
    _collect_known_from_dict(data, put)
    description = _get_norm(data, "description")
    if description:
        put("Positive Prompt", description, "Prompts")

    comment = _get_norm(data, "comment")
    if isinstance(comment, str):
        comment = _as_dict(comment)
    if isinstance(comment, dict):
        _collect_known_from_dict(comment, put)
        for nested_field, nested_label in (("v4_prompt", "Positive Prompt"), ("v4_negative_prompt", "Negative Prompt")):
            nested_value = _get_norm(comment, nested_field)
            if isinstance(nested_value, dict):
                nested_caption = _get_norm(nested_value, "caption")
                if isinstance(nested_caption, dict):
                    nested_base = _get_norm(nested_caption, "base_caption", "baseCaption")
                    if nested_base:
                        put(nested_label, nested_base, "Prompts")
                    nested_chars = _get_norm(nested_caption, "char_captions", "charCaptions")
                    if isinstance(nested_chars, list):
                        for index, item in enumerate(nested_chars, start=1):
                            if isinstance(item, dict):
                                char_text = _get_norm(item, "char_caption", "charCaption", "caption")
                                if char_text:
                                    put(f"{nested_label} · Character {index}", char_text, "Prompts")

    for field_name, label in (("v4_prompt", "Positive Prompt"), ("v4_negative_prompt", "Negative Prompt")):
        value = _get_norm(data, field_name)
        if not isinstance(value, dict):
            continue
        caption = _get_norm(value, "caption")
        if isinstance(caption, dict):
            base = _get_norm(caption, "base_caption", "baseCaption")
            if base:
                put(label, base, "Prompts")
            characters = _get_norm(caption, "char_captions", "charCaptions")
            if isinstance(characters, list):
                for index, item in enumerate(characters, start=1):
                    if isinstance(item, dict):
                        text_value = _get_norm(item, "char_caption", "charCaption", "caption")
                        if text_value:
                            put(f"{label} · Character {index}", text_value, "Prompts")
        # Keep the structured v4 prompt as an individual field too.
        put(f"NovelAI {field_name} JSON", value, "Raw Metadata")


def _iter_scalars(data: Any, prefix: str = "", depth: int = 0):
    if depth > 6:
        return
    if isinstance(data, dict):
        for key, value in data.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, (dict, list)):
                yield from _iter_scalars(value, name, depth + 1)
            elif value is not None and str(value).strip() != "":
                yield name, value
    elif isinstance(data, list):
        for index, value in enumerate(data):
            name = f"{prefix}[{index}]"
            if isinstance(value, (dict, list)):
                yield from _iter_scalars(value, name, depth + 1)
            elif value is not None and str(value).strip() != "":
                yield name, value


def _parse_comfy_graph(data: dict, put):
    """Extract useful scalar inputs from ComfyUI prompt/workflow JSON."""
    nodes = data.get("nodes") if isinstance(data.get("nodes"), list) else data
    if isinstance(nodes, dict):
        sampler_nodes = [
            (node_id, node)
            for node_id, node in nodes.items()
            if isinstance(node, dict)
            and "ksampler" in str(node.get("class_type") or node.get("type") or "").lower()
        ]
        multiple_samplers = len(sampler_nodes) > 1
        for node_id, node in nodes.items():
            if not isinstance(node, dict):
                continue
            class_type = node.get("class_type") or node.get("type") or node.get("_meta", {}).get("title") or "Node"
            inputs = node.get("inputs") or {}
            if not isinstance(inputs, dict):
                continue
            low_type = str(class_type).lower()
            title = f"{class_type} {node_id}"

            common_keys = set()
            if "ksampler" in low_type:
                common_keys.update({"seed", "noise_seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"})
                aliases = {
                    "Seed": ("seed", "noise_seed"), "Steps": ("steps",),
                    "CFG": ("cfg",), "Sampler": ("sampler_name",),
                    "Scheduler": ("scheduler",), "Denoise": ("denoise",),
                }
                for label, keys in aliases.items():
                    for key in keys:
                        if key in inputs:
                            unique_label = f"{label} ({class_type} {node_id})" if multiple_samplers else label
                            put(unique_label, inputs[key], "Generation")
                            break
            if "checkpointloader" in low_type or "unclipcheckpoint" in low_type:
                common_keys.update({"ckpt_name", "checkpoint", "model_name"})
                for key in ("ckpt_name", "checkpoint", "model_name"):
                    if key in inputs:
                        put("Checkpoint", inputs[key], "Models")
                        break
            if "vaeloader" in low_type:
                common_keys.add("vae_name")
                if "vae_name" in inputs:
                    put("VAE", inputs["vae_name"], "Models")
            if "loraloader" in low_type:
                common_keys.update({"lora_name", "strength_model", "strength_clip", "strength"})
                name = inputs.get("lora_name")
                if name:
                    put(f"LoRA · {name} · Name", name, "LoRAs")
                for key in ("strength_model", "strength_clip", "strength"):
                    if key in inputs:
                        put(f"LoRA · {name or node_id} · {key}", inputs[key], "LoRAs")
            if "cliptextencode" in low_type or "textencode" in low_type:
                common_keys.add("text")
                if "text" in inputs:
                    put(f"{title} · Text", inputs["text"], "Prompts")

            for key, value in inputs.items():
                if key in common_keys or isinstance(value, (dict, list)):
                    continue
                if isinstance(value, (int, float, bool)) or (isinstance(value, str) and value.strip()):
                    put(f"{title} · {key}", value, "ComfyUI Nodes")

    # Some workflow exports keep widgets_values rather than API prompt inputs.
    if isinstance(data.get("nodes"), list):
        for node in data["nodes"]:
            if not isinstance(node, dict):
                continue
            title = node.get("title") or node.get("type") or f"Node {node.get('id', '')}"
            values = node.get("widgets_values")
            if isinstance(values, list):
                for index, value in enumerate(values):
                    if isinstance(value, (str, int, float, bool)) and str(value).strip():
                        put(f"{title} · Widget {index + 1}", value, "ComfyUI Nodes")


def _collect_known_from_dict(data: dict, put):
    aliases = {
        "positiveprompt": ("Positive Prompt", "Prompts"),
        "prompt": ("Positive Prompt", "Prompts"),
        "c": ("Positive Prompt", "Prompts"),
        "text": ("Prompt Text", "Prompts"),
        "textg": ("CLIP G Prompt", "Prompts"),
        "textl": ("CLIP L Prompt", "Prompts"),
        "clipg": ("CLIP G Prompt", "Prompts"),
        "clipl": ("CLIP L Prompt", "Prompts"),
        "negativeprompt": ("Negative Prompt", "Prompts"),
        "uc": ("Negative Prompt", "Prompts"),
        "negative": ("Negative Prompt", "Prompts"),
        "seed": ("Seed", "Generation"),
        "steps": ("Steps", "Generation"),
        "cfg": ("CFG", "Generation"),
        "cfgscale": ("CFG", "Generation"),
        "guidancescale": ("Guidance Scale", "Generation"),
        "guidance": ("Guidance Scale", "Generation"),
        "scale": ("CFG", "Generation"),
        "cfg_scale": ("CFG", "Generation"),
        "guidance_scale": ("Guidance Scale", "Generation"),
        "sampler": ("Sampler", "Generation"),
        "samplername": ("Sampler", "Generation"),
        "sampler_name": ("Sampler", "Generation"),
        "scheduler": ("Scheduler", "Generation"),
        "schedulername": ("Scheduler", "Generation"),
        "scheduler_name": ("Scheduler", "Generation"),
        "numinferencesteps": ("Steps", "Generation"),
        "inferencessteps": ("Steps", "Generation"),
        "denoise": ("Denoise", "Generation"),
        "denoisingstrength": ("Denoise", "Generation"),
        "model": ("Checkpoint", "Models"),
        "modelname": ("Checkpoint", "Models"),
        "ckptname": ("Checkpoint", "Models"),
        "modelhash": ("Checkpoint Hash", "Models"),
        "model_hash": ("Checkpoint Hash", "Models"),
        "modelid": ("Model ID", "Models"),
        "model_id": ("Model ID", "Models"),
        "modelversionid": ("Model Version ID", "Models"),
        "model_version_id": ("Model Version ID", "Models"),
        "sdmodel": ("Platform Model ID", "Models"),
        "sdvae": ("Platform VAE ID", "Models"),
        "loramodel": ("Platform LoRA ID", "LoRAs"),
        "basemodel": ("Base Model", "Models"),
        "basemodelhash": ("Base Model Hash", "Models"),
        "refinermodelhash": ("Refiner Model Hash", "Models"),
        "use_stable_diffusion_model": ("Checkpoint", "Models"),
        "usestablediffusionmodel": ("Checkpoint", "Models"),
        "use_vae_model": ("VAE", "Models"),
        "usevaemodel": ("VAE", "Models"),
        "vae": ("VAE", "Models"),
        "vaename": ("VAE", "Models"),
        "clip_skip": ("CLIP Skip", "Advanced"),
        "clipskip": ("CLIP Skip", "Advanced"),
        "clipstopatlastlayer": ("CLIP Skip", "Advanced"),
        "appversion": ("Generator Version", "Image Metadata"),
        "swarmversion": ("SwarmUI Version", "Image Metadata"),
        "metadataversion": ("Metadata Version", "Image Metadata"),
        "width": ("Width", "Image"),
        "height": ("Height", "Image"),
        "upscaler": ("Upscaler", "Advanced"),
        "upscalername": ("Upscaler", "Advanced"),
        "refinermodel": ("Refiner Model", "Advanced"),
        "refiner_model": ("Refiner Model", "Advanced"),
        "refinersteps": ("Refiner Steps", "Advanced"),
        "hiresfix": ("Hires Fix Enabled", "Advanced"),
        "hiresupscaler": ("Hires Upscaler", "Advanced"),
        "hiresupscale": ("Hires Upscale", "Advanced"),
        "variationseed": ("Variation Seed", "Generation"),
        "subseed": ("Variation Seed", "Generation"),
        "subseedstrength": ("Variation Strength", "Generation"),
        "seedresizefromw": ("Seed Resize Width", "Advanced"),
        "seedresizefromh": ("Seed Resize Height", "Advanced"),
        "eta": ("ETA", "Advanced"),
        "etanoiseseeddelta": ("ETA Noise Seed Delta", "Advanced"),
        "schurn": ("S Churn", "Advanced"),
        "stmin": ("S Tmin", "Advanced"),
        "stmax": ("S Tmax", "Advanced"),
        "snoise": ("S Noise", "Advanced"),
        "tiling": ("Tiling", "Advanced"),
        "enablehr": ("Hires Fix Enabled", "Advanced"),
        "hrscale": ("Hires Scale", "Advanced"),
        "hrupscaler": ("Hires Upscaler", "Advanced"),
        "hrsecondpasssteps": ("Hires Steps", "Advanced"),
        "hrresizex": ("Hires Resize Width", "Advanced"),
        "hrresizey": ("Hires Resize Height", "Advanced"),
        "refinercheckpoint": ("Refiner Model", "Advanced"),
        "refinerswitchat": ("Refiner Switch At", "Advanced"),
        "restorefaces": ("Face Restoration", "Advanced"),
        "facerestoration": ("Face Restoration", "Advanced"),
        "smea": ("SMEA", "Advanced"),
        "sm": ("SMEA", "Advanced"),
        "smeadyn": ("SMEA Dynamic", "Advanced"),
        "smdyn": ("SMEA Dynamic", "Advanced"),
        "dynamicthresholding": ("Dynamic Thresholding", "Advanced"),
        "cfgrescale": ("CFG Rescale", "Advanced"),
        "noiseschedule": ("Noise Schedule", "Advanced"),
        "uncondscale": ("Unconditional Scale", "Advanced"),
        "qualitytoggle": ("Quality Toggle", "Advanced"),
        "fooocusv2expansion": ("Fooocus V2 Expansion", "Advanced"),
        "fooocusstyles": ("Fooocus Styles", "Advanced"),
        "fooocusperformance": ("Fooocus Performance", "Advanced"),
        "fooocussharpness": ("Fooocus Sharpness", "Advanced"),
        "admguidance": ("ADM Guidance", "Advanced"),
        "adaptivecfg": ("Adaptive CFG", "Advanced"),
        "sharpness": ("Sharpness", "Advanced"),
        "strength": ("Strength", "Generation"),
        "noise": ("Noise", "Generation"),
        "resolution": ("Resolution", "Image"),
        "performance": ("Fooocus Performance", "Advanced"),
        "styles": ("Fooocus Styles", "Advanced"),
        "sharpness": ("Sharpness", "Advanced"),
        "adaptive_cfg": ("Adaptive CFG", "Advanced"),
        "adm_guidance": ("ADM Guidance", "Advanced"),
        "loraweights": ("LoRA Weights", "LoRAs"),
        "controlnets": ("ControlNet Settings", "Advanced"),
    }
    for key, value in data.items():
        norm = _norm_key(key)
        if norm in aliases and not isinstance(value, (dict, list)):
            label, group = aliases[norm]
            put(label, value, group)

    # Resource metadata formats used by tools such as NovelAI/Civitai may expose
    # LoRAs as an array of named resources.
    # Draw Things, Fooocus, Civitai and platform APIs use different LoRA arrays.
    resources = (data.get("resources") or data.get("civitai_resources") or
                 data.get("civitaiResources") or data.get("loras") or
                 data.get("lora") or data.get("items"))
    if isinstance(resources, dict):
        resources = resources.get("items") or resources.get("resources") or []
    if isinstance(resources, list):
        lora_index = 0
        for resource in resources:
            if isinstance(resource, (list, tuple)) and resource:
                name = resource[0]
                strength = resource[1] if len(resource) > 1 else None
                resource_type = "lora"
                resource_dict = {}
            elif isinstance(resource, dict):
                resource_dict = resource
                resource_type = str(resource.get("type", resource.get("modelType", ""))).lower()
                name = resource.get("name") or resource.get("model") or resource.get("loraModel") or resource.get("id")
                strength = resource.get("strength", resource.get("weight", resource.get("strength_model")))
            else:
                continue
            # Arrays under a key explicitly named `lora`/`loras` are LoRAs
            # even if their entries only contain {model, weight}.
            explicit_lora_array = any(str(k).lower() in {"lora", "loras", "used_loras", "lora_combined"} for k in data)
            if "lora" in resource_type or "lycoris" in resource_type or explicit_lora_array:
                lora_index += 1
                if name:
                    put(f"LoRA {lora_index} Name", name, "LoRAs")
                if strength is not None:
                    put(f"LoRA {lora_index} Strength", strength, "LoRAs")
                strength_clip = resource_dict.get("strength_clip", resource_dict.get("clip_strength")) if isinstance(resource_dict, dict) else None
                if strength_clip is not None:
                    put(f"LoRA {lora_index} CLIP Strength", strength_clip, "LoRAs")
                if isinstance(resource_dict, dict):
                    version_id = resource_dict.get("modelVersionId") or resource_dict.get("model_version_id")
                    model_id = resource_dict.get("modelId") or resource_dict.get("model_id")
                    if version_id is not None:
                        put(f"LoRA {lora_index} Model Version ID", version_id, "LoRAs")
                    if model_id is not None:
                        put(f"LoRA {lora_index} Model ID", model_id, "LoRAs")
            elif isinstance(resource_dict, dict):
                # Civitai resources also describe checkpoints, VAEs and textual
                # inversions. Preserve their provider IDs separately instead of
                # misclassifying every resource as a LoRA.
                resource_labels = {
                    "checkpoint": ("Checkpoint", "Models"),
                    "model": ("Model", "Models"),
                    "vae": ("VAE", "Models"),
                    "textualinversion": ("Textual Inversion", "Models"),
                }
                resource_label, resource_group = resource_labels.get(resource_type.replace("_", ""), ("Resource", "Models"))
                if name:
                    put(f"{resource_label} Resource", name, resource_group)
                version_id = resource_dict.get("modelVersionId") or resource_dict.get("model_version_id")
                model_id = resource_dict.get("modelId") or resource_dict.get("model_id")
                if version_id is not None:
                    put(f"{resource_label} Model Version ID", version_id, resource_group)
                if model_id is not None:
                    put(f"{resource_label} Model ID", model_id, resource_group)

    size_value = data.get("size") or data.get("resolution")
    if isinstance(size_value, str):
        match = re.search(r"(\d{2,5})\s*[xX*]\s*(\d{2,5})", size_value)
        if match:
            put("Width", int(match.group(1)), "Image")
            put("Height", int(match.group(2)), "Image")


def _collect_nested_platform_fields(data: Any, put, depth: int = 0):
    """Normalize parameter records nested in platform API envelopes.

    Handles Easy Diffusion reqBody, PixAI sampling/LoRA objects, Tensor.Art
    TAMS stages/diffusion records, Civitai meta and our Kiri interop manifest.
    Unknown values are still shown in the raw metadata viewer.
    """
    if depth > 7:
        return
    if isinstance(data, dict):
        _collect_known_from_dict(data, put)
        # Common platform aliases that have semantic differences at nested levels.
        sampling = data.get("sampling")
        if isinstance(sampling, dict):
            _collect_known_from_dict(sampling, put)
        generation = data.get("generation")
        if isinstance(generation, dict):
            _collect_nested_platform_fields(generation, put, depth + 1)
        req_body = data.get("reqBody") or data.get("req_body")
        if isinstance(req_body, dict):
            _collect_nested_platform_fields(req_body, put, depth + 1)
        for key in ("meta", "extraMetadata", "extra_metadata", "civitai_meta",
                    "sui_image_params", "sui_extra_data", "sui_models",
                    "diffusion", "platform_handoff", "resources", "civitaiResources"):
            value = data.get(key)
            if isinstance(value, (dict, list)):
                _collect_nested_platform_fields(value, put, depth + 1)
        stages = data.get("stages")
        if isinstance(stages, list):
            for stage in stages:
                if isinstance(stage, dict):
                    _collect_nested_platform_fields(stage, put, depth + 1)
        # Preserve TAMS prompt arrays as readable prompt text.
        for key, label in (("prompts", "Positive Prompt"), ("negativePrompts", "Negative Prompt")):
            value = data.get(key)
            if isinstance(value, list):
                snippets = [str(x.get("text", "")).strip() for x in value if isinstance(x, dict) and x.get("text")]
                if snippets:
                    put(label, "\n".join(snippets), "Prompts")
        # PixAI v2 and Tensor.Art LoRA item arrays.
        lora_obj = data.get("loras") or data.get("lora")
        if isinstance(lora_obj, dict) and isinstance(lora_obj.get("items"), list):
            _collect_known_from_dict({"loras": lora_obj["items"]}, put)
        for key, value in data.items():
            if isinstance(value, (dict, list)) and key not in {
                "generation", "reqBody", "req_body", "sampling", "meta",
                "extraMetadata", "extra_metadata", "platform_handoff",
                "stages", "resources", "civitaiResources", "sui_image_params",
                "sui_extra_data", "sui_models", "diffusion", "loras", "lora",
                "prompts", "negativePrompts", "civitai_meta",
            }:
                _collect_nested_platform_fields(value, put, depth + 1)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                _collect_nested_platform_fields(item, put, depth + 1)


def _parse_settings_string(text: str, put):
    if not text:
        return
    positive, negative, settings, pairs = _split_a1111_parameters(text)
    # The Civitai plugin appends a JSON array as one details item. It contains
    # commas and colons, so extract it from the original string before the
    # conventional CSV-like parameter parser can split its JSON internals.
    civitai_match = re.search(r'(?:,\s*|^)Civitai resources:\s*(\[.*\])\s*$', settings, flags=re.IGNORECASE | re.DOTALL)
    if civitai_match:
        try:
            civitai_items = json.loads(civitai_match.group(1))
            if isinstance(civitai_items, list):
                _collect_known_from_dict({"resources": civitai_items}, put)
        except Exception:
            pass
    if positive:
        put("Positive Prompt", positive, "Prompts")
    if negative:
        put("Negative Prompt", negative, "Prompts")

    aliases = {
        "steps": "Steps", "num inference steps": "Steps", "cfg scale": "CFG", "cfg": "CFG",
        "sampler": "Sampler", "sampler name": "Sampler",
        "scheduler": "Scheduler", "schedule type": "Scheduler",
        "guidance scale": "Guidance Scale", "subseed": "Variation Seed",
        "subseed strength": "Variation Strength", "variation seed": "Variation Seed",
        "variation seed strength": "Variation Strength", "face restoration": "Face Restoration",
        "seed": "Seed", "model": "Checkpoint", "checkpoint": "Checkpoint",
        "checkpoint / model": "Checkpoint", "model hash": "Checkpoint Hash",
        "checkpoint hash": "Checkpoint Hash", "vae": "VAE", "vae hash": "VAE Hash",
        "refiner": "Refiner Model", "refiner hash": "Refiner Hash", "clip skip": "CLIP Skip",
        "denoising strength": "Denoise", "denoise": "Denoise",
        "hires upscaler": "Hires Upscaler", "hires upscale": "Hires Upscale",
        "hires steps": "Hires Steps", "hires cfg scale": "Hires CFG",
        "refiner": "Refiner Model", "refiner model": "Refiner Model",
        "refiner switch at": "Refiner Switch At", "upscaler": "Upscaler",
        "width": "Width", "height": "Height", "hires steps": "Hires Steps",
        "hires cfg scale": "Hires CFG", "hires sampler": "Hires Sampler",
        "hires scheduler": "Hires Scheduler", "refiner switch at": "Refiner Switch At",
        "fooocus styles": "Fooocus Styles", "fooocus performance": "Fooocus Performance",
        "fooocus sharpness": "Fooocus Sharpness", "adm guidance": "ADM Guidance",
        "adaptive cfg": "Adaptive CFG", "sharpness": "Sharpness",
    }
    for key, value in pairs.items():
        lower = key.strip().lower()
        if lower in {"civitai resources", "civitairesources"}:
            parsed_resources = _as_dict(value)
            if parsed_resources is None:
                try:
                    parsed_resources = json.loads(value)
                except Exception:
                    parsed_resources = None
            if isinstance(parsed_resources, list):
                _collect_known_from_dict({"resources": parsed_resources}, put)
            else:
                put("Civitai Resources", value, "Models")
        elif lower == "size" and "x" in value:
            w, h = value.lower().split("x", 1)
            put("Width", w.strip(), "Image")
            put("Height", h.strip(), "Image")
        elif re.fullmatch(r"lora\s*\d+(?:\s+(?:name|hash))?", lower):
            match = re.fullmatch(r"lora\s*(\d+)(?:\s+(name|hash))?", lower)
            if match:
                index, suffix = match.groups()
                field_label = f"LoRA {index} {suffix.title()}" if suffix else f"LoRA {index} Name"
                if suffix == "name":
                    value_text = value
                elif suffix == "hash":
                    value_text = value
                else:
                    value_text = value
                    strength_match = re.search(r"\s+\[model=([^;\]]+)(?:;\s*clip=([^\]]+))?\]\s*$", value)
                    if strength_match:
                        value_text = value[:strength_match.start()].strip()
                        put(f"LoRA {index} Strength", strength_match.group(1).strip(), "LoRAs")
                        if strength_match.group(2):
                            put(f"LoRA {index} CLIP Strength", strength_match.group(2).strip(), "LoRAs")
                put(field_label, value_text, "LoRAs")
        elif lower in aliases:
            label = aliases[lower]
            group = "Image" if label in {"Width", "Height"} else ("Models" if label in {"Checkpoint", "Checkpoint Hash", "VAE", "VAE Hash", "Refiner Model"} else ("Advanced" if label not in {"Steps", "CFG", "Sampler", "Scheduler", "Seed", "Denoise"} else "Generation"))
            put(label, value, group)
        else:
            put(key, value, "Other Parameters")

    # Standard A1111 LoRA prompt syntax: <lora:name:weight>.
    loras = re.findall(r"<lora:([^:>]+)(?::([-+]?\d*\.?\d+))?(?::([-+]?\d*\.?\d+))?>", positive + "\n" + negative, re.IGNORECASE)
    for index, (name, model_strength, clip_strength) in enumerate(loras, start=1):
        put(f"LoRA {index} Name", name, "LoRAs")
        if model_strength:
            put(f"LoRA {index} Strength", model_strength, "LoRAs")
        if clip_strength:
            put(f"LoRA {index} CLIP Strength", clip_strength, "LoRAs")


def _parse_easy_diffusion_text(text: str, put):
    """Read Easy Diffusion's text task file used by drag-and-drop import."""
    if not text:
        return
    lines = text.splitlines()
    if lines:
        put("Positive Prompt", lines[0].strip(), "Prompts")
    aliases = {
        "width": ("Width", "Image"), "height": ("Height", "Image"),
        "seed": ("Seed", "Generation"), "steps": ("Steps", "Generation"),
        "guidance scale": ("Guidance Scale", "Generation"),
        "prompt strength": ("Denoise", "Generation"),
        "sampler": ("Sampler", "Generation"), "scheduler": ("Scheduler", "Generation"),
        "negative prompt": ("Negative Prompt", "Prompts"),
        "stable diffusion model": ("Checkpoint", "Models"), "vae": ("VAE", "Models"),
    }
    for line in lines[1:]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        output = aliases.get(key.strip().lower())
        if output:
            put(output[0], value.strip(), output[1])
        elif key.strip().lower().startswith("loras"):
            put("Easy Diffusion LoRAs", value.strip(), "LoRAs")
        elif key.strip() and value.strip():
            put(f"Easy Diffusion · {key.strip()}", value.strip(), "Other Parameters")


def _get_norm(data: dict, *names: str):
    # Respect caller preference order (e.g. an InvokeAI display name should
    # win over its internal model key regardless of JSON object key order).
    for name in names:
        wanted = _norm_key(name)
        for key, value in data.items():
            if _norm_key(key) == wanted:
                return value
    return None


def _parse_model_identifier(value: Any, label: str, category: str, put):
    """Extract model identity fields from InvokeAI/SwarmUI resource objects."""
    if isinstance(value, dict):
        # InvokeAI's ModelIdentifier.key is an internal model identifier;
        # prefer the human-facing name when one is present.
        name = _get_norm(value, "name", "model_name", "filename", "key")
        model_hash = _get_norm(value, "hash", "model_hash", "sha256", "tensor_hash")
        if name is not None:
            put(label, name, category)
        if model_hash is not None:
            put(f"{label} Hash", model_hash, category)
        for key in ("base", "type", "submodel_type", "key"):
            field_value = _get_norm(value, key)
            if field_value is not None and field_value != name:
                put(f"{label} · {key.replace('_', ' ').title()}", field_value, category)
        # Retain any additional model-identifier fields unknown to this parser.
        known = {_norm_key(x) for x in ("name", "model_name", "filename", "key", "hash", "model_hash", "sha256", "tensor_hash", "base", "type", "submodel_type")}
        for key, field_value in value.items():
            if _norm_key(key) not in known and not isinstance(field_value, (dict, list)):
                put(f"{label} · {key}", field_value, "Other Parameters")
    elif value is not None and str(value).strip():
        put(label, value, category)


def _parse_swarmui(data: dict, put, source_label: str = ""):
    """Parse SwarmUI's sui_image_params / sui_extra_data / sui_models envelope."""
    params = _get_norm(data, "sui_image_params")
    extra = _get_norm(data, "sui_extra_data")
    models = _get_norm(data, "sui_models")
    if params is None and extra is None and models is None:
        # A standalone sui_image_params object may be passed directly.
        if source_label and _norm_key(source_label) == "suiimageparams":
            params = data
        else:
            return False

    if isinstance(params, dict):
        _collect_known_from_dict(params, put)
        aliases = {
            "prompt", "positiveprompt", "negativeprompt", "seed", "steps", "cfg", "cfgscale",
            "guidancescale", "sampler", "samplername", "scheduler", "schedulername", "denoise",
            "denoisingstrength", "model", "modelname", "ckptname", "vae", "vaename", "width", "height",
            "upscaler", "upscalername", "clipskip", "clip_skip", "hiresupscaler", "hiressteps",
            "hiresupscale", "refiner", "refinermodel", "refinerswitchat", "swarmversion"
        }
        for key, value in params.items():
            if _norm_key(key) in aliases:
                continue
            if isinstance(value, (dict, list)):
                put(f"SwarmUI · {key}", value, "Other Parameters")
            elif value is not None:
                put(f"SwarmUI · {key}", value, "Other Parameters")
        # SwarmUI can store used LoRAs as a list or as a comma-delimited value.
        used_loras = _get_norm(params, "used_loras", "loras")
        if isinstance(used_loras, list):
            for index, item in enumerate(used_loras, start=1):
                if isinstance(item, dict):
                    name = _get_norm(item, "name", "model", "filename")
                    strength = _get_norm(item, "strength", "weight", "strength_model")
                    if name is not None:
                        put(f"LoRA {index} Name", name, "LoRAs")
                    if strength is not None:
                        put(f"LoRA {index} Strength", strength, "LoRAs")
                    clip_strength = _get_norm(item, "strength_clip", "clip_strength")
                    if clip_strength is not None:
                        put(f"LoRA {index} CLIP Strength", clip_strength, "LoRAs")
                else:
                    put(f"LoRA {index}", item, "LoRAs")
        elif used_loras is not None:
            put("Used LoRAs", used_loras, "LoRAs")

    if isinstance(extra, dict):
        for key, value in extra.items():
            normalized = _norm_key(key)
            if normalized in {"originalprompt", "originalpositiveprompt"}:
                put("Original Prompt", value, "Prompts")
            elif normalized in {"date", "generationdate", "timestamp"}:
                put("Generation Date", value, "Image Metadata")
            else:
                put(f"SwarmUI Extra · {key}", value, "Other Parameters")
    elif extra is not None:
        put("SwarmUI Extra Data", extra, "Other Parameters")

    if isinstance(models, list):
        lora_index = 0
        for index, model in enumerate(models, start=1):
            if not isinstance(model, dict):
                put(f"SwarmUI Model {index}", model, "Models")
                continue
            param = str(_get_norm(model, "param", "type", "model_type") or f"model {index}")
            name = _get_norm(model, "name", "model", "filename")
            model_hash = _get_norm(model, "hash", "model_hash", "tensor_hash")
            if "lora" in param.lower() or "lycoris" in param.lower() or "used_loras" in param.lower():
                lora_index += 1
                prefix = f"LoRA {lora_index}"
                if name is not None:
                    put(f"{prefix} Name", name, "LoRAs")
                if model_hash is not None:
                    put(f"{prefix} Hash", model_hash, "LoRAs")
            else:
                prefix = f"SwarmUI Model · {param}"
                if name is not None:
                    put(f"{prefix} Name", name, "Models")
                if model_hash is not None:
                    put(f"{prefix} Hash", model_hash, "Models")
            for key, value in model.items():
                if _norm_key(key) not in {"name", "model", "filename", "hash", "modelhash", "tensorhash", "param", "type", "modeltype"} and not isinstance(value, (dict, list)):
                    put(f"SwarmUI Model {index} · {key}", value, "Other Parameters")
    elif models is not None:
        put("SwarmUI Models", models, "Models")

    return True


def _parse_invokeai_metadata(data: dict, put):
    """Parse current InvokeAI core_metadata plus pre-1.0-style aliases."""
    _collect_known_from_dict(data, put)
    _parse_model_identifier(_get_norm(data, "model"), "Checkpoint", "Models", put)
    _parse_model_identifier(_get_norm(data, "vae"), "VAE", "Models", put)

    loras = _get_norm(data, "loras", "lora")
    if isinstance(loras, list):
        for index, item in enumerate(loras, start=1):
            if not isinstance(item, dict):
                put(f"LoRA {index}", item, "LoRAs")
                continue
            model = _get_norm(item, "model", "lora", "model_identifier")
            _parse_model_identifier(model, f"LoRA {index} Name", "LoRAs", put)
            weight = _get_norm(item, "weight", "strength", "strength_model")
            if weight is not None:
                put(f"LoRA {index} Strength", weight, "LoRAs")
            clip_weight = _get_norm(item, "clip_weight", "strength_clip", "clip_strength")
            if clip_weight is not None:
                put(f"LoRA {index} CLIP Strength", clip_weight, "LoRAs")

    # Preserve every remaining scalar in the flat generation record. This also
    # future-proofs the reader when InvokeAI adds optional metadata fields.
    known_roots = {
        "positiveprompt", "negativeprompt", "prompt", "seed", "steps", "cfgscale",
        "width", "height", "model", "vae", "loras"
    }
    for key, value in _iter_scalars(data):
        root = key.split(".")[0].split("[")[0]
        if _norm_key(root) in known_roots:
            continue
        put(f"InvokeAI · {key}", value, "Other Parameters")


def _parse_invokeai_graph(data: dict, put):
    """Expose graph JSON and individual scalar node properties when available."""
    nodes = data.get("nodes")
    if isinstance(nodes, dict):
        for node_id, node in nodes.items():
            if not isinstance(node, dict):
                continue
            node_type = _get_norm(node, "type", "id", "name") or "Node"
            for key, value in _iter_scalars(node):
                put(f"InvokeAI Graph · {node_id} · {node_type} · {key}", value, "Workflow Details")
    elif isinstance(nodes, list):
        for index, node in enumerate(nodes, start=1):
            if not isinstance(node, dict):
                continue
            node_type = _get_norm(node, "type", "id", "name") or "Node"
            for key, value in _iter_scalars(node):
                put(f"InvokeAI Graph · Node {index} · {node_type} · {key}", value, "Workflow Details")


def _is_comfyui_workflow_graph(value: Any) -> bool:
    """Return True only for a serialized ComfyUI canvas workflow, not an API prompt."""
    if not isinstance(value, dict):
        return False
    nodes = value.get("nodes")
    links = value.get("links")
    if not isinstance(nodes, list) or not isinstance(links, list):
        return False
    if not nodes:
        # An empty workflow is valid only when it still carries recognizable
        # ComfyUI graph-level fields; this avoids treating arbitrary JSON arrays
        # as an openable workflow.
        return any(key in value for key in ("last_node_id", "last_link_id", "version", "groups"))
    return any(
        isinstance(node, dict)
        and "id" in node
        and (isinstance(node.get("type"), str) or isinstance(node.get("class_type"), str))
        for node in nodes
    )


def _find_embedded_comfyui_workflow(info: dict, exif: dict) -> dict | None:
    """Find a valid ComfyUI canvas workflow in embedded metadata.

    Only named workflow fields are considered. API prompt JSON, A1111 parameter
    strings, InvokeAI graphs, and arbitrary nested JSON are deliberately not
    exposed as ComfyUI workflows.
    """
    workflow_keys = {"workflow", "comfyuiworkflow", "workflowjson"}
    containers = (info, exif)

    def parse_candidate(value: Any, depth: int = 0) -> dict | None:
        if depth > 4:
            return None
        parsed = _as_dict(value)
        if not isinstance(parsed, dict):
            return None
        if _is_comfyui_workflow_graph(parsed):
            return parsed
        # A few writers wrap the actual ComfyUI graph in an envelope.
        for key in ("workflow", "graph", "comfyui_workflow", "comfyuiWorkflow"):
            if key in parsed:
                result = parse_candidate(parsed[key], depth + 1)
                if result is not None:
                    return result
        return None

    def visit(mapping: dict, depth: int = 0) -> dict | None:
        if depth > 3:
            return None
        for key, value in mapping.items():
            if _norm_key(key) in workflow_keys:
                candidate = parse_candidate(value)
                if candidate is not None:
                    return candidate
            # Handle metadata envelopes such as {"comfyui": {"workflow": ...}}
            # without recursively interpreting unrelated prompt/model records.
            if isinstance(value, dict) and depth < 3:
                candidate = visit(value, depth + 1)
                if candidate is not None:
                    return candidate
            elif isinstance(value, str) and depth < 2 and value.lstrip().startswith("{"):
                nested = _as_dict(value)
                if nested is not None:
                    candidate = visit(nested, depth + 1)
                    if candidate is not None:
                        return candidate
        return None

    for container in containers:
        candidate = visit(container)
        if candidate is not None:
            return _json_safe(candidate)
    return None


def read_png_text_chunks(path: str) -> list[dict[str, str]]:
    """Return editable PNG text chunks, retaining duplicate keys and order.

    The editor intentionally works with PNG text chunks only. Image pixels,
    ICC profiles, EXIF, animation chunks, and every other binary chunk remain
    outside the editable surface and are preserved byte-for-byte on write.
    """
    signature = b"\x89PNG\r\n\x1a\n"
    entries: list[dict[str, str]] = []
    with open(path, "rb") as stream:
        if stream.read(8) != signature:
            raise ValueError("The selected file is not a valid PNG image.")
        while True:
            header = stream.read(8)
            if len(header) != 8:
                raise ValueError("The PNG file ended before its IEND chunk.")
            length, chunk_type = struct.unpack(">I4s", header)
            if chunk_type not in {b"tEXt", b"zTXt", b"iTXt"}:
                # Pixel/image chunks can be very large; skip them without
                # materializing their payload in memory.
                try:
                    stream.seek(length, 1)
                except OSError as exc:
                    raise ValueError("The PNG file contains an invalid chunk length.") from exc
                crc = stream.read(4)
                if len(crc) != 4:
                    raise ValueError("The PNG file contains a truncated chunk.")
                if chunk_type == b"IEND":
                    break
                continue
            if length > 64 * 1024 * 1024:
                raise ValueError("A PNG text metadata chunk is too large to inspect safely.")
            payload = stream.read(length)
            crc = stream.read(4)
            if len(payload) != length or len(crc) != 4:
                raise ValueError("The PNG file contains a truncated text chunk.")

            try:
                if chunk_type == b"tEXt":
                    keyword, text_value = payload.split(b"\x00", 1)
                    keyword_text = keyword.decode("latin-1")
                    value_text = text_value.decode("latin-1")
                    format_name = "tEXt"
                elif chunk_type == b"zTXt":
                    keyword, remainder = payload.split(b"\x00", 1)
                    if not remainder or remainder[0] != 0:
                        raise ValueError("Unsupported PNG zTXt compression method.")
                    keyword_text = keyword.decode("latin-1")
                    value_text = zlib.decompress(remainder[1:]).decode("latin-1")
                    format_name = "zTXt"
                else:
                    keyword, remainder = payload.split(b"\x00", 1)
                    if len(remainder) < 2:
                        raise ValueError("Malformed PNG iTXt chunk.")
                    compressed, method = remainder[0], remainder[1]
                    language, remainder = remainder[2:].split(b"\x00", 1)
                    translated, text_value = remainder.split(b"\x00", 1)
                    keyword_text = keyword.decode("latin-1")
                    if compressed:
                        if method != 0:
                            raise ValueError("Unsupported PNG iTXt compression method.")
                        text_value = zlib.decompress(text_value)
                    value_text = text_value.decode("utf-8")
                    format_name = "iTXt"
                entries.append({"keyword": keyword_text, "value": value_text, "format": format_name})
            except Exception as exc:
                raise ValueError(f"A PNG text metadata chunk could not be decoded: {exc}") from exc
            if chunk_type == b"IEND":
                break
    return entries


def _generation_data_payload(info: dict, embedded_workflow: dict | None = None) -> dict | None:
    """Expose a compact, execution-ready node-input map for the settings importer.

    Kiri metadata stores only active upstream nodes contributing to Image A.
    For other ComfyUI PNGs, the API ``prompt`` chunk is the best available
    executed-graph source. Canvas workflow JSON is used only to decorate nodes
    with readable titles/workflow identity; it is never used to guess inputs.
    """
    kiri_record = _as_dict(info.get("kiri_metadata")) or {}
    kiri_graph = kiri_record.get("graph") if isinstance(kiri_record.get("graph"), dict) else {}
    raw_nodes = kiri_graph.get("active_upstream_nodes") if isinstance(kiri_graph.get("active_upstream_nodes"), list) else []
    source = "Kiri structured graph" if raw_nodes else "ComfyUI executed prompt"

    if not raw_nodes:
        prompt = _as_dict(info.get("prompt")) or {}
        # Some encoders wrap arrays under `items`; normal API prompt metadata
        # is an object keyed by node ID.
        if isinstance(prompt.get("items"), list):
            prompt = {str(index): item for index, item in enumerate(prompt["items"]) if isinstance(item, dict)}
        for node_id, entry in prompt.items():
            if not isinstance(entry, dict) or not isinstance(entry.get("inputs"), dict):
                continue
            class_type = entry.get("class_type") or entry.get("type")
            if not class_type:
                continue
            raw_nodes.append({
                "id": str(node_id),
                "class_type": str(class_type),
                "inputs": _json_safe(entry.get("inputs", {})),
            })

    labels = {}
    workflow_id = None
    if isinstance(embedded_workflow, dict):
        workflow_id = embedded_workflow.get("id")
        for visual_node in embedded_workflow.get("nodes", []) if isinstance(embedded_workflow.get("nodes"), list) else []:
            if isinstance(visual_node, dict) and visual_node.get("id") is not None:
                labels[str(visual_node["id"])] = visual_node.get("title")

    nodes = []
    seen = set()
    for item in raw_nodes[:1200]:
        if not isinstance(item, dict):
            continue
        node_id = str(item.get("id", ""))
        class_type = str(item.get("class_type", "")).strip()
        inputs = item.get("inputs") if isinstance(item.get("inputs"), dict) else {}
        if not class_type or not isinstance(inputs, dict):
            continue
        identity = (node_id, class_type)
        if identity in seen:
            continue
        seen.add(identity)
        node = {
            "id": node_id,
            "class_type": class_type,
            "inputs": _json_safe(inputs),
        }
        title = item.get("title") or labels.get(node_id)
        if isinstance(title, str) and title.strip():
            node["title"] = title.strip()
        nodes.append(node)

    if not nodes:
        return None
    return {
        "schema_version": 1,
        "source": source,
        "workflow_id": workflow_id or kiri_record.get("source_workflow_id"),
        "nodes": nodes,
    }


def inspect_image(path: str) -> dict[str, Any]:
    """Read embedded image metadata and return dynamic, individually copyable fields."""
    copy_items: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def put(label: Any, value: Any, category: str = "Other"):
        label = str(label).strip()
        text_value = _text(value)
        if not label or not text_value.strip():
            return
        # Never add multi-kilobyte duplicate aliases as separate menu items.
        key = (label.casefold(), text_value)
        if key in seen:
            return
        seen.add(key)
        copy_items.append({"label": label, "value": text_value, "category": category})

    file_path = Path(path)
    with Image.open(path) as image:
        image_format = image.format or file_path.suffix.lstrip(".").upper()
        width, height = image.size
        mode = image.mode
        info = {str(k): _json_safe(v) for k, v in image.info.items()}
        exif = _read_exif(image)
        stealth_data = _extract_novelai_stealth(image)

    # For Kiri-generated PNGs, the structured Kiri record is the editable
    # source of truth for prompt text. The standard ComfyUI `prompt` graph is
    # retained as provenance, but may intentionally contain the original
    # workflow text after a metadata-only edit. Suppress only the stale prompt
    # aliases while continuing to parse its workflow graph and sampler data.
    kiri_authority = _as_dict(info.get("kiri_metadata"))
    kiri_graph = kiri_authority.get("graph", {}) if isinstance(kiri_authority, dict) else {}
    kiri_prompts = kiri_graph.get("prompts", {}) if isinstance(kiri_graph, dict) else {}
    authoritative_positive = isinstance(kiri_prompts, dict) and "positive" in kiri_prompts
    authoritative_negative = isinstance(kiri_prompts, dict) and "negative" in kiri_prompts

    def put_prompt_aware(label: Any, value: Any, category: str = "Other"):
        # Once a Kiri record supplies the authoritative prompt branches, do
        # not also expose per-node text-widget values from the retained graph
        # or stale A1111 Parameters string as prompt entries.
        if category == "Prompts" and (authoritative_positive or authoritative_negative):
            return
        put(label, value, category)

    embedded_workflow = _find_embedded_comfyui_workflow(info, exif)
    generation_data = _generation_data_payload(info, embedded_workflow)

    # Base image facts are also individually copyable.
    put("Filename", file_path.name, "Image")
    put("Format", image_format, "Image")
    put("Width", width, "Image")
    put("Height", height, "Image")
    put("Color Mode", mode, "Image")
    node_names_text = _get_norm(info, "kiri_nodes_used")
    if isinstance(node_names_text, str) and node_names_text.strip():
        put("Nodes Used", node_names_text, "ComfyUI Nodes")

    # Parse all common embedded text metadata fields.
    text_sources: list[tuple[str, str]] = []
    for key, value in info.items():
        if isinstance(value, (str, bytes)):
            decoded = _decode_metadata_text(value)
            if decoded.strip():
                text_sources.append((key, decoded))
    for key, value in exif.items():
        if isinstance(value, (str, bytes)):
            decoded = _decode_metadata_text(value)
            if decoded.strip():
                text_sources.append((key, decoded))

    source_names = []
    json_objects: list[tuple[str, dict]] = []

    # Generator/software tags identify legacy formats whose metadata records
    # do not carry a dedicated format key of their own.
    for identity_key in ("Software", "Creator", "Generator", "Application", "Producer", "Host", "Source"):
        identity_value = info.get(identity_key, exif.get(identity_key))
        if identity_value:
            identity_text = _decode_metadata_text(identity_value)
            lower_identity = identity_text.lower()
            for marker, display in (
                ("novelai", "NovelAI"), ("fooocus", "Fooocus / Fooocus-MRE"),
                ("easy diffusion", "Easy Diffusion"), ("draw things", "Draw Things / XMP"),
                ("naifu", "Naifu"), ("swarmui", "SwarmUI"), ("invokeai", "InvokeAI"),
                ("comfyui", "ComfyUI"), ("pixai", "PixAI"),
                ("tensor.art", "Tensor.Art"), ("seaart", "SeaArt"),
                ("civitai", "Civitai"), ("tensorhub", "TensorHub / Cozy Creator"),
            ):
                if marker in lower_identity:
                    source_names.append(display)

    for key, value in info.items():
        if isinstance(value, (dict, list)):
            json_objects.append((key, value if isinstance(value, dict) else {"items": value}))
            continue
        obj = _as_dict(value)
        if obj:
            json_objects.append((key, obj))

    if isinstance(stealth_data, dict):
        json_objects.append(("NovelAI Stealth Metadata", stealth_data))
        source_names.append("NovelAI Stealth PNG")

    fooocus_scheme = info.get("fooocus_scheme") or exif.get("MakerNote")
    parameter_object = _as_dict(info.get("parameters"))
    if parameter_object and "fooocus" in _decode_metadata_text(fooocus_scheme or "").lower():
        json_objects.append(("Fooocus Metadata", parameter_object))
        source_names.append("Fooocus / Fooocus-MRE")

    for key, text_value in text_sources:
        norm = _norm_key(key)
        if norm in {"parameters", "comment", "description", "usercomment", "prompt", "positiveprompt", "negativeprompt", "workflow", "invokeaimetadata", "invokeaiworkflow", "invokeaigraph", "sdmetadata", "suiimageparams", "suiextradata", "suimodels", "xmlcomadobexmp", "xmp", "dream", "software", "creator", "make", "model", "easydiffusiontext"}:
            source_names.append(key)
            if norm in {"xmlcomadobexmp", "xmp"} or text_value.lstrip().startswith("<"):
                if _parse_xmp_metadata(text_value, put):
                    source_names.append("Draw Things / XMP")
                else:
                    put(f"{key} (Text)", text_value, "Raw Metadata")
            elif norm in {"usercomment", "drawthingsusercomment"}:
                user_comment = _as_dict(text_value)
                if user_comment:
                    json_objects.append(("Draw Things UserComment", user_comment))
                    source_names.append("Draw Things")
                else:
                    put(f"{key} (Text)", text_value, "Image Metadata")
            elif norm == "dream":
                source_names.append("InvokeAI")
                dream_data = _as_dict(text_value)
                if dream_data:
                    json_objects.append(("InvokeAI Dream", dream_data))
                elif re.search(r"(?:^|\n)\s*(?:Steps|Negative prompt|Seed)\s*:", text_value, flags=re.IGNORECASE):
                    _parse_settings_string(text_value, put)
                else:
                    put("InvokeAI Dream Prompt", text_value, "Prompts")
            elif norm in {"parameters", "comment", "description", "usercomment", "suiimageparams"} and re.search(r"(?:^|\n)\s*(?:Steps|Negative prompt|Seed)\s*:", text_value, flags=re.IGNORECASE):
                _parse_settings_string(text_value, put_prompt_aware)
            elif norm == "easydiffusiontext":
                _parse_easy_diffusion_text(text_value, put_prompt_aware)
                put("Easy Diffusion Task Text", text_value, "Raw Metadata")
            elif norm == "parameters":
                maybe_json = _as_dict(text_value)
                if maybe_json:
                    json_objects.append((key, maybe_json))
                elif text_value.strip():
                    put("Positive Prompt", text_value, "Prompts")
            elif norm in {"comment", "description", "usercomment"}:
                maybe_json = _as_dict(text_value)
                if maybe_json:
                    json_objects.append((key, maybe_json))
                else:
                    put(f"{key} (Text)", text_value, "Raw Metadata")
            elif norm in {"prompt", "positiveprompt", "negativeprompt"}:
                maybe_json = _as_dict(text_value)
                if maybe_json:
                    json_objects.append((key, maybe_json))
                else:
                    label = "Negative Prompt" if norm == "negativeprompt" else "Positive Prompt"
                    put_prompt_aware(label, text_value, "Prompts")
            elif norm in {"workflow", "invokeaiworkflow", "invokeaigraph"}:
                maybe_json = _as_dict(text_value)
                label = "InvokeAI Workflow JSON" if norm == "invokeaiworkflow" else ("InvokeAI Graph JSON" if norm == "invokeaigraph" else "Workflow JSON")
                if maybe_json:
                    json_objects.append((key, maybe_json))
                    put(label, text_value, "Workflow")
                else:
                    put(label, text_value, "Workflow")
            else:
                maybe_json = _as_dict(text_value)
                if maybe_json:
                    json_objects.append((key, maybe_json))
                elif text_value.strip():
                    put(f"{key} (Text)", text_value, "Raw Metadata")

    # Handle EXIF UserComment/XPComment and common caption fields.
    for key in ("UserComment", "ImageDescription", "XPComment", "XPTitle", "DocumentName"):
        value = exif.get(key)
        if value:
            decoded = _decode_metadata_text(value)
            maybe_json = _as_dict(decoded)
            if maybe_json:
                # Draw Things JSON and Fooocus EXIF records are parameters, not
                # opaque comments. MakerNote is Fooocus' scheme discriminator.
                maker_note = _decode_metadata_text(exif.get("MakerNote", "")).lower()
                source_key = "Fooocus Metadata" if "fooocus" in maker_note else ("Draw Things Metadata" if key == "UserComment" else key)
                json_objects.append((source_key, maybe_json))
            elif re.search(r"(?:^|\n)\s*(?:Steps|Negative prompt|Seed)\s*:", decoded, flags=re.IGNORECASE):
                _parse_settings_string(decoded, put_prompt_aware)
            elif key not in {"UserComment"}:
                put(key, decoded, "Image Metadata")

    # Identify metadata dialect for the menu and scan structured formats.
    for source_key, data in json_objects:
        norm_source = _norm_key(source_key)
        source_names.append(source_key)

        if norm_source.startswith("novelai") or any(_norm_key(k) in {"v4prompt", "v4negativeprompt", "qualitytoggle", "smea", "smeadyn", "sm", "smdyn"} for k in data):
            source_names.append("NovelAI")
            _parse_novelai(data, put)
            put("NovelAI Metadata JSON", data, "Raw Metadata")

        elif norm_source in {"fooocus", "fooocusmetadata", "fooocusmre", "fooocushandoff"} or any(
            _norm_key(k).startswith("fooocus") or _norm_key(k) in {"fullprompt", "fullnegativeprompt", "basemodel", "loracombined1"}
            for k in data
        ):
            source_names.append("Fooocus / Fooocus-MRE")
            _collect_known_from_dict(data, put)
            for key, value in _iter_scalars(data):
                put(f"Fooocus · {key}", value, "Other Parameters")
            put("Fooocus Metadata JSON", data, "Raw Metadata")

        elif norm_source in {"drawthingsmetadata", "drawthingsusercomment", "drawthings"}:
            source_names.append("Draw Things")
            _collect_nested_platform_fields(data, put)
            for key, value in _iter_scalars(data):
                put(f"Draw Things · {key}", value, "Other Parameters")
            put("Draw Things Metadata JSON", data, "Raw Metadata")

        elif norm_source in {"naifu", "naifumetadata", "naifuhandoff"}:
            source_names.append("Naifu")
            _collect_known_from_dict(data, put)
            for key, value in _iter_scalars(data):
                put(f"Naifu · {key}", value, "Other Parameters")
            put("Naifu Metadata JSON", data, "Raw Metadata")

        elif norm_source in {"easydiffusion", "easydiffusionmetadata", "easydiffusiontask", "sdmetadata"} or any(
            _norm_key(k) == "numinferencesteps" for k in data
        ):
            source_names.append("Easy Diffusion")
            _collect_nested_platform_fields(data, put)
            for key, value in _iter_scalars(data):
                put(f"Easy Diffusion · {key}", value, "Other Parameters")
            put("Easy Diffusion Metadata JSON", data, "Raw Metadata")

        # A SwarmUI envelope can itself be nested in a Parameters/Comment JSON
        # object, or each sui_* section can be stored as its own PNG text chunk.
        if norm_source in {"suiimageparams", "suiextradata", "suimodels"} or any(
            _norm_key(k) in {"suiimageparams", "suiextradata", "suimodels"} for k in data
        ):
            if norm_source == "suiimageparams":
                _parse_swarmui({"sui_image_params": data}, put, source_key)
            elif norm_source == "suiextradata":
                _parse_swarmui({"sui_extra_data": data}, put, source_key)
            elif norm_source == "suimodels":
                _parse_swarmui({"sui_models": data.get("items", data) if isinstance(data, dict) else data}, put, source_key)
            else:
                _parse_swarmui(data, put, source_key)
            source_names.append("SwarmUI")
            # Also preserve the exact individual SwarmUI chunk as its own copy entry.
            put(f"SwarmUI {source_key} JSON", data, "Raw Metadata")

        elif norm_source in {"invokeaimetadata", "invokeaidream"}:
            source_names.append("InvokeAI")
            _parse_invokeai_metadata(data, put)
            put("InvokeAI Metadata JSON", data, "Raw Metadata")

        elif norm_source in {"pixaiapi", "tensorarttams", "seaarthandoff", "tensorhubcozycreatorhandoff", "civitaimeta"} or any(_norm_key(k) in {"modelversionid", "requestid", "stages", "civitairesources"} for k in data):
            display = {
                "pixaiapi": "PixAI API Handoff", "tensorarttams": "Tensor.Art TAMS Handoff",
                "seaarthandoff": "SeaArt Handoff", "tensorhubcozycreatorhandoff": "TensorHub / Cozy Creator Handoff",
                "civitaimeta": "Civitai Metadata",
            }.get(norm_source, source_key)
            source_names.append(display)
            _collect_nested_platform_fields(data, put)
            put(f"{display} JSON", data, "Raw Metadata")

        elif norm_source == "kirimetadata":
            source_names.append("ComfyUI / Kiri Graph Metadata")
            graph = data.get("graph", {}) if isinstance(data.get("graph"), dict) else {}
            # Prefer the collector's explicit positive/negative branch values.
            # They are resolved by following links behind the conditioning
            # sockets, including multiline string widgets. This authoritative
            # pair prevents generic node fields or KSampler settings from being
            # misidentified as prompt text by the clipboard menu.
            explicit_prompts = graph.get("prompts", {}) if isinstance(graph.get("prompts"), dict) else {}
            # Presence, not non-empty content, marks these branches as
            # authoritative. This lets the editor intentionally clear a prompt
            # without resurrecting stale text-node values retained in the graph.
            has_positive = "positive" in explicit_prompts
            has_negative = "negative" in explicit_prompts
            if has_positive and str(explicit_prompts.get("positive") or "").strip():
                put("Positive Prompt", explicit_prompts["positive"], "Prompts")
            if has_negative and str(explicit_prompts.get("negative") or "").strip():
                put("Negative Prompt", explicit_prompts["negative"], "Prompts")
            for text_node in graph.get("text_encoders_and_prompts", []) if isinstance(graph.get("text_encoders_and_prompts"), list) else []:
                branch = str(text_node.get("branch", "")).lower()
                if (branch == "positive" and has_positive) or (branch == "negative" and has_negative):
                    continue
                for field, value in (text_node.get("values", {}) or {}).items():
                    label = "Negative Prompt" if branch == "negative" else "Positive Prompt" if branch in {"positive", "shared"} else f"Prompt · {field}"
                    put(label, value, "Prompts")
            for sampler_index, sampler in enumerate(graph.get("samplers", []) if isinstance(graph.get("samplers"), list) else [], 1):
                for field, value in (sampler.get("settings", {}) or {}).items():
                    put(f"Sampler {sampler_index} · {field}", value, "Generation")
            for resource in graph.get("model_resources", []) if isinstance(graph.get("model_resources"), list) else []:
                resource_type = str(resource.get("type", "Model"))
                name = resource.get("name")
                if name:
                    category = "LoRAs" if resource_type == "LoRA" else "Models"
                    put(f"{resource_type} · {resource.get('input', 'name')}", name, category)
                if resource.get("sha256"):
                    put(f"{resource_type} SHA256 · {name or resource.get('node_id', '')}", resource["sha256"], "Models")
                if resource_type == "LoRA":
                    for field, value in resource.items():
                        if "strength" in str(field).lower():
                            put(f"LoRA · {name or resource.get('node_id', '')} · {field}", value, "LoRAs")
            custom = data.get("custom_metadata")
            if isinstance(custom, dict) and custom.get("value") is not None:
                put("Custom Metadata", custom.get("value"), "Other Parameters")
                put("Custom Metadata Source", custom.get("source", ""), "Other Parameters")
            for node in graph.get("active_upstream_nodes", []) if isinstance(graph.get("active_upstream_nodes"), list) else []:
                node_label = node.get("title") or node.get("class_type") or f"Node {node.get('id', '')}"
                for field, value in (node.get("inputs", {}) or {}).items():
                    if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
                        continue
                    if isinstance(value, (str, int, float, bool)) and str(value).strip():
                        put(f"Node · {node_label} · {field}", value, "ComfyUI Nodes")
                    elif isinstance(value, (dict, list)) and value:
                        put(f"Node · {node_label} · {field} JSON", value, "ComfyUI Nodes")
            put("Kiri Metadata JSON", data, "Workflow Details")
            if graph.get("active_upstream_nodes"):
                put("Active Upstream Nodes JSON", graph["active_upstream_nodes"], "Workflow Details")

        elif norm_source == "invokeaiworkflow":
            source_names.append("InvokeAI Workflow")
            put("InvokeAI Workflow JSON", data, "Workflow")

        elif norm_source == "invokeaigraph":
            source_names.append("InvokeAI Graph")
            _parse_invokeai_graph(data, put)
            put("InvokeAI Graph JSON", data, "Workflow")

        elif norm_source in {"kiriinterop", "kiriuniversalmetadata", "platformhandoff"}:
            source_names.append("Kiri Universal Metadata / Platform Handoff")
            _collect_nested_platform_fields(data, put)
            put("Kiri Universal Metadata JSON", data, "Raw Metadata")

        elif norm_source in {"meta", "civitai", "civitaimetadata", "civitaimeta"} or any(_norm_key(k) in {"civitairesources", "resources"} for k in data):
            source_names.append("Civitai / Generation Metadata")
            _collect_nested_platform_fields(data, put)
            put("Civitai Metadata JSON", data, "Raw Metadata")

        elif norm_source in {"pixai", "pixaiapi", "tensorart", "tams", "seaart", "tensorhub", "cozycreator"}:
            source_names.append({"pixai": "PixAI", "pixaiapi": "PixAI API", "tensorart": "Tensor.Art", "tams": "Tensor.Art TAMS", "seaart": "SeaArt", "tensorhub": "TensorHub", "cozycreator": "Cozy Creator"}.get(norm_source, source_key))
            _collect_nested_platform_fields(data, put)
            put(f"{source_key} Metadata JSON", data, "Raw Metadata")

        elif norm_source == "prompt" and ("nodes" not in data) and all(isinstance(v, dict) for v in data.values()):
            source_names.append("ComfyUI Prompt")
            _parse_comfy_graph(data, put_prompt_aware)
            put("ComfyUI Prompt JSON", data, "Workflow")
        elif "nodes" in data or any(isinstance(v, dict) and "class_type" in v for v in data.values()):
            source_names.append("ComfyUI Workflow/Prompt")
            _parse_comfy_graph(data, put)
            if norm_source == "workflow":
                put("Workflow JSON", data, "Workflow")
        else:
            # Handle envelopes nested under generic fields such as parameters or comment.
            if _parse_swarmui(data, put, source_key):
                source_names.append("SwarmUI")
            _collect_nested_platform_fields(data, put)
            # Expose any remaining scalar settings, including undocumented or
            # future metadata keys, without hiding them behind a predefined list.
            for key, value in _iter_scalars(data):
                normalized = _norm_key(key.split(".")[-1].split("[")[0])
                if normalized in {"prompt", "positiveprompt", "negativeprompt", "seed", "steps", "cfg", "cfgscale", "sampler", "samplername", "scheduler", "schedulername", "model", "modelname", "ckptname", "width", "height", "vae", "vaename"}:
                    continue
                put(key, value, "Other Parameters")

    # Legacy NovelAI images commonly carry the prompt in Description and use
    # a Software/Creator tag as the only generator identifier. Prefer an
    # explicit prompt from the structured Comment payload when it exists.
    if any("novelai" in name.lower() for name in source_names) and not any(
        item["label"] == "Positive Prompt" for item in copy_items
    ):
        description = info.get("Description") or info.get("description")
        if description:
            put("Positive Prompt", _decode_metadata_text(description), "Prompts")

    # Preserve useful top-level metadata keys which have not already been
    # interpreted. Raw workflow/prompt objects get their own explicit action.
    omit = {
        "prompt", "workflow", "parameters", "invokeai_metadata", "invokeai_workflow",
        "invokeai_graph", "kiri_metadata", "kiri_custom_metadata", "kiri_nodes_used", "kiri_metadata_mode",
        "exif", "icc_profile", "icc_profile_digest", "transparency"
    }
    for key, value in info.items():
        if _norm_key(key) in {_norm_key(x) for x in omit}:
            continue
        if isinstance(value, (str, int, float, bool)) and str(value).strip():
            put(f"Metadata · {key}", value, "Raw Metadata")
        elif isinstance(value, (dict, list)) and value:
            # Each unknown embedded chunk is its own copy action, never a
            # combined all-metadata action.
            put(f"Metadata · {key}", value, "Raw Metadata")

    for key, value in exif.items():
        if value is None or str(value).strip() == "":
            continue
        put(f"EXIF · {key}", value, "Image Metadata")

    # Give the user a stable source identifier when parsers can infer one.
    if any("swarmui" in name.lower() or _norm_key(name).startswith("sui") for name in source_names):
        source = "SwarmUI"
    elif any("novelai" in name.lower() for name in source_names):
        source = "NovelAI"
    elif any("draw things" in name.lower() or "xmp" in name.lower() for name in source_names):
        source = "Draw Things / XMP"
    elif any("fooocus" in name.lower() for name in source_names):
        source = "Fooocus / Fooocus-MRE"
    elif any("easy diffusion" in name.lower() for name in source_names):
        source = "Easy Diffusion"
    elif any("naifu" in name.lower() for name in source_names):
        source = "Naifu"
    elif any("invokeai" in name.lower() for name in source_names):
        source = "InvokeAI"
    elif any(_norm_key(name) == "parameters" for name in source_names):
        source = "A1111-compatible / PNG parameters"
    elif any("comfyui" in name.lower() or _norm_key(name) == "prompt" for name in source_names):
        source = "ComfyUI"
    elif source_names:
        source = ", ".join(dict.fromkeys(source_names))
    else:
        source = "Image metadata only" if (info or exif) else "No embedded generation metadata"

    # Sort by category group order, preserving detection order inside each group.
    order = ["Prompts", "Generation", "Models", "LoRAs", "Image", "Advanced", "ComfyUI Nodes", "Workflow", "Workflow Details", "Other Parameters", "Image Metadata", "Raw Metadata"]
    rank = {name: index for index, name in enumerate(order)}
    copy_items.sort(key=lambda item: rank.get(item["category"], len(order)))

    return {
        "ok": True,
        "filename": file_path.name,
        "format": image_format,
        "width": width,
        "height": height,
        "source": source,
        "has_metadata": bool(info or exif or stealth_data),
        "embedded_workflow": embedded_workflow,
        "generation_data": generation_data,
        "editable_metadata": read_png_text_chunks(str(file_path)) if image_format.upper() == "PNG" else None,
        "metadata_editable": image_format.upper() == "PNG",
        "copy_items": copy_items,
    }
