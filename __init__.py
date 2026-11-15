"""Kiri Image Loader - ComfyUI custom node."""
from __future__ import annotations

import base64
import io
import os
import json
import hashlib
import threading
import math
import re
import uuid
import stat
import struct
import tempfile
import zlib
from datetime import datetime, timezone
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, PngImagePlugin

import folder_paths

from .kiri_metadata import inspect_image
from .metadata_adapters import (
    DEFAULT_EXPORT_SETTINGS, build_easy_diffusion_request, build_export_chunks,
    build_interchange_manifest, build_sidecar, embed_novelai_stealth,
    normalize_export_settings,
)

NODE_NAME = "KiriImageLoader"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
_KIRI_EXTERNAL_OUTPUT_PATHS = {}
_KIRI_EXTERNAL_OUTPUT_LOCK = threading.RLock()


def _get_input_files():
    """Recursively list image files under ComfyUI's configured input root.

    The root comes from folder_paths.get_input_directory(), so installations
    using --input-directory on another drive are handled without assuming the
    portable ComfyUI folder or the user's Documents directory. Returned names
    stay relative to that root and preserve nested subfolders.
    """
    input_dir = os.path.realpath(folder_paths.get_input_directory())
    found = []
    try:
        for current, directories, filenames in os.walk(input_dir, followlinks=False):
            # Do not follow directory symlinks: they can escape the configured
            # input root or create recursive loops across mounted drives.
            directories[:] = sorted(
                directory for directory in directories
                if not os.path.islink(os.path.join(current, directory))
            )
            for filename in filenames:
                if Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                absolute = os.path.join(current, filename)
                if not os.path.isfile(absolute) or not _inside_input_directory(absolute):
                    continue
                relative = os.path.relpath(absolute, input_dir).replace(os.sep, "/")
                found.append(relative)
    except OSError as exc:
        print(f"[Kiri Image Loader] Could not scan input image folder {input_dir!r}: {exc}")
    return sorted(found, key=lambda value: value.casefold())


def _inside_input_directory(path: str) -> bool:
    """Return True only for a real path contained by ComfyUI's input directory."""
    root = os.path.realpath(folder_paths.get_input_directory())
    candidate = os.path.realpath(path)
    try:
        return os.path.commonpath((root, candidate)) == root
    except (ValueError, OSError):
        return False


def _resolve_managed_image_path(filename: str, file_type: str = "input", subfolder: str = "") -> str:
    """Resolve an image from a ComfyUI-managed input/output/temp directory.

    Paths are always interpreted relative to the selected ComfyUI folder root.
    Absolute paths, traversal, and symlink escapes are rejected. The `type`
    value comes from ComfyUI image records, never from an arbitrary filesystem path.
    """
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("No image filename was provided.")

    file_type = str(file_type or "input").lower()
    if file_type not in {"input", "output", "temp"}:
        raise ValueError(f"Unsupported ComfyUI image folder type: {file_type}")

    if file_type == "input" and not subfolder:
        # Preserve ComfyUI's annotated-filename support for normal image widgets.
        try:
            resolved = folder_paths.get_annotated_filepath(filename)
            if resolved and os.path.isfile(resolved) and _inside_input_directory(resolved):
                return os.path.realpath(resolved)
        except Exception:
            pass

    folder_getters = {
        "input": folder_paths.get_input_directory,
        "output": getattr(folder_paths, "get_output_directory", None),
        "temp": getattr(folder_paths, "get_temp_directory", None),
    }
    getter = folder_getters.get(file_type)
    if not callable(getter):
        raise FileNotFoundError(f"ComfyUI does not expose a {file_type} directory in this installation.")

    root = os.path.realpath(getter())
    relative_parts = [part for part in (subfolder, filename) if part]
    if any(os.path.isabs(part) or os.path.splitdrive(part)[0] for part in relative_parts):
        raise ValueError("Absolute image paths are not accepted.")
    candidate = os.path.realpath(os.path.join(root, *relative_parts))
    try:
        inside_root = os.path.commonpath((root, candidate)) == root
    except (ValueError, OSError):
        inside_root = False
    if not inside_root or not os.path.isfile(candidate):
        raise FileNotFoundError(f"Image not found inside ComfyUI's {file_type} directory: {filename}")
    if Path(candidate).suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError("The selected file is not a supported image format.")
    return candidate


def _resolve_image_path(filename: str) -> str:
    """Resolve an input-directory-relative filename and reject path escapes."""
    return _resolve_managed_image_path(filename, "input", "")


def _load_disk_image_and_mask(image_path: str):
    with Image.open(image_path) as source:
        img = ImageOps.exif_transpose(source)
        has_alpha = "A" in img.getbands()
        rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(rgb)[None, ...]

        if has_alpha:
            alpha = np.asarray(img.getchannel("A"), dtype=np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(alpha)
        else:
            # Match ComfyUI's built-in Load Image fallback mask shape.
            mask = torch.zeros((64, 64), dtype=torch.float32)

    return image_tensor, mask.unsqueeze(0)


def _launch_detached(arguments: list[str]):
    """Launch a desktop opener without tying it to the ComfyUI process console."""
    kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(arguments, **kwargs)




def _png_text_chunk(keyword: str, value: str) -> bytes:
    """Build one uncompressed UTF-8 PNG iTXt chunk."""
    keyword = str(keyword)
    value = str(value)
    try:
        keyword_bytes = keyword.encode("latin-1")
    except UnicodeEncodeError as exc:
        raise ValueError(f"PNG metadata key {keyword!r} must use Latin-1 characters.") from exc
    if not 1 <= len(keyword_bytes) <= 79:
        raise ValueError("PNG metadata keys must contain 1–79 Latin-1 bytes.")
    if keyword_bytes[:1] == b" " or keyword_bytes[-1:] == b" " or b"  " in keyword_bytes:
        raise ValueError("PNG metadata keys cannot start/end with spaces or contain consecutive spaces.")
    if any(byte < 32 or byte == 127 for byte in keyword_bytes):
        raise ValueError("PNG metadata keys cannot contain control characters.")
    value_bytes = value.encode("utf-8")
    # keyword NUL, compression flag 0, method 0, empty language NUL,
    # empty translated keyword NUL, then UTF-8 text.
    payload = keyword_bytes + b"\x00\x00\x00\x00\x00" + value_bytes
    chunk_type = b"iTXt"
    crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", crc)


def _coerce_metadata_editor_value(value, original):
    """Keep the original JSON scalar type when a structured field is edited."""
    text = str(value)
    if isinstance(original, bool):
        return text.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(original, int) and not isinstance(original, bool):
        try:
            return int(text.strip())
        except (TypeError, ValueError):
            return text
    if isinstance(original, float):
        try:
            return float(text.strip())
        except (TypeError, ValueError):
            return text
    return text


def _make_text_only_parameters(editor_values):
    """Build plain A1111-style text without JSON, graph IDs or platform envelopes."""
    lines = []
    positive = str(editor_values.get("positive_prompt", "")).strip()
    negative = str(editor_values.get("negative_prompt", "")).strip()
    if positive:
        lines.append(positive)
    if negative:
        lines.append("Negative prompt: " + negative)

    settings = []
    for field in editor_values.get("parameters", []) if isinstance(editor_values.get("parameters"), list) else []:
        if not isinstance(field, dict) or field.get("kind") not in {"sampler", "generic"}:
            continue
        value = str(field.get("value", "")).strip()
        label = str(field.get("label") or field.get("key") or "Parameter").strip()
        if value and label:
            settings.append(f"{label}: {value}")
    width = str(editor_values.get("image_width", "")).strip()
    height = str(editor_values.get("image_height", "")).strip()
    if width and height:
        settings.append(f"Size: {width}x{height}")
    if settings:
        lines.append(", ".join(settings))
    return "\n".join(lines)


def _apply_structured_editor_values(metadata_entries, editor_values):
    """Apply the friendly Prompt/Negative Prompt/Parameters editor to PNG chunks.

    Workflow-OFF PNGs cannot gain a ComfyUI-loadable graph through editing.
    Their parameter-only app/site JSON records may be preserved and updated,
    because these contain generation settings/resources rather than node data.
    """
    if not isinstance(editor_values, dict):
        return metadata_entries

    entries = [dict(entry) for entry in metadata_entries]
    by_key = {}
    for index, entry in enumerate(entries):
        # PNG text keywords are case-sensitive. Keep `Comment` distinct from
        # an unrelated lowercase `comment` field supplied by another tool.
        key = str(entry.get("keyword", ""))
        if key not in by_key:
            by_key[key] = index

    mode_index = next((index for key, index in by_key.items() if _metadata_key_token(key) == "kirimetadatamode"), None)
    mode_text = str(entries[mode_index].get("value", "")).casefold() if mode_index is not None else ""
    if "workflow" in mode_text and "off" in mode_text:
        # A workflow-OFF image must never gain ComfyUI graph metadata through
        # the editor. Keep the external app/site parameter chunks that were
        # intentionally exported, but reject Kiri/ComfyUI graph envelopes.
        allowed = {
            _metadata_key_token(key) for key in (
                "parameters", "Kiri Nodes Used", "Kiri Metadata Mode",
                "invokeai_metadata", "Description", "Comment", "Software",
                "Generation time", "drawthings_user_comment", "XML:com.adobe.xmp",
                "sd_metadata", "fooocus_scheme", "easy_diffusion_task",
                "easy_diffusion_text", "pixai_api", "tensor_art_tams",
                "seaart_handoff", "tensorhub_cozy_creator_handoff",
                "civitai_meta", "novelai_stealth", "fooocus_handoff", "naifu_handoff",
            )
        }
        safe_entries = []
        for entry in entries:
            original_key = str(entry.get("keyword", ""))
            token = _metadata_key_token(original_key)
            # PNG keywords are case-sensitive: only NovelAI's exact `Comment`
            # field is part of this export allow-list. A generic lowercase
            # `comment` must not slip through as an unrelated custom field.
            if token in allowed and (token != "comment" or original_key == "Comment"):
                safe_entries.append(entry)
        parameters_text = _make_text_only_parameters(editor_values)
        parameter_index = next((index for index, entry in enumerate(safe_entries)
                                if str(entry.get("keyword", "")).casefold() == "parameters"), None)
        if parameter_index is None:
            if parameters_text:
                safe_entries.insert(0, {"keyword": "parameters", "value": parameters_text})
        else:
            # Preserve a selected external application's native parameters
            # envelope, updating only its prompt fields. The payload is app
            # parameter data, never a ComfyUI node graph.
            original = str(safe_entries[parameter_index].get("value", ""))
            try:
                payload = json.loads(original)
            except Exception:
                payload = None
            if isinstance(payload, dict):
                positive = str(editor_values.get("positive_prompt", ""))
                negative = str(editor_values.get("negative_prompt", ""))
                def update_parameter_payload(obj):
                    if not isinstance(obj, dict):
                        return
                    if isinstance(obj.get("sui_image_params"), dict):
                        target = obj["sui_image_params"]
                        target["prompt"] = positive
                        target["negativeprompt"] = negative
                        extra = obj.get("sui_extra_data")
                        if isinstance(extra, dict):
                            extra["original_prompt"] = positive
                            extra["original_negativeprompt"] = negative
                    if isinstance(obj.get("reqBody"), dict):
                        obj["reqBody"]["prompt"] = positive
                        obj["reqBody"]["negative_prompt"] = negative
                    if "prompt" in obj:
                        obj["prompt"] = positive
                    if "negative_prompt" in obj:
                        obj["negative_prompt"] = negative
                    if "negativePrompt" in obj:
                        obj["negativePrompt"] = negative
                    if "full_prompt" in obj:
                        obj["full_prompt"] = positive
                    if "full_negative_prompt" in obj:
                        obj["full_negative_prompt"] = negative
                update_parameter_payload(payload)
                safe_entries[parameter_index]["value"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            else:
                safe_entries[parameter_index]["value"] = parameters_text

        # Refresh common platform-specific prompt fields without constructing
        # any Kiri or ComfyUI graph JSON. Preserve extra platform attributes.
        positive = str(editor_values.get("positive_prompt", ""))
        negative = str(editor_values.get("negative_prompt", ""))
        for entry in safe_entries:
            key = str(entry.get("keyword", "")).casefold()
            raw = str(entry.get("value", ""))
            try:
                payload = json.loads(raw)
            except Exception:
                payload = None
            if isinstance(payload, dict):
                if key == "invokeai_metadata":
                    payload["positive_prompt"] = positive
                    payload["negative_prompt"] = negative
                elif key in {"comment", "drawthings user comment"}:
                    if "prompt" in payload or key == "comment": payload["prompt"] = positive
                    if "uc" in payload or key == "comment": payload["uc"] = negative
                elif key in {"easy diffusion task", "sd metadata"}:
                    body = payload.get("reqBody") if isinstance(payload.get("reqBody"), dict) else payload
                    body["prompt"] = positive
                    body["negative_prompt"] = negative
                elif key in {"pixai api", "fooocus handoff", "naifu handoff", "seaart handoff", "tensorhub cozy creator handoff"}:
                    payload["prompt"] = positive
                    payload["negative_prompt"] = negative
                elif key == "tensor art tams":
                    for stage in payload.get("stages", []) if isinstance(payload.get("stages"), list) else []:
                        diffusion = stage.get("diffusion") if isinstance(stage, dict) else None
                        if isinstance(diffusion, dict):
                            diffusion["prompts"] = ([{"text": positive, "weight": 1.0}] if positive else [])
                            diffusion["negativePrompts"] = ([{"text": negative, "weight": 1.0}] if negative else [])
                elif key == "kiri_interop":
                    generation = payload.setdefault("generation", {})
                    if isinstance(generation, dict):
                        generation["positive_prompt"] = positive
                        generation["negative_prompt"] = negative
                    handoff = payload.get("platform_handoff")
                    if isinstance(handoff, dict):
                        for handoff_value in handoff.values():
                            if isinstance(handoff_value, dict):
                                handoff_value["prompt"] = positive
                                handoff_value["negative_prompt"] = negative
                entry["value"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return safe_entries

    kiri_index = by_key.get("kiri_metadata")
    kiri_record = None
    if kiri_index is not None:
        try:
            parsed = json.loads(entries[kiri_index].get("value", ""))
            if isinstance(parsed, dict):
                kiri_record = parsed
        except (TypeError, ValueError, json.JSONDecodeError):
            kiri_record = None

    if kiri_record is None:
        kiri_record = {"schema_version": 1, "collector": "Kiri Image Loader Metadata Editor", "graph": {}}
    graph = kiri_record.setdefault("graph", {})
    if not isinstance(graph, dict):
        graph = {}
        kiri_record["graph"] = graph
    prompts = graph.setdefault("prompts", {})
    if not isinstance(prompts, dict):
        prompts = {}
        graph["prompts"] = prompts
    prompts["positive"] = str(editor_values.get("positive_prompt", ""))
    prompts["negative"] = str(editor_values.get("negative_prompt", ""))

    samplers = graph.setdefault("samplers", [])
    if not isinstance(samplers, list):
        samplers = []
        graph["samplers"] = samplers
    for field in editor_values.get("parameters", []) if isinstance(editor_values.get("parameters"), list) else []:
        if not isinstance(field, dict) or field.get("kind") not in {"sampler", "generic"}:
            continue
        try:
            sampler_index = max(0, int(field.get("sampler_index", 0)))
        except (TypeError, ValueError):
            continue
        while len(samplers) <= sampler_index:
            samplers.append({"class_type": "KSampler", "settings": {}})
        sampler = samplers[sampler_index]
        if not isinstance(sampler, dict):
            sampler = {"class_type": "KSampler", "settings": {}}
            samplers[sampler_index] = sampler
        settings = sampler.setdefault("settings", {})
        if not isinstance(settings, dict):
            settings = {}
            sampler["settings"] = settings
        key = str(field.get("key", "")).strip()
        if field.get("kind") == "generic":
            aliases = {
                "seed": "seed", "steps": "steps", "cfg": "cfg", "cfg scale": "cfg",
                "sampler": "sampler_name", "scheduler": "scheduler", "denoise": "denoise",
            }
            key = aliases.get(key.lower(), re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_"))
        if not key:
            continue
        settings[key] = _coerce_metadata_editor_value(field.get("value", ""), settings.get(key))

    width = str(editor_values.get("image_width", "")).strip()
    height = str(editor_values.get("image_height", "")).strip()
    dimensions = graph.setdefault("image_dimensions", {})
    if not isinstance(dimensions, dict):
        dimensions = {}
        graph["image_dimensions"] = dimensions
    if width:
        dimensions["width"] = _coerce_metadata_editor_value(width, dimensions.get("width", 0))
    if height:
        dimensions["height"] = _coerce_metadata_editor_value(height, dimensions.get("height", 0))

    serialized_kiri = json.dumps(kiri_record, ensure_ascii=False, separators=(",", ":"))
    if kiri_index is None:
        entries.append({"keyword": "kiri_metadata", "value": serialized_kiri})
    else:
        entries[kiri_index]["value"] = serialized_kiri

    # Rebuild the widely recognized Parameters chunk from the edited structured
    # values. Keep the original graph JSON as provenance/history; it is not the
    # user-facing prompt editor and does not drive the copy-prompt actions.
    parameters_text = _make_a1111_parameters(graph)
    if width and height:
        size_line = f"Size: {width}x{height}"
        if parameters_text:
            if "Size:" not in parameters_text:
                parameters_text = parameters_text.rstrip() + ", " + size_line
        else:
            parameters_text = size_line
    parameters_index = by_key.get("parameters")
    if parameters_index is None:
        if parameters_text:
            entries.append({"keyword": "parameters", "value": parameters_text})
    else:
        original_parameters = str(entries[parameters_index].get("value", ""))
        try:
            swarm_existing = json.loads(original_parameters)
        except (TypeError, ValueError, json.JSONDecodeError):
            swarm_existing = None
        if isinstance(swarm_existing, dict) and isinstance(swarm_existing.get("sui_image_params"), dict):
            # Keep SwarmUI's native parameters envelope intact when users edit
            # Kiri's friendly editor. This field cannot simultaneously be A1111.
            try:
                from .metadata_adapters import build_swarmui_payload
                temp_record = {"graph": graph, "created_utc": kiri_record.get("created_utc")}
                refreshed = build_swarmui_payload(temp_record, int(width or 0), int(height or 0))
                entries[parameters_index]["value"] = json.dumps(refreshed, ensure_ascii=False, separators=(",", ":"))
            except Exception:
                entries[parameters_index]["value"] = parameters_text
        else:
            entries[parameters_index]["value"] = parameters_text

    # Refresh Kiri's optional native adapter chunks if they already exist. Do
    # not add formats to images that were not exported with those profiles.
    has_invoke = "invokeai_metadata" in by_key
    description_key = "Description" if "Description" in by_key else "description" if "description" in by_key else None
    comment_key = "Comment" if "Comment" in by_key else "comment" if "comment" in by_key else None
    has_novelai = description_key is not None
    if comment_key is not None:
        try:
            existing_comment = json.loads(entries[by_key[comment_key]].get("value", "{}"))
            has_novelai = has_novelai or (
                isinstance(existing_comment, dict)
                and any(key in existing_comment for key in ("prompt", "uc", "seed", "steps", "scale", "sampler"))
            )
        except Exception:
            pass
    has_interop = "kiri_interop" in by_key
    if width and height and (has_invoke or has_novelai or has_interop):
        try:
            from .metadata_adapters import (build_invokeai_metadata, build_novelai_comment, build_interchange_manifest)
            temp_record = {"graph": graph, "created_utc": kiri_record.get("created_utc")}
            if has_invoke:
                idx = by_key["invokeai_metadata"]
                try:
                    old = json.loads(entries[idx].get("value", "{}"))
                except Exception:
                    old = {}
                fresh = build_invokeai_metadata(temp_record, int(width), int(height))
                if isinstance(old, dict):
                    old.update(fresh)
                    fresh = old
                entries[idx]["value"] = json.dumps(fresh, ensure_ascii=False, separators=(",", ":"))
            if has_novelai:
                description, comment = build_novelai_comment(temp_record)
                if description_key is not None:
                    entries[by_key[description_key]]["value"] = description
                elif description:
                    entries.append({"keyword": "Description", "value": description})
                if comment_key is not None:
                    idx = by_key[comment_key]
                    try:
                        old_comment = json.loads(entries[idx].get("value", "{}"))
                    except Exception:
                        old_comment = {}
                    if isinstance(old_comment, dict):
                        old_comment.update(comment)
                        comment = old_comment
                    entries[idx]["value"] = json.dumps(comment, ensure_ascii=False, separators=(",", ":"))
            if has_interop:
                entries[by_key["kiri_interop"]]["value"] = json.dumps(
                    build_interchange_manifest(temp_record, int(width), int(height)),
                    ensure_ascii=False, separators=(",", ":")
                )
        except Exception:
            # A failure in an optional adapter must never block editing the
            # canonical Kiri metadata or the original PNG.
            pass
    return entries


def _rewrite_png_text_metadata(path: str, metadata_entries, editor_values=None) -> int:
    """Replace PNG text chunks atomically while preserving image/binary chunks.

    IDAT and all non-text chunks are streamed verbatim into a same-directory
    temporary file. This updates the original file in place at the filesystem
    level, but never recompresses or changes its pixel data.
    """
    signature = b"\x89PNG\r\n\x1a\n"
    if not str(path).lower().endswith(".png"):
        raise ValueError("Internal metadata editing currently supports PNG images only.")
    if not isinstance(metadata_entries, list) or len(metadata_entries) > 512:
        raise ValueError("Metadata must be a list containing no more than 512 fields.")

    metadata_entries = _apply_structured_editor_values(metadata_entries, editor_values)
    new_chunks = []
    total_metadata_bytes = 0
    for index, entry in enumerate(metadata_entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Metadata field {index + 1} is not a valid object.")
        keyword = entry.get("keyword", "")
        value = entry.get("value", "")
        if not isinstance(keyword, str) or not isinstance(value, str):
            raise ValueError(f"Metadata field {index + 1} must have text key and value fields.")
        chunk = _png_text_chunk(keyword, value)
        total_metadata_bytes += len(chunk)
        if total_metadata_bytes > 32 * 1024 * 1024:
            raise ValueError("Edited PNG text metadata exceeds the 32 MiB safety limit.")
        new_chunks.append(chunk)

    target = os.path.realpath(path)
    directory = os.path.dirname(target) or os.curdir
    original_mode = stat.S_IMODE(os.stat(target).st_mode)
    fd, temporary = tempfile.mkstemp(prefix=".kiri-meta-", suffix=".tmp", dir=directory)
    inserted = False
    saw_iend = False
    try:
        with open(target, "rb") as source, os.fdopen(fd, "wb") as output:
            if source.read(8) != signature:
                raise ValueError("The selected file is not a valid PNG image.")
            output.write(signature)
            while True:
                header = source.read(8)
                if len(header) != 8:
                    raise ValueError("The PNG file ended before its IEND chunk.")
                length, chunk_type = struct.unpack(">I4s", header)
                source_size = os.fstat(source.fileno()).st_size
                if length > source_size:
                    raise ValueError("The PNG file contains an invalid chunk length.")

                if chunk_type in {b"tEXt", b"zTXt", b"iTXt"}:
                    if length > 64 * 1024 * 1024:
                        raise ValueError("An existing PNG text chunk is too large to rewrite safely.")
                    discarded = source.read(length + 4)
                    if len(discarded) != length + 4:
                        raise ValueError("The PNG file contains a truncated text chunk.")
                    if not inserted:
                        for chunk in new_chunks:
                            output.write(chunk)
                        inserted = True
                    continue

                # Place text metadata before the first IDAT chunk. Although PNG
                # permits ancillary text chunks after image data, a number of
                # readers (including Pillow's eager info parser) stop collecting
                # metadata once IDAT begins. Pre-IDAT placement maximizes
                # interoperability for freshly edited PNGs with no prior text.
                if chunk_type == b"IDAT" and not inserted:
                    for chunk in new_chunks:
                        output.write(chunk)
                    inserted = True

                if chunk_type == b"IEND" and not inserted:
                    for chunk in new_chunks:
                        output.write(chunk)
                    inserted = True

                output.write(header)
                remaining = length + 4  # data plus CRC
                while remaining:
                    block = source.read(min(1024 * 1024, remaining))
                    if not block:
                        raise ValueError("The PNG file contains a truncated chunk payload.")
                    output.write(block)
                    remaining -= len(block)
                if chunk_type == b"IEND":
                    saw_iend = True
                    # PNG permits no chunks after IEND; preserve any trailing bytes
                    # rather than silently truncating a nonstandard container.
                    trailing = source.read()
                    if trailing:
                        output.write(trailing)
                    break
            if not saw_iend:
                raise ValueError("The selected PNG does not contain a valid IEND chunk.")
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, original_mode)
        os.replace(temporary, target)
        # Keep an explicitly-created .kiri.json companion synchronized with
        # edits to its PNG. Sidecar failure must not roll back a valid PNG edit.
        sidecar_path = os.path.join(directory, Path(target).stem + ".kiri.json")
        if os.path.isfile(sidecar_path):
            try:
                values_by_key = {str(entry.get("keyword", "")).lower(): entry.get("value", "") for entry in metadata_entries if isinstance(entry, dict)}
                sidecar_data = {}
                with open(sidecar_path, "r", encoding="utf-8") as sidecar_handle:
                    parsed_sidecar = json.load(sidecar_handle)
                    if isinstance(parsed_sidecar, dict):
                        sidecar_data = parsed_sidecar
                if values_by_key.get("kiri_metadata"):
                    sidecar_data["kiri_metadata"] = json.loads(values_by_key["kiri_metadata"])
                if values_by_key.get("kiri_interop"):
                    sidecar_data["adapters"] = json.loads(values_by_key["kiri_interop"])
                sidecar_data["edited_utc"] = datetime.now(timezone.utc).isoformat()
                sidecar_temp = sidecar_path + ".tmp"
                with open(sidecar_temp, "w", encoding="utf-8") as sidecar_handle:
                    json.dump(sidecar_data, sidecar_handle, ensure_ascii=False, indent=2)
                os.replace(sidecar_temp, sidecar_path)
            except Exception as sidecar_error:
                print(f"[Kiri Image Loader] PNG metadata was edited, but its optional .kiri.json sidecar could not be synchronized: {sidecar_error}")
        return len(new_chunks)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise

def _open_in_default_viewer(image_path: str):
    if os.name == "nt":
        os.startfile(image_path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        _launch_detached(["open", image_path])
    else:
        _launch_detached(["xdg-open", image_path])


def _windows_explorer_select_parameters(image_path: str) -> str:
    """Return Explorer's /select argument with the complete path quoted.

    Explorer has its own command-line parsing rules. Passing an unquoted path
    containing spaces commonly makes it open a default shell folder instead
    of revealing the selected image.
    """
    return f'/select,"{os.path.normpath(image_path)}"'


def _reveal_in_file_manager(image_path: str):
    if os.name == "nt":
        # ShellExecuteW avoids Python's argv quoting layer and passes Explorer's
        # required /select,"full path" form intact, including alternate drives.
        import ctypes
        result = ctypes.windll.shell32.ShellExecuteW(
            None, "open", "explorer.exe",
            _windows_explorer_select_parameters(image_path),
            None, 1,
        )
        if result <= 32:
            raise OSError(f"Windows Explorer could not reveal the image (ShellExecuteW={result}).")
    elif sys.platform == "darwin":
        _launch_detached(["open", "-R", image_path])
    else:
        # Linux file managers do not share a universal "select file" CLI.
        # Opening the containing directory is the portable desktop behavior.
        _launch_detached(["xdg-open", os.path.dirname(image_path)])


def _move_to_trash(image_path: str):
    """Move an image to the OS trash/recycle bin; never permanently delete it."""
    try:
        from send2trash import send2trash
    except ImportError:
        send2trash = None

    if send2trash is not None:
        # send2trash uses the platform's native trash/recycle-bin mechanism.
        # Any failure is propagated; there is deliberately no permanent-delete fallback.
        send2trash(image_path)
        return

    if os.name == "nt":
        # Windows-native fallback for portable ComfyUI installs that have not
        # installed the optional send2trash package yet. FOF_ALLOWUNDO routes
        # the file to the Recycle Bin. The source path must be double-NUL ended.
        import ctypes
        from ctypes import wintypes

        class SHFILEOPSTRUCTW(ctypes.Structure):
            _fields_ = [
                ("hwnd", wintypes.HWND),
                ("wFunc", wintypes.UINT),
                ("pFrom", wintypes.LPCWSTR),
                ("pTo", wintypes.LPCWSTR),
                ("fFlags", ctypes.c_ushort),
                ("fAnyOperationsAborted", wintypes.BOOL),
                ("hNameMappings", ctypes.c_void_p),
                ("lpszProgressTitle", wintypes.LPCWSTR),
            ]

        from_buffer = ctypes.create_unicode_buffer(image_path + "\0\0")
        operation = SHFILEOPSTRUCTW()
        operation.hwnd = None
        operation.wFunc = 3  # FO_DELETE
        operation.pFrom = ctypes.cast(from_buffer, wintypes.LPCWSTR)
        operation.pTo = None
        operation.fFlags = 0x0040 | 0x0010 | 0x0004 | 0x0400  # ALLOWUNDO, no confirm/UI
        operation.fAnyOperationsAborted = False
        operation.hNameMappings = None
        operation.lpszProgressTitle = None
        result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(operation))
        if result != 0 or operation.fAnyOperationsAborted:
            raise OSError(result or "Recycle Bin operation was cancelled by Windows.")
        return

    raise RuntimeError(
        "Moving images to Trash requires the 'send2trash' package on this platform. "
        "Install it in the Python environment used by ComfyUI with: "
        "python -m pip install send2trash"
    )


def _image_value_is_usable(value) -> bool:
    """Return False for missing or zero-sized IMAGE/MASK values.

    A genuinely black image is still valid. Only absent values and tensors/arrays
    with no elements or a zero-sized dimension are treated as empty.
    """
    if value is None:
        return False
    try:
        if hasattr(value, "numel") and callable(value.numel):
            if int(value.numel()) <= 0:
                return False
        shape = getattr(value, "shape", None)
        if shape is not None:
            dimensions = tuple(int(dimension) for dimension in shape)
            if not dimensions or any(dimension <= 0 for dimension in dimensions):
                return False
            return True
        if isinstance(value, (list, tuple, bytes, bytearray)):
            return len(value) > 0
        return True
    except Exception:
        return False


def _reference_image_dimensions(value, default=(64, 64)):
    """Infer (batch, height, width, device) from a non-empty IMAGE/MASK value."""
    batch, height, width = 1, int(default[0]), int(default[1])
    device = None
    try:
        shape = tuple(int(dimension) for dimension in value.shape)
        device = getattr(value, "device", None)
        if len(shape) == 4:
            batch = max(1, shape[0])
            if shape[-1] in (1, 3, 4):
                height, width = shape[1], shape[2]
            elif shape[1] in (1, 3, 4):
                height, width = shape[2], shape[3]
        elif len(shape) == 3:
            batch = max(1, shape[0])
            if shape[-1] in (3, 4) and shape[0] > 4 and shape[1] > 4:
                batch, height, width = 1, shape[0], shape[1]
            else:
                height, width = shape[1], shape[2]
        elif len(shape) == 2:
            height, width = shape
        if height <= 0 or width <= 0:
            height, width = int(default[0]), int(default[1])
    except Exception:
        pass
    return batch, height, width, device


def _neutral_image_like(reference=None):
    """Create a valid black IMAGE fallback when neither input nor file exists."""
    batch, height, width, device = _reference_image_dimensions(reference)
    kwargs = {"dtype": torch.float32}
    if device is not None:
        kwargs["device"] = device
    return torch.zeros((batch, height, width, 3), **kwargs)


def _neutral_mask_like(reference=None):
    """Create an all-zero MASK (no masked region) as a last-resort fallback."""
    batch, height, width, device = _reference_image_dimensions(reference)
    kwargs = {"dtype": torch.float32}
    if device is not None:
        kwargs["device"] = device
    return torch.zeros((batch, height, width), **kwargs)


def _workflow_input_link(prompt, extra_pnginfo, unique_id, input_name):
    """Return (is_linked, output_type) for a node input when graph data exposes it.

    Execution prompts are authoritative for whether a socket was linked. The
    visual workflow supplies the upstream socket type, which lets an empty
    IMAGE output receive an IMAGE placeholder even when it feeds the B socket
    (and vice versa).
    """
    node_id = str(unique_id) if unique_id is not None else ""
    linked = False
    node_prompt = None
    if isinstance(prompt, dict):
        node_prompt = prompt.get(node_id)
        if node_prompt is None:
            try:
                node_prompt = prompt.get(int(node_id))
            except (TypeError, ValueError):
                pass
    prompt_inputs = node_prompt.get("inputs", {}) if isinstance(node_prompt, dict) else {}
    value = prompt_inputs.get(input_name) if isinstance(prompt_inputs, dict) else None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        linked = True

    workflow = extra_pnginfo.get("workflow") if isinstance(extra_pnginfo, dict) else None
    if not isinstance(workflow, dict):
        return linked, None

    nodes = workflow.get("nodes", [])
    links = workflow.get("links", [])
    node_record = next((item for item in nodes if isinstance(item, dict) and str(item.get("id")) == node_id), None) if isinstance(nodes, list) else None
    target_slot = None
    if node_record:
        for index, socket in enumerate(node_record.get("inputs", []) or []):
            if isinstance(socket, dict) and socket.get("name") == input_name:
                slot_value = socket.get("slot_index")
                target_slot = index if slot_value is None else slot_value
                linked = linked or bool(socket.get("link") is not None or socket.get("links"))
                break

    output_type = None
    if target_slot is not None and isinstance(links, list):
        for link in links:
            if isinstance(link, dict):
                target_id = link.get("target_id", link.get("targetId"))
                slot = link.get("target_slot", link.get("targetSlot"))
                kind = link.get("type", link.get("data_type"))
            elif isinstance(link, (list, tuple)) and len(link) >= 6:
                target_id, slot, kind = link[3], link[4], link[5]
            else:
                continue
            if str(target_id) == node_id and str(slot) == str(target_slot):
                linked = True
                output_type = str(kind or "").upper() or None
                break
    return linked, output_type


def _empty_tensor_kind(value):
    """Infer IMAGE vs MASK from an empty tensor's layout when graph type is ambiguous."""
    try:
        shape = tuple(int(dimension) for dimension in value.shape)
    except Exception:
        return None
    if len(shape) == 2:
        return "MASK"
    if len(shape) == 3:
        # HWC image (or CHW image) versus ComfyUI's [B,H,W] MASK layout.
        if shape[-1] in (3, 4) and shape[0] > 4 and shape[1] > 4:
            return "IMAGE"
        if shape[0] in (1, 3, 4) and shape[1] > 4 and shape[2] > 4:
            return "IMAGE"
        return "MASK"
    if len(shape) == 4:
        if shape[-1] in (1, 3, 4) or shape[1] in (1, 3, 4):
            return "IMAGE"
    return None


def _value_has_spatial_dimensions(value) -> bool:
    """Whether even an empty IMAGE/MASK tensor carries usable H/W dimensions."""
    try:
        _batch, height, width, _device = _reference_image_dimensions(value, default=(0, 0))
        return height > 0 and width > 0
    except Exception:
        return False


def _neutral_for_input_type(output_type, reference=None, default_kind="IMAGE", empty_value=None):
    """Return a zero-valued placeholder with the linked socket's IMAGE/MASK type."""
    kind = str(output_type or "").upper().replace(" ", "")
    if kind not in {"IMAGE", "MASK"}:
        kind = _empty_tensor_kind(empty_value) or str(default_kind).upper()
    if kind == "MASK":
        return _neutral_mask_like(reference)
    return _neutral_image_like(reference)


def _tensor_to_preview_image(value):
    """Convert one IMAGE or MASK tensor sample into a small-display PIL image."""
    if not _image_value_is_usable(value):
        raise ValueError("Cannot render an empty IMAGE/MASK value (no samples or zero-sized dimensions).")
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    array = value.detach().to(device="cpu", dtype=torch.float32).numpy()

    # ComfyUI IMAGE is normally [B,H,W,C]; MASK is normally [B,H,W].
    is_image = False
    if array.ndim == 4:
        if array.shape[-1] in (1, 3, 4):
            array = array[0]
            is_image = array.shape[-1] in (3, 4)
        elif array.shape[1] in (1, 3, 4):
            # Tolerate channel-first image tensors from compatible custom nodes.
            array = np.moveaxis(array[0], 0, -1)
            is_image = array.shape[-1] in (3, 4)
        else:
            array = array[0]
    elif array.ndim == 3:
        # ComfyUI MASK batches are [B,H,W]. A few custom nodes provide a
        # single HWC image without a batch dimension, which is also accepted.
        if array.shape[-1] in (3, 4) and array.shape[0] > 4 and array.shape[1] > 4:
            is_image = True
        else:
            array = array[0]
    elif array.ndim != 2:
        raise ValueError(f"Unsupported preview tensor shape: {tuple(array.shape)}")

    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    array = np.clip(array, 0.0, 1.0)

    if is_image:
        if array.ndim != 3:
            raise ValueError("Image preview tensor must have three dimensions after batch selection.")
        channels = array.shape[-1]
        if channels == 1:
            pixels = np.rint(array[..., 0] * 255.0).astype(np.uint8)
            return Image.fromarray(pixels, mode="L").convert("RGB"), "IMAGE"
        pixels = np.rint(array * 255.0).astype(np.uint8)
        if channels == 4:
            return Image.fromarray(pixels, mode="RGBA"), "IMAGE"
        return Image.fromarray(pixels, mode="RGB"), "IMAGE"

    # Masks render as grayscale for inspection; the actual output tensor is
    # never altered by this preview conversion.
    if array.ndim == 3:
        if array.shape[-1] == 1:
            array = array[..., 0]
        elif array.shape[0] in (1, 3, 4):
            array = array[0]
        else:
            array = array[..., 0]
    pixels = np.rint(array * 255.0).astype(np.uint8)
    return Image.fromarray(pixels, mode="L").convert("RGB"), "MASK"


def _encode_live_preview(value):
    """Encode an input tensor as an inline JPEG for ComfyUI's executed UI payload.

    This deliberately mirrors KJNodes Points Editor: the backend receives the
    upstream tensor, converts its first sample to an image, Base64-encodes it,
    and returns it in the node's ``ui`` response. The frontend can therefore
    display it directly from ``onExecuted`` without a temp file or /view fetch.
    """
    preview, detected_type = _tensor_to_preview_image(value)
    preview = preview.convert("RGB")
    # Keep the preview payload bounded while retaining enough detail for the
    # node's compact display. This never changes the tensor returned downstream.
    max_side = 1600
    if max(preview.size) > max_side:
        scale = max_side / max(preview.size)
        target = (
            max(1, round(preview.width * scale)),
            max(1, round(preview.height * scale)),
        )
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        preview = preview.resize(target, resampling)

    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=82, optimize=True)
    return {
        "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "mime": "image/jpeg",
        "kind": detected_type,
    }


def _create_live_preview_payload(image_a, image_b):
    """Build KJNodes-style one-value UI fields for each inline preview image.

    ComfyUI unwraps one-element UI lists into properties on the node's
    ``onExecuted(message)`` object. This is the same transport used by
    KJNodes Points Editor's ``ui: {"bg_image": [base64_string]}`` response.
    """
    ui_payload = {}
    encoded_a = False
    encoded_b = False
    if _image_value_is_usable(image_a):
        try:
            preview_a = _encode_live_preview(image_a)
            ui_payload["kiri_preview_a"] = [preview_a["data"]]
            ui_payload["kiri_preview_a_kind"] = [preview_a["kind"]]
            encoded_a = True
        except Exception as exc:
            ui_payload["kiri_preview_a_error"] = [str(exc)]
            print(f"[Kiri Image Loader] Could not encode Image A preview: {exc}")
    if _image_value_is_usable(image_b):
        try:
            preview_b = _encode_live_preview(image_b)
            ui_payload["kiri_preview_b"] = [preview_b["data"]]
            ui_payload["kiri_preview_b_kind"] = [preview_b["kind"]]
            encoded_b = True
        except Exception as exc:
            ui_payload["kiri_preview_b_error"] = [str(exc)]
            print(f"[Kiri Image Loader] Could not encode Image B preview: {exc}")

    mode = "compare" if encoded_a and encoded_b else ("single" if encoded_a or encoded_b else "file")
    ui_payload["kiri_preview_mode"] = [mode]
    return ui_payload


# Model hashes are expensive for multi-gigabyte checkpoints. Cache them by
# resolved path + size + mtime for this ComfyUI process, so unchanged files are
# never re-hashed on every generation.
_MODEL_HASH_CACHE: dict[tuple[str, int, int], str] = {}
_MODEL_HASH_LOCK = threading.Lock()
_SECRET_KEY_FRAGMENTS = (
    "apikey", "token", "secret", "password", "passwd", "authorization",
    "credential", "cookie", "bearer", "privatekey",
)

# ComfyUI treats these PNG text chunks as graph/execution payloads. The
# Embed Complete Workflow switch controls BOTH the visual canvas graph and the
# executed API prompt; leaving `prompt` behind can still make an image behave
# like a workflow when dragged back into ComfyUI.
_COMFYUI_WORKFLOW_METADATA_KEYS = frozenset({
    "workflow", "comfyuiworkflow", "workflowjson", "comfyuiworkflowjson",
})
_COMFYUI_PROMPT_METADATA_KEYS = frozenset({
    "prompt", "comfyuiprompt", "apiprompt", "promptjson", "comfyuiexecutionprompt",
})
_COMFYUI_GRAPH_METADATA_KEYS = _COMFYUI_WORKFLOW_METADATA_KEYS | _COMFYUI_PROMPT_METADATA_KEYS


def _metadata_key_token(value) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def _looks_like_comfyui_api_prompt(value) -> bool:
    """Recognize a ComfyUI API prompt object without mistaking prompt text for a graph."""
    if not isinstance(value, dict) or not value:
        return False
    return any(
        isinstance(entry, dict)
        and isinstance(entry.get("inputs"), dict)
        and isinstance(entry.get("class_type") or entry.get("type"), str)
        for entry in value.values()
    )


def _strip_workflow_metadata_fields(value, depth: int = 0):
    """Remove named workflow payloads from nested metadata while preserving settings."""
    if depth > 24:
        return "<maximum metadata nesting depth reached>"
    if isinstance(value, dict):
        cleaned = {}
        for key, child in value.items():
            token = _metadata_key_token(key)
            if token in _COMFYUI_WORKFLOW_METADATA_KEYS:
                continue
            if token in _COMFYUI_PROMPT_METADATA_KEYS and _looks_like_comfyui_api_prompt(child):
                continue
            cleaned[str(key)] = _strip_workflow_metadata_fields(child, depth + 1)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [_strip_workflow_metadata_fields(child, depth + 1) for child in value]
    return value


def _json_safe_metadata(value, depth: int = 0, key: str = ""):
    """Convert runtime values into safe, size-bounded JSON metadata.

    Tensor payloads and arbitrary Python objects are summarized instead of
    serializing their contents. Credential-like fields are redacted even when
    the user elects to embed the complete workflow.
    """
    normalized_key = "".join(ch.lower() for ch in str(key) if ch.isalnum())
    if any(fragment in normalized_key for fragment in _SECRET_KEY_FRAGMENTS):
        return "[REDACTED]"
    if depth > 24:
        return "<maximum metadata nesting depth reached>"
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, str) and len(value) > 2_000_000:
            return value[:2_000_000] + "… <truncated>"
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return {"type": "bytes", "length": len(value), "hex_preview": value[:256].hex()}
    if isinstance(value, torch.Tensor):
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, np.generic):
        return _json_safe_metadata(value.item(), depth + 1, key)
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): _json_safe_metadata(v, depth + 1, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_metadata(v, depth + 1, key) for v in value]
    if isinstance(value, set):
        return [_json_safe_metadata(v, depth + 1, key) for v in sorted(value, key=str)]
    if isinstance(value, Path):
        return str(value)
    # Preserve ordinary data attributes from custom metadata containers while
    # still recursively applying credential redaction and tensor summaries.
    try:
        attributes = vars(value)
    except Exception:
        attributes = None
    if isinstance(attributes, dict) and attributes:
        return {
            "type": f"{type(value).__module__}.{type(value).__name__}",
            "attributes": {str(k): _json_safe_metadata(v, depth + 1, str(k)) for k, v in attributes.items() if not str(k).startswith("_")},
        }
    # Do not leak arbitrary object reprs, which can contain memory addresses or
    # credentials. A class label is enough for opaque runtime objects.
    return {"type": f"{type(value).__module__}.{type(value).__name__}"}


def _json_dumps(value) -> str:
    return json.dumps(_json_safe_metadata(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _prompt_link(value, prompt: dict) -> tuple[str, int] | None:
    """Recognize ComfyUI's [source_node_id, output_index] prompt link form."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    source_id, output_index = value
    source_key = str(source_id)
    if source_key not in prompt:
        return None
    if not isinstance(output_index, (int, float)):
        return None
    return source_key, int(output_index)


def _collect_prompt_links(value, prompt: dict):
    found = []
    link = _prompt_link(value, prompt)
    if link:
        found.append(link)
    elif isinstance(value, dict):
        for child in value.values():
            found.extend(_collect_prompt_links(child, prompt))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.extend(_collect_prompt_links(child, prompt))
    return found


def _workflow_bypass_routes(workflow):
    """Build LiteGraph mode-4 bypass routes: output slot -> matching input source."""
    bypassed_ids = set()
    bypass_routes = {}
    if not isinstance(workflow, dict):
        return bypassed_ids, bypass_routes
    workflow_nodes = workflow.get("nodes") if isinstance(workflow.get("nodes"), list) else []
    workflow_links = workflow.get("links") if isinstance(workflow.get("links"), list) else []
    for visual_node in workflow_nodes:
        if not isinstance(visual_node, dict) or visual_node.get("id") is None:
            continue
        try:
            mode = int(visual_node.get("mode", 0) or 0)
        except (TypeError, ValueError):
            mode = 0
        if mode == 4:
            bypassed_ids.add(str(visual_node["id"]))
    for link in workflow_links:
        if not isinstance(link, (list, tuple)) or len(link) < 6:
            continue
        _link_id, origin_id, origin_slot, target_id, target_slot, _link_type = link[:6]
        target_key = str(target_id)
        if target_key in bypassed_ids:
            try:
                bypass_routes.setdefault(target_key, {})[int(target_slot)] = (str(origin_id), int(origin_slot))
            except (TypeError, ValueError):
                continue
    return bypassed_ids, bypass_routes


def _rewrite_bypassed_prompt_links(value, prompt_map, bypass_routes, depth=0):
    """Replace links through mode-4 nodes with the matching upstream link."""
    if depth > 32:
        return "<bypass link nesting limit reached>"
    link = _prompt_link(value, prompt_map)
    if link:
        node_id, output_slot = link
        visited = set()
        while node_id in bypass_routes:
            marker = (node_id, output_slot)
            if marker in visited:
                return {"bypassed_source": node_id, "output_slot": output_slot, "error": "bypass cycle"}
            visited.add(marker)
            relay = bypass_routes.get(node_id, {}).get(output_slot)
            if not relay:
                return {"bypassed_source": node_id, "output_slot": output_slot, "error": "no matching bypass input"}
            node_id, output_slot = relay
        return [node_id, output_slot]
    if isinstance(value, dict):
        return {str(key): _rewrite_bypassed_prompt_links(child, prompt_map, bypass_routes, depth + 1) for key, child in value.items()}
    if isinstance(value, list):
        return [_rewrite_bypassed_prompt_links(child, prompt_map, bypass_routes, depth + 1) for child in value]
    if isinstance(value, tuple):
        return [_rewrite_bypassed_prompt_links(child, prompt_map, bypass_routes, depth + 1) for child in value]
    return value


def _collect_active_upstream_nodes(prompt, unique_id, input_name: str = "image_in", workflow=None):
    """Return executed nodes reachable from the selected Kiri output branch.

    For KiriImageLoader sources, the output slot is meaningful: A traces only
    image_in, B traces only mask_in, and C is the selected Loaded Image with no
    upstream graph branch. This keeps C's provenance independent of A/B while
    avoiding unrelated branches in metadata and exported API prompts.
    """
    if not isinstance(prompt, dict):
        return [], {}, ""
    prompt_map = {str(k): v for k, v in prompt.items() if isinstance(v, dict)}
    root_id = str(unique_id) if unique_id is not None else ""
    root = prompt_map.get(root_id)
    if root is None:
        candidates = [(node_id, node) for node_id, node in prompt_map.items()
                      if str(node.get("class_type", "")).endswith("KiriImageLoader")]
        if len(candidates) == 1:
            root_id, root = candidates[0]
    if not isinstance(root, dict):
        return [], {}, root_id

    bypassed_ids, bypass_routes = _workflow_bypass_routes(workflow)
    inputs = root.get("inputs") if isinstance(root.get("inputs"), dict) else {}
    roots = _collect_prompt_links(inputs.get(input_name), prompt_map)
    if not roots:
        return [], {}, root_id

    ordered_ids: list[str] = []
    visited: set[tuple[str, int | None]] = set()
    kiri_output_slots: dict[str, set[int]] = {}

    def visit(node_id: str, output_slot: int | None = None):
        node = prompt_map.get(node_id)
        class_type = str(node.get("class_type", "")) if isinstance(node, dict) else ""
        is_kiri_source = class_type.casefold().endswith("kiriimageloader")
        slot_sensitive = node_id in bypassed_ids or is_kiri_source
        normalized_slot = int(output_slot or 0) if slot_sensitive and output_slot is not None else (0 if is_kiri_source else None)
        visit_key = (node_id, normalized_slot)
        if visit_key in visited:
            return
        visited.add(visit_key)

        if node_id in bypassed_ids:
            relay = bypass_routes.get(node_id, {}).get(int(output_slot or 0))
            if relay:
                visit(relay[0], relay[1])
            return
        if not isinstance(node, dict):
            return

        node_inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
        if is_kiri_source:
            slot = int(normalized_slot or 0)
            kiri_output_slots.setdefault(node_id, set()).add(slot)
            if slot == 0:
                relevant_links = (node_inputs.get("image_in"),)
            elif slot == 1:
                relevant_links = (node_inputs.get("mask_in"),)
            elif slot == 2:
                # C is only the selected Loaded Image. It deliberately does
                # not traverse A/B, even when both are connected on this node.
                relevant_links = ()
            else:
                relevant_links = ()
            for value in relevant_links:
                for source_id, source_slot in _collect_prompt_links(value, prompt_map):
                    visit(source_id, source_slot)
        else:
            for value in node_inputs.values():
                for source_id, source_slot in _collect_prompt_links(value, prompt_map):
                    visit(source_id, source_slot)
        ordered_ids.append(node_id)

    for source_id, source_slot in roots:
        visit(source_id, source_slot)

    nodes = []
    node_map = {}
    emitted = set()
    for node_id in ordered_ids:
        if node_id in emitted:
            continue
        emitted.add(node_id)
        original_entry = prompt_map.get(node_id, {})
        entry = dict(original_entry)
        original_inputs = entry.get("inputs") if isinstance(entry.get("inputs"), dict) else {}
        if node_id in kiri_output_slots:
            slots = kiri_output_slots[node_id]
            keep = set()
            # For A/B, record the selected-file widget only when that specific
            # input is not connected and therefore actually uses the loaded
            # file fallback. Output C always uses the selected file directly.
            if 0 in slots:
                if _collect_prompt_links(original_inputs.get("image_in"), prompt_map):
                    keep.add("image_in")
                else:
                    keep.add("image")
            if 1 in slots:
                if _collect_prompt_links(original_inputs.get("mask_in"), prompt_map):
                    keep.add("mask_in")
                else:
                    keep.add("image")
            if 2 in slots:
                keep.add("image")
            # Retain non-image custom metadata settings for transparency, but
            # never serialize the irrelevant opposite image branch.
            keep.update(key for key in original_inputs if key in {"custom_metadata", "custom_metadata_text"})
            selected_inputs = {str(key): value for key, value in original_inputs.items() if key in keep}
        else:
            selected_inputs = original_inputs
        rewritten_inputs = {
            str(key): _rewrite_bypassed_prompt_links(value, prompt_map, bypass_routes)
            for key, value in selected_inputs.items()
        }
        entry["inputs"] = rewritten_inputs
        node_map[node_id] = entry
        nodes.append({
            "id": node_id,
            "class_type": str(entry.get("class_type", "Unknown")),
            "inputs": _json_safe_metadata(rewritten_inputs),
        })
    return nodes, node_map, root_id


def _active_prompt_payload(prompt, node_map):
    """Return the contributing prompt branch with bypasses rewired and others omitted."""
    if not isinstance(node_map, dict):
        return None
    return {
        str(node_id): _json_safe_metadata(entry)
        for node_id, entry in node_map.items()
        if isinstance(entry, dict)
    }


def _workflow_node_labels(extra_pnginfo, unique_id):
    workflow = extra_pnginfo.get("workflow") if isinstance(extra_pnginfo, dict) else None
    if isinstance(workflow, str):
        try:
            workflow = json.loads(workflow)
        except Exception:
            workflow = None
    labels = {}
    if isinstance(workflow, dict):
        for node in workflow.get("nodes", []) if isinstance(workflow.get("nodes"), list) else []:
            if isinstance(node, dict) and node.get("id") is not None:
                labels[str(node.get("id"))] = {
                    "title": node.get("title"),
                    "type": node.get("type"),
                }
    return workflow if isinstance(workflow, dict) else None, labels


def _resolve_model_file(filename: str, categories: tuple[str, ...]):
    if not isinstance(filename, str) or not filename.strip():
        return None
    getter = getattr(folder_paths, "get_full_path", None)
    if not callable(getter):
        return None
    for category in categories:
        try:
            candidate = getter(category, filename)
        except Exception:
            candidate = None
        if candidate and os.path.isfile(candidate):
            return os.path.realpath(candidate), category
    return None


def _sha256_cached(path: str):
    try:
        stat = os.stat(path)
        cache_key = (os.path.realpath(path), int(stat.st_size), int(stat.st_mtime_ns))
    except OSError:
        return None
    with _MODEL_HASH_LOCK:
        cached = _MODEL_HASH_CACHE.get(cache_key)
    if cached:
        return cached
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        value = digest.hexdigest()
    except OSError:
        return None
    with _MODEL_HASH_LOCK:
        # Drop old versions for this same path to keep the cache bounded.
        for old_key in [item for item in _MODEL_HASH_CACHE if item[0] == cache_key[0]]:
            _MODEL_HASH_CACHE.pop(old_key, None)
        _MODEL_HASH_CACHE[cache_key] = value
    return value


def _classify_model_resource(class_type: str, input_name: str):
    cls = class_type.lower()
    key = "".join(ch for ch in input_name.lower() if ch.isalnum())
    if "lora" in cls or "lora" in key:
        if any(term in key for term in ("name", "path", "file", "lora")):
            return "LoRA", ("loras",)
    if "vae" in cls or key in {"vaename", "vaefile", "vae"}:
        if any(term in key for term in ("name", "file", "vae")):
            return "VAE", ("vae",)
    # More-specific CLIP-vision classification must precede generic clip_name
    # detection, otherwise a CLIPVisionLoader would be mislabeled as a text encoder.
    if "clipvision" in cls or "clipvision" in key:
        return "CLIP Vision", ("clip_vision",)
    if "controlnet" in cls or "controlnet" in key:
        return "ControlNet", ("controlnet",)
    if "ipadapter" in cls or "ipadapter" in key:
        return "IPAdapter", ("ipadapter", "ip_adapters")
    if any(term in cls for term in ("cliploader", "textencoder", "dualclip", "tripleclip")) or any(term in key for term in ("clipname", "textencoder", "t5xxl", "llama")):
        if any(term in key for term in ("name", "file", "clip", "t5", "llama", "encoder")):
            return "Text Encoder", ("text_encoders", "clip")
    if any(term in cls for term in ("checkpoint", "unetloader", "diffusionmodelload", "diffusionmodel", "diffusersloader")) or key in {"ckptname", "checkpointname", "modelname", "unetname", "diffusionmodel", "modelpath", "diffuserspath"}:
        if any(term in key for term in ("name", "file", "ckpt", "checkpoint", "model", "unet", "diffusion", "path")):
            if "diffusers" in cls or "diffusers" in key:
                categories = ("diffusers", "checkpoints", "diffusion_models")
            else:
                categories = ("checkpoints", "diffusion_models", "unet") if "checkpoint" in cls or "ckpt" in key else ("diffusion_models", "unet", "checkpoints")
            return "Checkpoint / Model", categories
    if "upscale" in cls and any(term in key for term in ("model", "name", "file")):
        return "Upscaler", ("upscale_models",)
    return None


def _is_auto_component(value) -> bool:
    normalized = "".join(ch for ch in str(value or "").casefold() if ch.isalnum())
    return normalized in {
        "", "auto", "default", "none", "null", "usecheckpoint", "fromcheckpoint",
        "autofromcheckpoint",
    }


def _lora_sidecar_metadata(path: str):
    """Read the small, durable Civitai provenance fields written by Kiri LoRA Loader."""
    if not path:
        return None
    model_path = Path(path)
    stem = model_path.with_suffix("")
    candidates = [
        Path(str(model_path) + ".cminfo.json"),
        Path(str(stem) + ".cminfo.json"),
        Path(str(stem) + ".kiri-civitai.json"),
        model_path.with_suffix(".metadata.json"),
    ]
    for sidecar in candidates:
        try:
            if not sidecar.is_file() or sidecar.stat().st_size > 2_000_000:
                continue
            with sidecar.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                continue
            provenance = data.get("CivitaiProvenance") if isinstance(data.get("CivitaiProvenance"), dict) else {}
            metadata = {
                "sidecar": sidecar.name,
                "model_id": data.get("CivitaiModelId") or data.get("ModelId") or provenance.get("model_id"),
                "version_id": data.get("CivitaiVersionId") or data.get("VersionId") or provenance.get("version_id"),
                "model_name": data.get("ModelName"),
                "version_name": data.get("VersionName"),
                "creator": data.get("CreatorUsername"),
                "base_model": data.get("BaseModel"),
                "trained_words": data.get("TrainedWords") if isinstance(data.get("TrainedWords"), list) else data.get("trigger_words"),
                "source_url": data.get("ModelUrl") or data.get("VersionUrl") or provenance.get("model_url"),
            }
            metadata = {key: value for key, value in metadata.items() if value not in (None, "", [])}
            return metadata or None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return None


def _kiri_lora_resources(class_type: str, inputs: dict):
    """Expand Kiri LoRA Loader's compact loras_json input into one resource per active LoRA."""
    cls = str(class_type or "").lower()
    if "keryulapowerloraloader" not in cls and "kiri" not in cls:
        return []
    raw = inputs.get("loras_json")
    try:
        rows = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        rows = []
    if not isinstance(rows, list):
        return []
    resources = []
    for item in rows:
        if not isinstance(item, dict) or not item.get("on", True):
            continue
        name = str(item.get("lora") or "").strip()
        if not name:
            continue
        model_strength = item.get("strength_model", item.get("strength", 1.0))
        clip_strength = item.get("strength_clip", item.get("strengthTwo", item.get("strength", 1.0)))
        resource = {
            "type": "LoRA",
            "class_type": class_type,
            "input": "loras_json",
            "name": name,
            "strength_model": _json_safe_metadata(model_strength),
            "strength_clip": _json_safe_metadata(clip_strength),
        }
        resolved = _resolve_model_file(name, ("loras",))
        if resolved:
            resolved_path, folder_category = resolved
            resource["registered_category"] = folder_category
            resource["sha256"] = _sha256_cached(resolved_path)
            try:
                resource["file_size_bytes"] = os.path.getsize(resolved_path)
            except OSError:
                pass
            sidecar = _lora_sidecar_metadata(resolved_path)
            if sidecar:
                resource["civitai_provenance"] = _json_safe_metadata(sidecar)
        resources.append(resource)
    return resources


def _extract_kiri_loader_details(class_type: str, inputs: dict):
    """Expose the dedicated metadata payloads from Kiri's own loader nodes."""
    cls = str(class_type or "").lower()
    details = {}
    if "universalcheckpointloader" in cls or ("kiri" in cls and "checkpointloader" in cls):
        checkpoint = inputs.get("checkpoint") or inputs.get("ckpt_name") or inputs.get("model")
        if isinstance(checkpoint, str) and checkpoint.strip():
            details["checkpoint_or_model"] = checkpoint
        encoders = []
        for key in ("text_encoder", "text_encoder_2", "text_encoder_3", "clip_name", "clip_name2"):
            value = inputs.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            encoders.append({"slot": key, "selection": value, "source": "auto / checkpoint-managed" if _is_auto_component(value) else "external file selection"})
        vaes = []
        for key in ("vae", "vae_2", "vae_name"):
            value = inputs.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            vaes.append({"slot": key, "selection": value, "source": "auto / checkpoint-managed" if _is_auto_component(value) else "external file selection"})
        if encoders:
            details["text_encoders"] = encoders
        if vaes:
            details["vaes"] = vaes
    if "keryulapowerloraloader" in cls:
        trigger_lora = inputs.get("trigger_lora")
        trigger_words = inputs.get("trigger_words")
        if isinstance(trigger_lora, str) and trigger_lora.strip():
            details["trigger_lora"] = trigger_lora
        if isinstance(trigger_words, str) and trigger_words.strip():
            details["trigger_words"] = trigger_words
    return details or None


def _extract_graph_metadata(active_nodes, node_map, labels):
    resources = []
    samplers = []
    text_nodes = []
    kiri_loader_details = []
    positive_nodes = set()
    negative_nodes = set()

    def normalize_key(value):
        return "".join(ch for ch in str(value).lower() if ch.isalnum())

    text_key_names = {
        "text", "string", "value", "prompt", "prompttext", "textvalue",
        "multiline", "multilinetext", "stringvalue", "inputtext",
        "textg", "textl", "clipg", "clipl", "t5xxl", "positive", "negative",
        "positiveprompt", "negativeprompt", "input", "source",
        "text1", "text2", "text3", "text4",
    }

    def _is_textual_node(class_type):
        cls_lower = str(class_type).lower().replace("_", "").replace(" ", "")
        return any(term in cls_lower for term in (
            "cliptextencode", "textencode", "promptencode", "string",
            "multiline", "textarea", "textbox", "primitive", "prompttext",
            "textconcatenate", "textcombine", "textjoin",
        ))

    def _is_text_input_name(name, class_type):
        normalized = normalize_key(name)
        if normalized in text_key_names:
            return True
        if normalized.startswith(("text", "prompt", "string", "multiline", "conditioning", "cond")):
            return True
        # Generic input/source sockets are safe only on nodes that explicitly
        # identify themselves as text/string/primitive nodes. This avoids
        # accidentally following CLIP/model sockets on loader nodes.
        return normalized in {"input", "source"} and _is_textual_node(class_type)

    def prompt_text_ancestors(value):
        """Walk only the text-producing path behind positive/negative conditioning.

        A conditioning node also links to CLIP. Walking every input recursively
        reaches checkpoint loaders and can misclassify selector values such as
        `Auto (from checkpoint)` as prompt text. Follow text-bearing sockets
        only, while still traversing StringMultiline/PrimitiveString chains.
        """
        roots = _collect_prompt_links(value, node_map)
        found = set()
        def walk(node_id):
            if node_id in found:
                return
            found.add(node_id)
            entry = node_map.get(node_id, {})
            class_type = str(entry.get("class_type", "Unknown"))
            inputs = entry.get("inputs") if isinstance(entry.get("inputs"), dict) else {}
            for key, child in inputs.items():
                if not _is_text_input_name(key, class_type):
                    continue
                for parent_id, _ in _collect_prompt_links(child, node_map):
                    walk(parent_id)
        for node_id, _ in roots:
            walk(node_id)
        return found

    # Resolve both prompt-conditioning branches by following text-bearing
    # inputs only. The encoder's `clip` input is deliberately excluded.
    for _node_id, entry in node_map.items():
        class_type = str(entry.get("class_type", "Unknown"))
        cls_lower = class_type.lower().replace("_", "").replace(" ", "")
        inputs = entry.get("inputs") if isinstance(entry.get("inputs"), dict) else {}
        is_sampler_or_guider = any(term in cls_lower for term in (
            "ksampler", "sampler", "guider", "cfgguider", "randomnoise",
        ))
        if not is_sampler_or_guider:
            continue
        for key, branch_set in (("positive", positive_nodes), ("negative", negative_nodes)):
            if key in inputs:
                branch_set.update(prompt_text_ancestors(inputs.get(key)))

    positive_text = []
    negative_text = []

    for node_id, entry in node_map.items():
        class_type = str(entry.get("class_type", "Unknown"))
        inputs = entry.get("inputs") if isinstance(entry.get("inputs"), dict) else {}
        cls_lower = class_type.lower().replace("_", "").replace(" ", "")

        if any(term in cls_lower for term in ("ksampler", "sampler", "guider", "randomnoise")):
            settings = {}
            for key, value in inputs.items():
                normalized = normalize_key(key)
                if any(term in normalized for term in (
                    "seed", "steps", "cfg", "sampler", "scheduler", "denoise",
                    "startatstep", "endatstep", "addnoise", "noise",
                    "returnwithleftovernoise", "forcefulldenoise", "variationstrength",
                )) and _prompt_link(value, node_map) is None:
                    settings[str(key)] = _json_safe_metadata(value)
            if settings:
                samplers.append({"node_id": node_id, "class_type": class_type, "settings": settings})

        # Read text from both traditional encoder widgets and upstream string
        # sources (StringMultiline, PrimitiveString, text-area nodes, etc.).
        # Non-text nodes are not treated as prompts merely because they have a
        # string-valued setting: they must be a recognized text/primitive node
        # or expose an explicit text/prompt input.
        class_looks_textual = any(term in cls_lower for term in (
            "cliptextencode", "textencode", "promptencode", "string",
            "multiline", "textarea", "textbox", "primitive", "prompttext",
        ))
        text_values = {}
        for key, value in inputs.items():
            normalized = normalize_key(key)
            is_text_field = normalized in text_key_names or normalized.startswith(("text", "prompt", "string", "multiline"))
            if class_looks_textual and normalized.startswith("value"):
                is_text_field = True
            if not is_text_field:
                continue
            if _prompt_link(value, node_map) is not None or isinstance(value, (dict, list, tuple)):
                continue
            if isinstance(value, str) and value.strip():
                text_values[str(key)] = value.strip()
        is_branch_member = node_id in positive_nodes or node_id in negative_nodes
        if text_values and (class_looks_textual or is_branch_member):
            in_positive = node_id in positive_nodes
            in_negative = node_id in negative_nodes
            if in_positive and not in_negative:
                branch = "positive"
                positive_text.extend(text_values.values())
            elif in_negative and not in_positive:
                branch = "negative"
                negative_text.extend(text_values.values())
            elif in_positive and in_negative:
                branch = "shared"
                positive_text.extend(text_values.values())
                negative_text.extend(text_values.values())
            else:
                branch = "unclassified"
            text_nodes.append({"node_id": node_id, "class_type": class_type, "branch": branch, "values": text_values})

        loader_details = _extract_kiri_loader_details(class_type, inputs)
        if loader_details:
            kiri_loader_details.append({"node_id": node_id, **_json_safe_metadata(loader_details)})

        if "keryulapowerloraloader" in cls_lower:
            resources.extend(_kiri_lora_resources(class_type, inputs))

        for key, value in inputs.items():
            normalized_key = normalize_key(key)
            if "keryulapowerloraloader" in cls_lower and normalized_key in {"lorasjson", "triggerlora", "triggerwords"}:
                continue
            resource_type = _classify_model_resource(class_type, str(key))
            if not resource_type or not isinstance(value, str):
                continue
            # Kiri CheckPoint Loader's Auto selectors are policies, not model files.
            # The loader-specific details above preserve the exact selected policy.
            if _is_auto_component(value) and ("universalcheckpointloader" in cls_lower or ("kiri" in cls_lower and "checkpointloader" in cls_lower)):
                continue
            category_name, categories = resource_type
            resolved = _resolve_model_file(value, categories)
            resource = {
                "type": category_name,
                "node_id": node_id,
                "class_type": class_type,
                "input": str(key),
                "name": value,
            }
            if resolved:
                resolved_path, folder_category = resolved
                resource["registered_category"] = folder_category
                resource["sha256"] = _sha256_cached(resolved_path)
                try:
                    resource["file_size_bytes"] = os.path.getsize(resolved_path)
                except OSError:
                    pass
            if category_name == "LoRA":
                for strength_key, strength_value in inputs.items():
                    if "strength" in str(strength_key).lower():
                        resource[str(strength_key)] = _json_safe_metadata(strength_value)
            resources.append(resource)
            if category_name == "Checkpoint / Model" and any(term in cls_lower for term in ("checkpointloadersimple", "checkpointloader")) and "universalcheckpointloader" not in cls_lower:
                for embedded_type in ("VAE", "Text Encoder"):
                    resources.append({
                        "type": embedded_type,
                        "node_id": node_id,
                        "class_type": class_type,
                        "name": f"Embedded in {value}",
                        "embedded_in": value,
                        **({"sha256": resource["sha256"]} if resource.get("sha256") else {}),
                    })

    for item in active_nodes:
        label = labels.get(str(item["id"]), {})
        if label.get("title"):
            item["title"] = str(label["title"])
    lora_trigger_words = []
    for item in kiri_loader_details:
        words = item.get("trigger_words")
        if isinstance(words, str) and words.strip():
            lora_trigger_words.append({"node_id": item.get("node_id"), "value": words})

    # Keep one authoritative prompt string per branch. This prevents a generic
    # widget value or a neighbouring sampler setting from being mistaken for
    # the prompt by the image metadata reader / clipboard menu.
    dedupe_text = lambda values: "\n".join(dict.fromkeys(value.strip() for value in values if value.strip()))
    prompts = {
        "positive": dedupe_text(positive_text),
        "negative": dedupe_text(negative_text),
    }
    return {
        "prompts": prompts,
        "samplers": samplers,
        "text_encoders_and_prompts": text_nodes,
        "model_resources": resources,
        "kiri_loader_details": kiri_loader_details,
        "lora_trigger_words": lora_trigger_words,
        "active_upstream_nodes": active_nodes,
    }


def _make_a1111_parameters(
    graph_metadata,
    include_civitai_resources=True,
    include_prompts=None,
    include_positive_prompt=True,
    include_negative_prompt=True,
):
    """Build a parameter string without confusing settings for a prompt.

    An empty positive prompt is represented by an empty first line and an
    explicit Negative prompt marker. This is important for ComfyUI's A1111
    importer: otherwise a settings-only first line can be mistaken for prompt
    text, or the settings cannot be reconstructed as a basic sampler graph.
    ``include_prompts`` is retained as a compatibility alias for older callers.
    """
    if include_prompts is not None:
        include_positive_prompt = bool(include_prompts)
        include_negative_prompt = bool(include_prompts)

    prompts = graph_metadata.get("prompts", {}) if isinstance(graph_metadata.get("prompts"), dict) else {}
    positive = str(prompts.get("positive") or "").strip() if include_positive_prompt else ""
    negative = str(prompts.get("negative") or "").strip() if include_negative_prompt else ""
    if include_positive_prompt and not positive and not negative:
        positive_values = []
        negative_values = []
        for item in graph_metadata.get("text_encoders_and_prompts", []):
            if not isinstance(item, dict):
                continue
            target = positive_values if item.get("branch") == "positive" else negative_values if item.get("branch") == "negative" else None
            if target is None:
                continue
            target.extend(str(value) for value in item.get("values", {}).values() if str(value).strip())
        positive = "\n".join(dict.fromkeys(positive_values))
        negative = "\n".join(dict.fromkeys(negative_values))
        if not include_negative_prompt:
            negative = ""

    # Start with an actual prompt section even when the positive prompt is
    # private. The leading blank line makes ComfyUI recognize the following
    # Negative prompt / Steps structure instead of treating settings as text.
    lines = [positive]
    if negative or not include_negative_prompt or not positive:
        lines.append("Negative prompt: " + negative)

    settings = []
    for sampler in graph_metadata.get("samplers", []):
        values = sampler.get("settings", {}) if isinstance(sampler, dict) else {}
        aliases = [
            ("Steps", ("steps",)), ("Sampler", ("sampler_name", "sampler")),
            ("Scheduler", ("scheduler",)), ("CFG scale", ("cfg", "cfg_scale")),
            ("Seed", ("seed", "noise_seed")), ("Denoise", ("denoise",)),
        ]
        for label, candidates in aliases:
            for candidate in candidates:
                if candidate in values:
                    settings.append(f"{label}: {values[candidate]}")
                    break

    resources = [item for item in graph_metadata.get("model_resources", []) if isinstance(item, dict)]
    resource_labels = {
        "Checkpoint / Model": "Model",
        "VAE": "VAE",
        "Refiner Model": "Refiner",
        "Text Encoder": "Text Encoder",
        "ControlNet": "ControlNet",
        "IPAdapter": "IPAdapter",
        "CLIP Vision": "CLIP Vision",
        "Upscaler": "Upscaler",
    }
    encoder_index = 0
    for resource in resources:
        resource_type = str(resource.get("type") or "")
        name = str(resource.get("name") or "").strip()
        if not name:
            continue
        if resource_type == "Text Encoder":
            encoder_index += 1
            label = f"Text Encoder {encoder_index}"
        else:
            label = resource_labels.get(resource_type)
        if label:
            settings.append(f"{label}: {name}")
        digest = str(resource.get("sha256") or "").strip().lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            digest = ""
        if digest:
            hash_label = {
                "Checkpoint / Model": "Model hash",
                "VAE": "VAE hash",
                "Refiner Model": "Refiner hash",
            }.get(resource_type)
            if hash_label:
                settings.append(f"{hash_label}: {digest}")

    loras = [resource for resource in resources if resource.get("type") == "LoRA" and resource.get("name")]
    for index, lora in enumerate(loras, 1):
        model_strength = lora.get("strength_model", 1.0)
        clip_strength = lora.get("strength_clip")
        suffix = f" [model={model_strength}"
        if clip_strength is not None:
            suffix += f"; clip={clip_strength}"
        suffix += "]"
        settings.append(f"LoRA {index}: {lora.get('name', '')}{suffix}")
        digest = str(lora.get("sha256") or "").strip().lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            digest = ""
        if digest:
            settings.append(f"LoRA {index} hash: {digest}")

    if settings:
        lines.append(", ".join(settings))
    # Keep a leading newline when the positive prompt is intentionally absent;
    # metadata readers use it to distinguish an empty prompt from settings.
    return "\n".join(lines).rstrip()


_PROMPT_TEXT_INPUT_KEYS = {
    "text", "string", "value", "prompt", "prompttext", "textvalue",
    "multiline", "multilinetext", "stringvalue", "inputtext", "textg",
    "textl", "clipg", "clipl", "t5xxl", "positive", "negative",
    "positiveprompt", "negativeprompt", "input", "source", "text1",
    "text2", "text3", "text4",
}


def _normalized_metadata_key(value):
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _prompt_node_ids(graph_metadata, save_positive_prompt=True, save_negative_prompt=True):
    """Return prompt-producing node IDs whose text must be omitted.

    Shared text remains when either branch is allowed, because the same
    upstream text is genuinely used by that retained branch.
    """
    found = set()
    items = graph_metadata.get("text_encoders_and_prompts", []) if isinstance(graph_metadata, dict) else []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        branch = item.get("branch")
        redact = (
            branch == "positive" and not save_positive_prompt
            or branch == "negative" and not save_negative_prompt
            or branch == "shared" and not (save_positive_prompt or save_negative_prompt)
        )
        if redact and item.get("node_id") is not None:
            found.add(str(item["node_id"]))
    return found


def _redact_prompt_values_from_graph(graph_metadata, save_positive_prompt=True, save_negative_prompt=True):
    """Remove only the prompt branch text disabled by the user's preferences."""
    if not isinstance(graph_metadata, dict):
        return set()
    prompt_ids = _prompt_node_ids(graph_metadata, save_positive_prompt, save_negative_prompt)
    prompts = graph_metadata.get("prompts")
    if isinstance(prompts, dict):
        if not save_positive_prompt:
            prompts.pop("positive", None)
        if not save_negative_prompt:
            prompts.pop("negative", None)

    text_items = graph_metadata.get("text_encoders_and_prompts")
    if isinstance(text_items, list):
        def keep_text_item(item):
            if not isinstance(item, dict):
                return True
            branch = item.get("branch")
            if branch == "positive":
                return save_positive_prompt
            if branch == "negative":
                return save_negative_prompt
            if branch == "shared":
                return save_positive_prompt or save_negative_prompt
            return True
        graph_metadata["text_encoders_and_prompts"] = [item for item in text_items if keep_text_item(item)]

    active_nodes = graph_metadata.get("active_upstream_nodes")
    if isinstance(active_nodes, list):
        for item in active_nodes:
            if not isinstance(item, dict) or str(item.get("id")) not in prompt_ids:
                continue
            inputs = item.get("inputs")
            if not isinstance(inputs, dict):
                continue
            class_type = str(item.get("class_type") or "").lower()
            for key, value in list(inputs.items()):
                normalized = _normalized_metadata_key(key)
                is_text_field = normalized in _PROMPT_TEXT_INPUT_KEYS or normalized.startswith(("text", "prompt", "string", "multiline"))
                if normalized == "value" and any(term in class_type for term in ("string", "multiline", "textarea", "textbox", "primitive", "prompt")):
                    is_text_field = True
                if is_text_field and isinstance(value, str):
                    inputs[key] = ""
    return prompt_ids


def _redact_prompt_values_from_api_prompt(api_prompt, prompt_node_ids):
    """Blank prompt-bearing inputs in ComfyUI's API-prompt object, preserving its topology."""
    if not isinstance(api_prompt, dict) or not prompt_node_ids:
        return api_prompt
    for node_id in prompt_node_ids:
        entry = api_prompt.get(str(node_id))
        if not isinstance(entry, dict) or not isinstance(entry.get("inputs"), dict):
            continue
        inputs = entry["inputs"]
        class_type = str(entry.get("class_type") or "").lower()
        for key, value in list(inputs.items()):
            normalized = _normalized_metadata_key(key)
            is_text_field = normalized in _PROMPT_TEXT_INPUT_KEYS or normalized.startswith(("text", "prompt", "string", "multiline"))
            if normalized == "value" and any(term in class_type for term in ("string", "multiline", "textarea", "textbox", "primitive", "prompt")):
                is_text_field = True
            if is_text_field and isinstance(value, str):
                inputs[key] = ""
    return api_prompt


def _redact_prompt_values_from_canvas_workflow(workflow, prompt_node_ids):
    """Blank prompt widgets in a full canvas workflow when prompt saving is OFF."""
    if not isinstance(workflow, dict) or not prompt_node_ids:
        return workflow
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list):
        return workflow
    for node in nodes:
        if not isinstance(node, dict) or str(node.get("id")) not in prompt_node_ids:
            continue
        widget_values = node.get("widgets_values")
        if isinstance(widget_values, list):
            node["widgets_values"] = ["" if isinstance(value, str) else value for value in widget_values]
    return workflow


def _platform_export_record(record):
    """Build a parameter/resource-only record with no ComfyUI node graph data."""
    source_graph = record.get("graph", {}) if isinstance(record, dict) else {}
    if not isinstance(source_graph, dict):
        source_graph = {}

    # Explicitly project only the fields required by external app/site adapters.
    prompts = source_graph.get("prompts", {})
    if not isinstance(prompts, dict):
        prompts = {}
    samplers = []
    for sampler in source_graph.get("samplers", []) if isinstance(source_graph.get("samplers"), list) else []:
        if isinstance(sampler, dict) and isinstance(sampler.get("settings"), dict):
            samplers.append({"settings": _json_safe_metadata(sampler["settings"])})

    forbidden = {"node_id", "class_type", "inputs", "input", "nodes", "links", "workflow", "widgets_values", "active_upstream_nodes"}

    def clean_resource(value):
        if isinstance(value, dict):
            return {
                str(key): clean_resource(child)
                for key, child in value.items()
                if _normalized_metadata_key(key) not in {_normalized_metadata_key(item) for item in forbidden}
            }
        if isinstance(value, list):
            return [clean_resource(item) for item in value]
        return _json_safe_metadata(value)

    resources = [
        clean_resource(item)
        for item in source_graph.get("model_resources", [])
        if isinstance(item, dict)
    ]
    return {
        "schema_version": 1,
        "created_utc": record.get("created_utc"),
        "collector": "Kiri Image Loader platform-parameter adapter",
        "graph": {
            "prompts": {
                "positive": str(prompts.get("positive") or ""),
                "negative": str(prompts.get("negative") or ""),
            },
            "samplers": samplers,
            "model_resources": resources,
        },
    }


def _apply_metadata_save_preferences(record, preferences, preserve_comfy_graph=True):
    """Apply prompt-branch and node-name preferences without touching parameters/resources."""
    safe_record = _json_safe_metadata(record) if isinstance(record, dict) else {}
    graph = safe_record.get("graph") if isinstance(safe_record, dict) else None
    if not isinstance(graph, dict):
        return safe_record

    save_positive = bool(preferences.get("save_positive_prompt", preferences.get("save_prompts", True)))
    save_negative = bool(preferences.get("save_negative_prompt", preferences.get("save_prompts", True)))
    if not save_positive or not save_negative:
        _redact_prompt_values_from_graph(graph, save_positive, save_negative)
        # Regenerate only the human-readable parameter string. Sampler values,
        # model resources, hashes and LoRA strengths stay in graph and are
        # intentionally passed through unchanged.
        safe_record["parameters"] = _make_a1111_parameters(
            graph,
            include_civitai_resources=True,
            include_positive_prompt=save_positive,
            include_negative_prompt=save_negative,
        )
    if not preferences.get("save_node_names", True) and not preserve_comfy_graph:
        graph.pop("active_upstream_nodes", None)
    return safe_record


def _workflow_for_embedding(extra_pnginfo, enabled):
    if not enabled or not isinstance(extra_pnginfo, dict):
        return None
    workflow = extra_pnginfo.get("workflow")
    if isinstance(workflow, str):
        try:
            workflow = json.loads(workflow)
        except Exception:
            return None
    if isinstance(workflow, dict) and isinstance(workflow.get("nodes"), list) and isinstance(workflow.get("links"), list):
        safe_workflow = _json_safe_metadata(workflow)
        # Kiri stores a downscaled display-only comparison snapshot in the
        # serialized node properties so workflow tabs can restore their UI.
        # It is not generation metadata and must not be recursively copied
        # into PNGs when Complete Workflow embedding is enabled.
        for node in safe_workflow.get("nodes", []):
            if not isinstance(node, dict):
                continue
            node_type = str(node.get("type") or node.get("comfyClass") or "")
            if node_type not in {"KiriImageLoader", "KiriImageLoader.KiriImageLoader"}:
                continue
            properties = node.get("properties")
            if isinstance(properties, dict):
                properties.pop("kiri_live_preview", None)
        return safe_workflow
    return None


def _metadata_disabled():
    try:
        from comfy.cli_args import args
        return bool(getattr(args, "disable_metadata", False))
    except Exception:
        return False


def _tensor_samples_for_saving(value):
    """Yield PIL images from IMAGE or MASK tensors without altering outputs."""
    if not _image_value_is_usable(value):
        raise ValueError("The IMAGE/MASK input is empty (no samples or zero-sized dimensions).")
    tensor = value.detach().to(device="cpu", dtype=torch.float32) if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
    array = tensor.numpy()
    if array.ndim == 4:
        if array.shape[-1] in (1, 3, 4):
            samples = [array[index] for index in range(array.shape[0])]
            for sample in samples:
                clean = np.clip(np.nan_to_num(sample, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
                pixels = np.rint(clean * 255.0).astype(np.uint8)
                if pixels.shape[-1] == 1:
                    yield Image.fromarray(pixels[..., 0], mode="L")
                elif pixels.shape[-1] == 4:
                    yield Image.fromarray(pixels, mode="RGBA")
                else:
                    yield Image.fromarray(pixels, mode="RGB")
            return
        if array.shape[1] in (1, 3, 4):
            array = np.moveaxis(array, 1, -1)
            for sample in array:
                clean = np.clip(np.nan_to_num(sample, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
                pixels = np.rint(clean * 255.0).astype(np.uint8)
                yield Image.fromarray(pixels[..., 0], mode="L") if pixels.shape[-1] == 1 else Image.fromarray(pixels, mode="RGBA" if pixels.shape[-1] == 4 else "RGB")
            return
    if array.ndim == 3:
        # MASK tensors are [B,H,W]. A single HWC image is accepted as well.
        if array.shape[-1] in (3, 4) and array.shape[0] > 4 and array.shape[1] > 4:
            array = array[None, ...]
            yield from from_image_batch(array)
            return
        for sample in array:
            clean = np.clip(np.nan_to_num(sample, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            yield Image.fromarray(np.rint(clean * 255.0).astype(np.uint8), mode="L")
        return
    if array.ndim == 2:
        clean = np.clip(np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        yield Image.fromarray(np.rint(clean * 255.0).astype(np.uint8), mode="L")
        return
    raise ValueError(f"Cannot save IMAGE/MASK tensor with shape {tuple(array.shape)}")


def from_image_batch(array):
    for sample in array:
        clean = np.clip(np.nan_to_num(sample, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        pixels = np.rint(clean * 255.0).astype(np.uint8)
        yield Image.fromarray(pixels, mode="RGBA" if pixels.shape[-1] == 4 else "RGB")


def _build_kiri_metadata_record(prompt, extra_pnginfo, unique_id, input_name, custom_metadata_text, custom_metadata):
    """Collect the exact generation graph feeding one Kiri image input."""
    workflow, labels = _workflow_node_labels(extra_pnginfo if isinstance(extra_pnginfo, dict) else {}, unique_id)
    active_nodes, node_map, root_id = _collect_active_upstream_nodes(prompt, unique_id, input_name, workflow=workflow)
    graph_metadata = _extract_graph_metadata(active_nodes, node_map, labels)
    manual_custom = str(custom_metadata_text or "")
    if custom_metadata is not None:
        custom_value = _json_safe_metadata(custom_metadata)
        custom_source = "connected input"
    elif manual_custom.strip():
        try:
            custom_value = json.loads(manual_custom)
            custom_source = "manual editor (JSON)"
        except (json.JSONDecodeError, TypeError):
            custom_value = manual_custom
            custom_source = "manual editor"
    else:
        custom_value = None
        custom_source = None

    record = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "collector": "Kiri Image Loader",
        "root_node_id": root_id,
        "source_workflow_id": workflow.get("id") if isinstance(workflow, dict) else None,
        "metadata_scope": f"active upstream graph contributing to {input_name}",
        "source_input": "A" if input_name == "image_in" else "B" if input_name == "mask_in" else str(input_name),
        "graph": graph_metadata,
        "custom_metadata": {"source": custom_source, "value": custom_value} if custom_value is not None else None,
    }
    # Either image branch can be independently saved. Each record is built
    # from only the graph that contributes to that exact input socket.
    if input_name in {"image_in", "mask_in"}:
        record["parameters"] = _make_a1111_parameters(graph_metadata)
    return record, workflow, _active_prompt_payload(prompt, node_map)



def _safe_output_directory(save_path: str):
    """Resolve a relative output subfolder or an explicitly chosen absolute folder."""
    output_root = os.path.realpath(folder_paths.get_output_directory())
    raw = os.path.expandvars(os.path.expanduser(str(save_path or "").strip()))
    if not raw:
        raise ValueError("Save Path is empty; image saving is disabled until a destination is provided.")
    elif os.path.isabs(raw):
        candidate = os.path.realpath(raw)
        try:
            is_managed_output = os.path.commonpath((output_root, candidate)) == output_root
        except (ValueError, OSError):
            is_managed_output = False
    elif os.path.splitdrive(raw)[0]:
        raise ValueError("Save Path must be a complete absolute path or a relative folder under ComfyUI/output.")
    else:
        candidate = os.path.realpath(os.path.join(output_root, raw))
        try:
            is_managed_output = os.path.commonpath((output_root, candidate)) == output_root
        except (ValueError, OSError):
            is_managed_output = False
        if not is_managed_output:
            raise ValueError("Relative Save Path must stay inside ComfyUI/output. Use an absolute path to choose another folder.")
    os.makedirs(candidate, exist_ok=True)
    if is_managed_output:
        relative = os.path.relpath(candidate, output_root)
        subfolder = "" if relative == "." else relative.replace(os.sep, "/")
    else:
        subfolder = ""
    return candidate, subfolder, not is_managed_output


def _register_external_output_path(path: str):
    """Issue a short-lived opaque token for one file saved outside ComfyUI/output."""
    token = uuid.uuid4().hex
    real_path = os.path.realpath(path)
    with _KIRI_EXTERNAL_OUTPUT_LOCK:
        _KIRI_EXTERNAL_OUTPUT_PATHS[token] = real_path
        while len(_KIRI_EXTERNAL_OUTPUT_PATHS) > 512:
            _KIRI_EXTERNAL_OUTPUT_PATHS.pop(next(iter(_KIRI_EXTERNAL_OUTPUT_PATHS)))
    return token


def _resolve_external_output_path(token: str):
    """Resolve only a path previously registered by this Kiri saver instance."""
    token = str(token or "").strip()
    with _KIRI_EXTERNAL_OUTPUT_LOCK:
        path = _KIRI_EXTERNAL_OUTPUT_PATHS.get(token)
    if not path:
        raise FileNotFoundError("This Kiri external output reference is no longer available. Queue the node again to refresh it.")
    path = os.path.realpath(path)
    if not os.path.isfile(path) or Path(path).suffix.lower() != ".png":
        raise FileNotFoundError("The registered Kiri output image no longer exists.")
    return path


def _forget_external_output_path(token: str):
    with _KIRI_EXTERNAL_OUTPUT_LOCK:
        _KIRI_EXTERNAL_OUTPUT_PATHS.pop(str(token or ""), None)


def _sampler_filename_values(graph_metadata, batch_number: int, counter: int):
    settings = {}
    for sampler in graph_metadata.get("samplers", []) if isinstance(graph_metadata, dict) else []:
        values = sampler.get("settings", {}) if isinstance(sampler, dict) else {}
        if isinstance(values, dict):
            settings.update(values)
    normalized = {"".join(ch for ch in str(key).lower() if ch.isalnum()): value for key, value in settings.items()}
    def first(*keys, default="Unknown"):
        for key in keys:
            value = normalized.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return default
    now = datetime.now()
    seed = first("seed", "noiseseed")
    values = {
        "seed": seed,
        "gottenseed": seed,
        "noiseseed": seed,
        "steps": first("steps"),
        "cfg": first("cfg", "cfgscale"),
        "cfgscale": first("cfg", "cfgscale"),
        "sampler": first("samplername", "sampler"),
        "samplername": first("samplername", "sampler"),
        "scheduler": first("scheduler"),
        "denoise": first("denoise"),
        "counter": str(counter),
        "batch": str(batch_number + 1),
        "batchnum": str(batch_number + 1),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H-%M-%S"),
    }
    return values


def _render_filename_template(template: str, token_values: dict):
    """Replace [Seed], [Gotten Seed], [Steps], etc. and sanitize the filename."""
    raw = str(template or "").strip()
    if not raw:
        raise ValueError("File Name is empty; image saving is disabled until a filename is provided.")
    raw = re.sub(r"\.(?:png|jpe?g|webp|bmp|tiff?)$", "", raw, flags=re.IGNORECASE)
    def replace(match):
        key = "".join(ch for ch in match.group(1).lower() if ch.isalnum())
        return str(token_values.get(key, "Unknown"))
    rendered = re.sub(r"\[([^\]]+)\]", replace, raw)
    # Replace characters forbidden by Windows and POSIX path segments. The
    # directory is a separate widget, so filename input can never add a path.
    rendered = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", rendered)
    rendered = re.sub(r"\s+", " ", rendered).strip(" .")
    return rendered or "Kirin Image"


def _unique_png_filename(folder: str, base: str, counter: int):
    candidate = f"{base}.png"
    if not os.path.exists(os.path.join(folder, candidate)):
        return candidate, counter
    while True:
        candidate = f"{base}_{counter:05}_.png"
        counter += 1
        if not os.path.exists(os.path.join(folder, candidate)):
            return candidate, counter

def _plain_node_names_text(graph_metadata):
    """Return only human-readable node names; never serialize graph structure."""
    nodes = graph_metadata.get("active_upstream_nodes", []) if isinstance(graph_metadata, dict) else []
    names = []
    seen = set()
    if not isinstance(nodes, list):
        return ""
    for item in nodes:
        if not isinstance(item, dict):
            continue
        custom_title = str(item.get("title") or "").strip()
        class_type = str(item.get("class_type") or "").strip()
        # A user-assigned ComfyUI title is the display name. Fall back to the
        # implementation name only when no custom title is present.
        name = custom_title or class_type
        if not name:
            continue
        identity = name.casefold()
        if identity in seen:
            continue
        seen.add(identity)
        names.append(name.replace("\r", " ").replace("\n", " ").strip())
    return "\n".join(names)


def _save_kiri_images(images, prompt, extra_pnginfo, unique_id, embed_workflow, custom_metadata_text, custom_metadata,
                      save_path="", filename_template="", metadata_export_settings=None, input_name="image_in"):

    """Save one Kiri image branch with automatic branch-aware graph metadata.

    Saving is explicitly opt-in: both a valid destination and filename template
    must be supplied. Empty values are not interpreted as default paths/names.
    """
    if not str(save_path or "").strip():
        raise ValueError("Save Path is empty; image saving is disabled until a destination is provided.")
    if not str(filename_template or "").strip():
        raise ValueError("File Name is empty; image saving is disabled until a filename is provided.")
    samples = list(_tensor_samples_for_saving(images))
    if not samples:
        raise ValueError("The primary IMAGE/MASK input contained no samples to save.")

    if input_name not in {"image_in", "mask_in"}:
        raise ValueError(f"Unsupported Kiri save source: {input_name}")
    kiri_metadata, workflow, active_prompt = _build_kiri_metadata_record(
        prompt, extra_pnginfo, unique_id, input_name, custom_metadata_text, custom_metadata
    )
    kiri_metadata["saved_input"] = "A" if input_name == "image_in" else "B"
    export_preferences = normalize_export_settings(metadata_export_settings)
    embed_workflow_enabled = bool(embed_workflow)
    graph_metadata = kiri_metadata.get("graph", {})
    # Keep the human-readable names separate from all serialized records.
    # They can be included in plain text without retaining node IDs or inputs.
    node_names_text = (
        _plain_node_names_text(graph_metadata)
        if export_preferences.get("save_node_names", True) else ""
    )
    prompt_node_ids = _prompt_node_ids(
        graph_metadata,
        export_preferences.get("save_positive_prompt", True),
        export_preferences.get("save_negative_prompt", True),
    )
    kiri_metadata = _apply_metadata_save_preferences(
        kiri_metadata, export_preferences, preserve_comfy_graph=embed_workflow_enabled
    )
    graph_metadata = kiri_metadata.get("graph", {})
    if not export_preferences.get("save_positive_prompt", True) or not export_preferences.get("save_negative_prompt", True):
        active_prompt = _redact_prompt_values_from_api_prompt(active_prompt, prompt_node_ids)
    parameters = str(kiri_metadata.get("parameters") or "")
    platform_record = _platform_export_record(kiri_metadata)
    custom_entry = kiri_metadata.get("custom_metadata") or {}
    custom_value = custom_entry.get("value") if custom_entry else None
    custom_source = custom_entry.get("source") if custom_entry else None
    if not embed_workflow_enabled:
        # The workflow switch controls only ComfyUI graph serialization.
        # Non-ComfyUI apps may still receive their own parameter-only metadata
        # when the separate platform toggle is enabled.
        kiri_metadata["source_workflow_id"] = None
        kiri_metadata["root_node_id"] = None
        parameters = _make_a1111_parameters(
            graph_metadata,
            include_civitai_resources=True,
            include_positive_prompt=export_preferences.get("save_positive_prompt", True),
            include_negative_prompt=export_preferences.get("save_negative_prompt", True),
        )
        kiri_metadata["parameters"] = parameters

    safe_prompt = (
        _json_safe_metadata(active_prompt)
        if embed_workflow_enabled and isinstance(active_prompt, dict) else None
    )
    workflow_to_embed = _workflow_for_embedding(extra_pnginfo, embed_workflow_enabled)
    if embed_workflow_enabled and (not export_preferences.get("save_positive_prompt", True) or not export_preferences.get("save_negative_prompt", True)):
        workflow_to_embed = _redact_prompt_values_from_canvas_workflow(workflow_to_embed, prompt_node_ids)
    extra_chunks = {}
    if embed_workflow_enabled and isinstance(extra_pnginfo, dict):
        for key, value in extra_pnginfo.items():
            token = _metadata_key_token(key)
            # Workflow and API-prompt payloads are written only through the
            # explicit opt-in paths below. Never pass aliases through as
            # arbitrary PNG text chunks.
            if token in _COMFYUI_GRAPH_METADATA_KEYS:
                continue
            extra_chunks[str(key)] = _json_safe_metadata(value)

    full_output_folder, subfolder, external_output = _safe_output_directory(save_path)
    # Fail before creating any output when the supplied template cannot produce
    # a safe filename. The returned output tensors remain independent of saving.
    _render_filename_template(filename_template, _sampler_filename_values(graph_metadata, 0, 1))
    results = []
    metadata_enabled = not _metadata_disabled()
    counter = 1
    for batch_number, pil_image in enumerate(samples):
        token_values = _sampler_filename_values(graph_metadata, batch_number, counter)
        file_base = _render_filename_template(filename_template, token_values)
        file_name, counter = _unique_png_filename(full_output_folder, file_base, counter)
        file_path = os.path.join(full_output_folder, file_name)
        pnginfo = None
        drawthings_comment = None
        if metadata_enabled:
            pnginfo = PngImagePlugin.PngInfo()
            # The actual saved dimensions are authoritative for the exported image.
            graph_metadata.setdefault("image_dimensions", {})
            graph_metadata["image_dimensions"].update({"width": int(pil_image.width), "height": int(pil_image.height)})
            if (not export_preferences.get("platform_metadata", True)) or export_preferences.get("parameters_format") == "a1111":
                sized_parameters = str(kiri_metadata.get("parameters") or parameters)
                size_line = f"Size: {pil_image.width}x{pil_image.height}"
                if sized_parameters and "Size:" not in sized_parameters:
                    sized_parameters = sized_parameters.rstrip() + ", " + size_line
                elif not sized_parameters:
                    sized_parameters = size_line
                kiri_metadata["parameters"] = sized_parameters
                parameters = sized_parameters
            export_chunks = (
                build_export_chunks(platform_record, pil_image.width, pil_image.height, export_preferences)
                if export_preferences.get("platform_metadata", True) else {}
            )
            if embed_workflow_enabled:
                if safe_prompt is not None:
                    pnginfo.add_text("prompt", json.dumps(safe_prompt, ensure_ascii=False))
                if isinstance(kiri_metadata, dict):
                    pnginfo.add_text("kiri_metadata", json.dumps(_json_safe_metadata(kiri_metadata), ensure_ascii=False))
                if export_preferences.get("platform_metadata", True) and export_preferences.get("parameters_format") in {"swarmui", "fooocus", "easy_diffusion"}:
                    # These native profiles own PNG's single `parameters` key
                    # and intentionally use incompatible encodings.
                    if "parameters" in export_chunks:
                        pnginfo.add_text("parameters", export_chunks["parameters"])
                elif parameters:
                    pnginfo.add_text("parameters", parameters)
                if custom_value is not None:
                    pnginfo.add_text("kiri_custom_metadata", json.dumps(_json_safe_metadata({"source": custom_source, "value": custom_value}), ensure_ascii=False))
                if workflow_to_embed is not None:
                    safe_workflow = _json_safe_metadata(workflow_to_embed)
                    pnginfo.add_text("workflow", json.dumps(safe_workflow, ensure_ascii=False))
                # Platform-native fields are independently optional and use
                # a parameter-only record (never ComfyUI node objects).
                for key, value in export_chunks.items():
                    if key in {"parameters", "parameters_format"}:
                        continue
                    try:
                        chunk_value = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                        if key == "XML:com.adobe.xmp" and hasattr(pnginfo, "add_itxt"):
                            pnginfo.add_itxt(key, chunk_value, lang="", tkey="")
                        else:
                            pnginfo.add_text(key, chunk_value)
                    except Exception:
                        continue
                reserved_chunks = {
                    "kirimetadata", "kiricustommetadata", "parameters",
                    "invokeaimetadata", "invokeaiworkflow", "invokeaigraph",
                    "description", "comment", "software", "kiriinterop",
                    "drawthingsusercomment", "sdmetadata", "fooocusscheme",
                    "easydiffusiontask", "easydiffusiontext", "pixaiapi",
                    "tensorarttams", "seaart_handoff", "tensorhub_cozy_creator_handoff",
                    "civitai_meta", "novelai_stealth", "xmlcomadobexmp",
                }
                for key, value in extra_chunks.items():
                    token = _metadata_key_token(key)
                    if token in reserved_chunks or token in _COMFYUI_GRAPH_METADATA_KEYS:
                        continue
                    try:
                        pnginfo.add_text(key, json.dumps(value, ensure_ascii=False))
                    except Exception:
                        continue
                drawthings_comment = export_chunks.get("drawthings_user_comment") if metadata_enabled else None
            else:
                # OFF suppresses all ComfyUI prompt/workflow/Kiri graph records.
                # A selected external-app profile may still own `parameters`
                # (e.g. SwarmUI/Fooocus JSON); these contain app settings only.
                if export_preferences.get("platform_metadata", True) and export_preferences.get("parameters_format") in {"swarmui", "fooocus", "easy_diffusion"} and "parameters" in export_chunks:
                    pnginfo.add_text("parameters", export_chunks["parameters"])
                elif parameters:
                    pnginfo.add_text("parameters", str(parameters))
                if node_names_text:
                    pnginfo.add_text("Kiri Nodes Used", node_names_text)
                platform_state = "ON" if export_preferences.get("platform_metadata", True) else "OFF"
                pnginfo.add_text("Kiri Metadata Mode", f"ComfyUI Workflow OFF; Other-Platform Metadata {platform_state}; no ComfyUI graph data")
                # External app/site parameter records are safe here because
                # their input record is explicitly projected without node IDs,
                # class types, node inputs, links, or workflow arrays.
                for key, value in export_chunks.items():
                    if key in {"parameters", "parameters_format"}:
                        continue
                    try:
                        chunk_value = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                        if key == "XML:com.adobe.xmp" and hasattr(pnginfo, "add_itxt"):
                            pnginfo.add_itxt(key, chunk_value, lang="", tkey="")
                        else:
                            pnginfo.add_text(key, chunk_value)
                    except Exception:
                        continue
                drawthings_comment = export_chunks.get("drawthings_user_comment") if metadata_enabled else None
            if embed_workflow_enabled and node_names_text:
                pnginfo.add_text("Kiri Nodes Used", node_names_text)
        # Draw Things consumes its compact JSON parameter record from EXIF
        # UserComment (0x9286). It is governed by the independent platform toggle.
        drawthings_exif = None
        if drawthings_comment:
            try:
                exif_data = Image.Exif()
                exif_data[0x9286] = drawthings_comment
                exif_data[0x0131] = f"Kiri Image Loader {__version__ if '__version__' in globals() else '0.5.19'}"
                drawthings_exif = exif_data.tobytes()
            except Exception:
                drawthings_exif = None
        if metadata_enabled and export_preferences.get("platform_metadata", True) and export_preferences.get("novelai"):
            # NovelAI reads this gzip JSON from the alpha-channel LSB stream.
            # It is deliberately not signed and does not carry their optional FEC.
            pil_image = embed_novelai_stealth(pil_image, kiri_metadata)
        pil_image.save(file_path, format="PNG", pnginfo=pnginfo, exif=drawthings_exif, compress_level=4)
        file_record = {"filename": file_name, "subfolder": subfolder, "type": "kiri_external" if external_output else "output"}
        sidecar_record = kiri_metadata if embed_workflow_enabled else platform_record
        if metadata_enabled and export_preferences.get("platform_metadata", True) and export_preferences.get("write_sidecar"):
            sidecar_name = Path(file_name).stem + ".kiri.json"
            sidecar_path = os.path.join(full_output_folder, sidecar_name)
            temporary_sidecar = sidecar_path + ".tmp"
            try:
                sidecar_data = build_sidecar(sidecar_record, pil_image.width, pil_image.height, export_preferences)
                with open(temporary_sidecar, "w", encoding="utf-8") as sidecar_handle:
                    json.dump(sidecar_data, sidecar_handle, ensure_ascii=False, indent=2)
                os.replace(temporary_sidecar, sidecar_path)
                file_record["metadata_sidecar"] = sidecar_name
            except Exception as sidecar_error:
                try:
                    if os.path.exists(temporary_sidecar):
                        os.remove(temporary_sidecar)
                except OSError:
                    pass
                print(f"[Kiri Image Loader] Could not write optional metadata sidecar for {file_name}: {sidecar_error}")
        if metadata_enabled and export_preferences.get("platform_metadata", True) and export_preferences.get("write_easy_diffusion_task"):
            task_name = Path(file_name).stem + ".easy-diffusion.json"
            task_path = os.path.join(full_output_folder, task_name)
            temporary_task = task_path + ".tmp"
            try:
                task_data = build_easy_diffusion_request(sidecar_record, pil_image.width, pil_image.height)
                with open(temporary_task, "w", encoding="utf-8") as task_handle:
                    json.dump(task_data, task_handle, ensure_ascii=False, indent=2)
                os.replace(temporary_task, task_path)
                file_record["easy_diffusion_task"] = task_name
            except Exception as task_error:
                try:
                    if os.path.exists(temporary_task):
                        os.remove(temporary_task)
                except OSError:
                    pass
                print(f"[Kiri Image Loader] Could not write Easy Diffusion task JSON for {file_name}: {task_error}")
        if external_output:
            file_record["path_token"] = _register_external_output_path(file_path)
        results.append(file_record)

    if results:
        target_label = full_output_folder
        print(f"[Kiri Image Loader] Saved {len(results)} image(s) to {target_label}: " + ", ".join(item["filename"] for item in results))
    if not metadata_enabled:
        print("[Kiri Image Loader] ComfyUI --disable-metadata is active; images were saved without metadata.")
    return results, kiri_metadata


class KiriImageLoader:
    """Compact image loader with dual-type A/B passthrough and a dedicated Loaded Image C output."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (_get_input_files(), {"image_upload": True}),
                "save_path": ("STRING", {"default": "", "multiline": False, "tooltip": "Optional save folder. Leave blank to disable saving. Relative paths stay under ComfyUI/output; a complete absolute path can target another folder."}),
                "filename_template": ("STRING", {"default": "", "multiline": False, "tooltip": "Optional filename template. Saving occurs only when BOTH Save Path and File Name are filled in. Tokens: [Seed] or [Gotten Seed], [Steps], [CFG], [Sampler], [Scheduler], [Denoise], [Date], [Time], [Counter], [Batch]."}),
                # These serialized settings are hidden by the frontend and
                # edited through Kiri's right-click Metadata Saving menu.
                "custom_metadata_text": ("STRING", {"default": "", "multiline": True, "hidden": True}),
                "embed_workflow": ("BOOLEAN", {"default": False, "hidden": True}),
                "metadata_export_settings": ("STRING", {"default": json.dumps(DEFAULT_EXPORT_SETTINGS, separators=(",", ":")), "multiline": False, "hidden": True}),
            },
            "optional": {
                "image_in": ("IMAGE,MASK", {"tooltip": "Comparison source A. Accepts IMAGE or MASK and passes the exact input through to output A."}),
                "mask_in": ("IMAGE,MASK", {"tooltip": "Comparison source B. Accepts IMAGE or MASK and passes the exact input through to output B."}),
                # ComfyUI's documented wildcard input accepts arbitrary output
                # types. A connected value overrides the manual text editor.
                "custom_metadata": ("*", {"forceInput": True, "tooltip": "Optional custom metadata. Any connected value overrides the manual Custom Metadata editor."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "dynprompt": "DYNPROMPT",
            },
        }


    # A and B are dual-type passthrough sockets. C always exposes the selected
    # Loaded Image as IMAGE, regardless of whether A and/or B are connected.
    RETURN_TYPES = ("IMAGE,MASK", "IMAGE,MASK", "IMAGE")
    RETURN_NAMES = ("A", "B", "C")
    FUNCTION = "load_image"
    CATEGORY = "Kiri/Image"
    OUTPUT_NODE = True  # Ensures the live preview is refreshed even if outputs are not wired onward.

    @classmethod
    def IS_CHANGED(cls, image, custom_metadata_text="", embed_workflow=False,
                   image_in=None, mask_in=None, custom_metadata=None,
                   unique_id=None, prompt=None, extra_pnginfo=None, dynprompt=None,
                   save_path="", filename_template="", **kwargs):
        # Kiri is an output/saver node. Every explicit queue should create a new
        # output image, even when the input file and graph are otherwise cached.
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, input_types=None, **kwargs):
        # Wildcard input contract: per ComfyUI docs, accepting input_types and
        # returning True delegates runtime interpretation to this node.
        return True

    def load_image(self, image, image_in=None, mask_in=None, unique_id=None,
                   custom_metadata_text="", embed_workflow=False, custom_metadata=None,
                   prompt=None, extra_pnginfo=None, dynprompt=None,
                   save_path="", filename_template="", metadata_export_settings=None):
        # A/B are optional and can be connected to upstream nodes that do not
        # produce a value on a particular execution (for example bypassed image
        # paths or an uninitialized VAE Decode branch). Treat None and zero-sized
        # tensors as empty; preserve every usable tensor object exactly.
        input_a_usable = _image_value_is_usable(image_in)
        input_b_usable = _image_value_is_usable(mask_in)
        input_a_linked, input_a_type = _workflow_input_link(prompt, extra_pnginfo, unique_id, "image_in")
        input_b_linked, input_b_type = _workflow_input_link(prompt, extra_pnginfo, unique_id, "mask_in")
        # A supplied value (including an empty tensor) proves that the socket
        # was connected. Prompt/workflow links also catch upstream nodes that
        # return None before their first usable execution.
        input_a_linked = input_a_linked or image_in is not None
        input_b_linked = input_b_linked or mask_in is not None
        # Always attempt to load the selected file once: output C is a dedicated
        # Loaded Image output and must remain available even when both A and B
        # are connected. The same decoded pair is reused for A/B file fallbacks.
        disk_image = disk_mask = None
        fallback_error = None
        selected_file = str(image or "").strip()
        if selected_file:
            try:
                disk_image, disk_mask = _load_disk_image_and_mask(_resolve_image_path(selected_file))
            except Exception as exc:
                fallback_error = str(exc)
                print(f"[Kiri Image Loader] Selected-file image unavailable: {exc}")

        # Both unconnected comparison branches default to the selected Loaded
        # Image. A/B remain dual IMAGE,MASK passthrough sockets: when a MASK is
        # actually connected it is passed through unchanged, but an unused
        # socket must not unexpectedly emit the file's alpha mask instead of
        # the selected image. Users who need a mask can connect one explicitly.
        fallback_a = disk_image
        fallback_b = disk_image
        output_a = image_in if input_a_usable else (None if input_a_linked else fallback_a)
        output_b = mask_in if input_b_usable else (None if input_b_linked else fallback_b)
        synthetic_a = not _image_value_is_usable(output_a)
        synthetic_b = not _image_value_is_usable(output_b)
        # Empty tensors still carry useful H/W and IMAGE/MASK layout hints.
        # Prefer each branch's own dimensions, then borrow the opposite branch
        # only when its own tensor provides no usable spatial extent.
        reference_a = image_in if _value_has_spatial_dimensions(image_in) else (
            output_b if _image_value_is_usable(output_b) else mask_in
        )
        reference_b = mask_in if _value_has_spatial_dimensions(mask_in) else (
            output_a if _image_value_is_usable(output_a) else image_in
        )
        if synthetic_a:
            output_a = _neutral_for_input_type(input_a_type, reference_a, "IMAGE", empty_value=image_in)
        if synthetic_b:
            output_b = _neutral_for_input_type(input_b_type, reference_b, "IMAGE", empty_value=mask_in)

        fallback_messages = []
        if not input_a_usable:
            if input_a_linked:
                fallback_messages.append("Connected Image A input was empty; using a neutral fallback without saving it as a generated image.")
            else:
                fallback_messages.append("Image A is unconnected; using the selected image or a neutral fallback.")
        if not input_b_usable:
            if input_b_linked:
                fallback_messages.append("Connected Image B input was empty; using a neutral fallback.")
            else:
                fallback_messages.append("Image B is unconnected; using the selected Loaded Image or a neutral fallback.")
        if fallback_error:
            fallback_messages.append(f"Selected-file fallback could not be loaded: {fallback_error}")

        # Image saving can target A, B, or both. A linked-but-empty branch is
        # never saved (its neutral tensor is only a graph-safety placeholder);
        # an unconnected branch may save the selected Loaded Image fallback.
        save_ui = {}
        save_path_value = str(save_path or "").strip()
        filename_value = str(filename_template or "").strip()
        export_preferences = normalize_export_settings(metadata_export_settings)
        save_images_from = export_preferences.get("save_images_from", "a")
        save_targets = []
        if save_images_from in {"a", "both"}:
            save_targets.append(("a", output_a, synthetic_a, "image_in"))
        if save_images_from in {"b", "both"}:
            save_targets.append(("b", output_b, synthetic_b, "mask_in"))

        if not save_path_value or not filename_value:
            missing = []
            if not save_path_value:
                missing.append("Save Path")
            if not filename_value:
                missing.append("File Name")
            save_ui["kiri_save_skipped"] = [
                f"Image saving is disabled because { ' and '.join(missing) } { 'is' if len(missing) == 1 else 'are' } empty. Image outputs are passed through normally."
            ]
        else:
            all_saved_files = []
            save_errors = []
            for target_key, target_tensor, target_is_synthetic, target_input_name in save_targets:
                if target_is_synthetic:
                    save_ui.setdefault("kiri_save_skipped", []).append(
                        f"Image {target_key.upper()} had no real source sample; its neutral safety fallback was not saved."
                    )
                    continue
                try:
                    saved_files, saved_metadata = _save_kiri_images(
                        target_tensor, prompt, extra_pnginfo, unique_id, embed_workflow,
                        custom_metadata_text, custom_metadata, save_path_value, filename_value,
                        metadata_export_settings, input_name=target_input_name,
                    )
                    all_saved_files.extend(saved_files)
                    save_ui[f"kiri_saved_files_{target_key}"] = [saved_files]
                    save_ui[f"kiri_runtime_metadata_{target_key}"] = [
                        json.dumps(_json_safe_metadata(saved_metadata), ensure_ascii=False)
                    ]
                except Exception as exc:
                    save_errors.append(f"Image {target_key.upper()}: {exc}")
                    print(f"[Kiri Image Loader] Could not save Image {target_key.upper()} with metadata: {exc}")
            if "kiri_saved_files_a" in save_ui:
                # Preserve the legacy A-only field for clients that predate
                # branch-selectable saving. A B-only save must never be
                # misidentified as the upstream Kiri A output.
                save_ui["kiri_saved_files"] = [save_ui["kiri_saved_files_a"][0]]
            if save_errors:
                save_ui["kiri_save_error"] = save_errors

        if input_b_usable and "kiri_runtime_metadata_b" not in save_ui:
            try:
                metadata_record_b, _workflow_b, _active_prompt_b = _build_kiri_metadata_record(
                    prompt, extra_pnginfo, unique_id, "mask_in", custom_metadata_text, custom_metadata
                )
                preferences_b = normalize_export_settings(metadata_export_settings)
                metadata_record_b = _apply_metadata_save_preferences(
                    metadata_record_b, preferences_b, preserve_comfy_graph=bool(embed_workflow)
                )
                save_ui["kiri_runtime_metadata_b"] = [json.dumps(_json_safe_metadata(metadata_record_b), ensure_ascii=False)]
            except Exception as exc:
                print(f"[Kiri Image Loader] Could not collect Image B graph metadata: {exc}")

        # C is deliberately independent of A/B: it always exposes the selected
        # Loaded Image. If the selector is empty/unavailable, return a typed
        # neutral IMAGE so the optional output remains safe to connect.
        output_c = disk_image if _image_value_is_usable(disk_image) else _neutral_image_like(
            output_a if _image_value_is_usable(output_a) else output_b
        )

        # Only actual upstream pixels belong in the live comparison preview.
        # Neutral safety tensors must not masquerade as generated previews: if
        # neither branch has real pixels, the frontend falls back to Loaded
        # Image. A single valid branch is displayed alone.
        preview_image_a = image_in if input_a_usable else None
        preview_image_b = mask_in if input_b_usable else None
        if preview_image_a is not None or preview_image_b is not None:
            live_preview_ui = _create_live_preview_payload(preview_image_a, preview_image_b)
        else:
            live_preview_ui = {"kiri_preview_mode": ["file"]}
        if fallback_messages:
            live_preview_ui["kiri_input_warning"] = [" ".join(fallback_messages)]
        live_preview_ui.update(save_ui)

        return {"ui": live_preview_ui, "result": (output_a, output_b, output_c)}



NODE_CLASS_MAPPINGS = {
    "KiriImageLoader": KiriImageLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KiriImageLoader": "Kiri Image Loader",
}

WEB_DIRECTORY = "./web"


# These local-server routes are independent of workflow execution. They serve
# automatic metadata reads and explicit file actions from the preview menu.
try:
    from server import PromptServer

    @PromptServer.instance.routes.get("/kiri-image-loader/metadata")
    async def kiri_image_loader_metadata(request):
        from aiohttp import web

        filename = request.query.get("filename", "")
        file_type = request.query.get("type", "input")
        subfolder = request.query.get("subfolder", "")
        path_token = request.query.get("path_token", "")
        try:
            image_path = _resolve_external_output_path(path_token) if path_token else _resolve_managed_image_path(filename, file_type, subfolder)
            payload = inspect_image(image_path)
            payload["selected_filename"] = filename
            payload["selected_type"] = file_type
            payload["selected_subfolder"] = subfolder
            return web.json_response(payload)
        except FileNotFoundError as exc:
            return web.json_response({"ok": False, "error": str(exc), "copy_items": []}, status=404)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc), "copy_items": []}, status=500)

    @PromptServer.instance.routes.get("/kiri-image-loader/files")
    async def kiri_image_loader_files(_request):
        from aiohttp import web
        return web.json_response({"ok": True, "files": _get_input_files()})

    @PromptServer.instance.routes.get("/kiri-image-loader/image")
    async def kiri_image_loader_image(request):
        """Serve an image for the explicit clipboard-copy action.

        Managed paths are resolved through the same input/output/temp allowlist
        as metadata reads. External outputs require an opaque session token.
        """
        from aiohttp import web

        filename = request.query.get("filename", "")
        file_type = request.query.get("type", "input")
        subfolder = request.query.get("subfolder", "")
        path_token = request.query.get("path_token", "")
        try:
            image_path = _resolve_external_output_path(path_token) if path_token else _resolve_managed_image_path(filename, file_type, subfolder)
            return web.FileResponse(image_path, headers={"Cache-Control": "no-store"})
        except FileNotFoundError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=404)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    @PromptServer.instance.routes.post("/kiri-image-loader/file-action")
    async def kiri_image_loader_file_action(request):
        from aiohttp import web

        try:
            body = await request.json()
            filename = body.get("filename", "") if isinstance(body, dict) else ""
            file_type = body.get("type", "input") if isinstance(body, dict) else "input"
            subfolder = body.get("subfolder", "") if isinstance(body, dict) else ""
            path_token = body.get("path_token", "") if isinstance(body, dict) else ""
            action = body.get("action", "") if isinstance(body, dict) else ""
            image_path = _resolve_external_output_path(path_token) if path_token else _resolve_managed_image_path(filename, file_type, subfolder)

            if action == "edit_metadata":
                if str(file_type).lower() == "temp":
                    return web.json_response({"ok": False, "error": "Temporary preview images cannot be edited."}, status=400)
                entries = body.get("metadata_entries") if isinstance(body, dict) else None
                editor_values = body.get("editor_values") if isinstance(body, dict) else None
                count = _rewrite_png_text_metadata(image_path, entries, editor_values)
                message = f"Updated {count} embedded PNG metadata field(s). Image pixels and non-text PNG chunks were preserved."
            elif action == "open":
                _open_in_default_viewer(image_path)
                message = "Opened image in the system's default image viewer."
            elif action == "reveal":
                _reveal_in_file_manager(image_path)
                message = "Opened the image's containing folder."
            elif action == "delete":
                if str(file_type).lower() == "temp":
                    return web.json_response({"ok": False, "error": "Temporary preview files cannot be deleted through Kiri Image Loader."}, status=400)
                _move_to_trash(image_path)
                if path_token:
                    _forget_external_output_path(path_token)
                message = "Moved the selected image to the system Recycle Bin / Trash."
            else:
                return web.json_response({"ok": False, "error": "Unknown file action."}, status=400)

            return web.json_response({"ok": True, "action": action, "message": message})
        except FileNotFoundError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=404)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

except Exception as _route_error:
    # Import-time registration must not prevent ComfyUI from loading the node
    # on older builds. The package logs a useful warning instead.
    print(f"[Kiri Image Loader] Local routes were not registered: {_route_error}")


try:
    print(
        "[Kiri Image Loader] Loaded. Recursive image discovery root: "
        f"{os.path.realpath(folder_paths.get_input_directory())}"
    )
except Exception:
    print("[Kiri Image Loader] Loaded (dual IMAGE/MASK passthrough and metadata reader enabled).")
