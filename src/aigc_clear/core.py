"""Lossless RGB projection and measurable scope preservation."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

def _rgb_array(value):
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) < 1 or array.dtype != np.uint8:
        raise ValueError("Images must be nonempty H x W x 3 uint8 RGB arrays")
    return array

def _mask_array(value, shape):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "buif" or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("Mask must be a finite, nonnegative H x W array matching the image")
    return array > 0

def read_rgb(path):
    with Image.open(path) as image:
        has_alpha = "A" in image.getbands() or "transparency" in image.info
        if has_alpha and np.asarray(image.convert("RGBA").getchannel("A")).min() < 255:
            raise ValueError("Transparent input is not supported; flatten it explicitly first")
        return np.asarray(image.convert("RGB")).copy()

def read_mask(path, size=None):
    with Image.open(path) as image:
        if size is not None and image.size != tuple(size):
            raise ValueError("Mask geometry does not match the image")
        return np.asarray(image.convert("L")) > 0

def read_alpha(path, size=None):
    with Image.open(path) as image:
        if size is not None and image.size != tuple(size):
            raise ValueError("Alpha geometry does not match the image")
        return np.asarray(image.convert("L")).astype(np.float32) / 255

def project_arrays(base, proposal, mask, alpha=None):
    base, proposal = _rgb_array(base), _rgb_array(proposal)
    if proposal.shape != base.shape:
        raise ValueError("Proposal geometry must match the input; no automatic resize or warp")
    allowed = _mask_array(mask, base.shape[:2])
    weight = allowed.astype(np.float32) if alpha is None else np.asarray(alpha, dtype=np.float32)
    if weight.shape != allowed.shape or not np.isfinite(weight).all() or (weight < 0).any() or (weight > 1).any():
        raise ValueError("Alpha must be finite H x W values in [0, 1]")
    if np.any(weight[~allowed] != 0):
        raise ValueError("Alpha must be zero outside the allowed mask")
    x, z = base.astype(np.float32), proposal.astype(np.float32)
    result = np.uint8(np.clip(np.rint(x + weight[..., None] * (z - x)), 0, 255))
    result[~allowed] = base[~allowed]
    return result

def audit_arrays(reference, result, mask):
    reference, result = _rgb_array(reference), _rgb_array(result)
    if reference.shape != result.shape:
        raise ValueError("Audit images must have identical geometry")
    allowed = _mask_array(mask, reference.shape[:2])
    protected = ~allowed
    difference = np.abs(result.astype(np.int16) - reference.astype(np.int16))
    protected_count = int(protected.sum())
    return {
        "geometry": [reference.shape[1], reference.shape[0]],
        "protected_pixels": protected_count,
        "protected_fraction": float(protected.mean()),
        "protected_max_rgb_error": int(difference[protected].max()) if protected_count else 0,
        "protected_changed_pixels": int(np.any(difference > 0, axis=2)[protected].sum()),
        "editable_pixels": int(allowed.sum()),
        "editable_changed_pixels": int(np.any(difference > 0, axis=2)[allowed].sum()),
        "scope_preservation_passed": not bool(np.any(difference[protected])),
        "aesthetic_quality_measured": False,
    }

def _write_target(path, overwrite):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def save_rgb(image, path, overwrite=False):
    image = _rgb_array(image)
    path = _write_target(path, overwrite)
    if path.suffix.lower() != ".png":
        raise ValueError("Use a .png output for lossless pixel preservation")
    with path.open("wb" if overwrite else "xb") as stream:
        Image.fromarray(image).save(stream, format="PNG")

def save_gray(array, path, overwrite=False):
    values = np.asarray(array)
    if values.dtype == np.bool_:
        values = values.astype(np.uint8) * 255
    elif values.dtype != np.uint8:
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("Float grayscale arrays must be in [0, 1]")
        values = np.uint8(np.rint(values * 255))
    if values.ndim != 2:
        raise ValueError("Grayscale output must be H x W")
    path = _write_target(path, overwrite)
    if path.suffix.lower() != ".png":
        raise ValueError("Use a .png grayscale output")
    with path.open("wb" if overwrite else "xb") as stream:
        Image.fromarray(values).save(stream, format="PNG")

def write_json(data, path, overwrite=False):
    path = _write_target(path, overwrite)
    with path.open("w" if overwrite else "x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
