"""Manual material masks, optional color guards, and inward-only feathering."""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from .core import _mask_array, _rgb_array

def _polygon(points, size):
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] < 3 or not np.isfinite(values).all():
        raise ValueError("Each polygon needs at least three finite [x, y] points")
    if (values < 0).any() or (values[:, 0] > size[0]).any() or (values[:, 1] > size[1]).any():
        raise ValueError("Polygon points must be inside the input canvas")
    image = Image.new("L", size)
    ImageDraw.Draw(image).polygon([tuple(p) for p in values], fill=255)
    return np.asarray(image) > 0

def _color_guard(source, guard):
    selected = np.ones(source.shape[:2], dtype=bool)
    rgb = source.astype(np.int16)
    for key, operation in (("min_rgb", np.greater_equal), ("max_rgb", np.less_equal)):
        if key in guard:
            threshold = np.asarray(guard[key], dtype=float)
            if threshold.shape != (3,) or not np.isfinite(threshold).all() or (threshold < 0).any() or (threshold > 255).any():
                raise ValueError(f"{key} must contain three values in [0, 255]")
            selected &= np.all(operation(rgb, threshold), axis=2)
    if "b_minus_r_min" in guard:
        threshold = float(guard["b_minus_r_min"])
        if not np.isfinite(threshold):
            raise ValueError("b_minus_r_min must be finite")
        selected &= rgb[..., 2] - rgb[..., 0] >= threshold
    return selected

def build_masks(source, config, base_mask=None):
    source = _rgb_array(source)
    height, width = source.shape[:2]
    size = (width, height)
    if "coordinate_size" in config and list(config["coordinate_size"]) != list(size):
        raise ValueError("coordinate_size must match the input; relabel polygons for another image")
    allowed = np.zeros((height, width), dtype=bool) if base_mask is None else _mask_array(base_mask, (height, width)).copy()
    allow_specs = config.get("allow", [])
    if not allow_specs and base_mask is None:
        raise ValueError("Provide allow polygons or a base_mask")
    for spec in allow_specs:
        part = _polygon(spec["points"], size)
        if "color_guard" in spec:
            part &= _color_guard(source, spec["color_guard"])
        allowed |= part
    for spec in config.get("protect", []):
        allowed &= ~_polygon(spec["points"], size)
    active = allowed.copy()
    if "active_rects" in config:
        roi = Image.new("L", size)
        draw = ImageDraw.Draw(roi)
        for rectangle in config["active_rects"]:
            values = np.asarray(rectangle, dtype=float)
            if values.shape != (4,) or not np.isfinite(values).all():
                raise ValueError("active_rects requires [left, top, right, bottom] rectangles")
            left, top, right, bottom = values
            if left < 0 or top < 0 or right > width or bottom > height or right <= left or bottom <= top:
                raise ValueError("Active rectangles must fit inside the input canvas")
            # Rectangles use half-open coordinates, matching PIL crop.
            if not np.equal(values, np.floor(values)).all():
                raise ValueError("Active rectangle coordinates must be integers")
            draw.rectangle((int(left), int(top), int(right)-1, int(bottom)-1), fill=255)
        active &= np.asarray(roi) > 0
    feather = float(config.get("feather_px", 7))
    if not np.isfinite(feather) or feather < 0:
        raise ValueError("feather_px must be finite and nonnegative")
    if feather == 0:
        alpha = active.astype(np.float32)
    else:
        # Padding gives a defined zero background even for an all-white bitmap.
        distance = distance_transform_edt(np.pad(active, 1, constant_values=False))[1:-1, 1:-1]
        alpha = np.clip(distance / feather, 0, 1).astype(np.float32)
    alpha[~allowed] = 0
    meta = {"geometry": list(size), "editable_fraction": float(allowed.mean()),
            "active_fraction": float(active.mean()), "feather_px": feather,
            "allow_polygons": len(allow_specs), "protect_polygons": len(config.get("protect", [])),
            "mask_method": "manual polygons/bitmap plus optional input color guards",
            "distance_domain": "active mask, zero-padded image boundary", "active_mask": active}
    return allowed, alpha, meta
