"""Command-line workflow: scope, proposal, projection, audit and comparison."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .core import (audit_arrays, file_sha256, project_arrays, read_alpha, read_mask,
                   read_rgb, save_gray, save_rgb, write_json)
from .masks import build_masks

MASK_FILES = ("mask.png", "active_mask.png", "alpha.png", "overlay.png", "mask_report.json")

def _preflight(paths, overwrite=False, sources=()):
    resolved = [Path(p).resolve() for p in paths if p is not None]
    if len(resolved) != len(set(resolved)):
        raise ValueError("Output paths must be distinct")
    if set(resolved) & {Path(p).resolve() for p in sources if p is not None}:
        raise ValueError("Output paths must not replace input files")
    for path in resolved:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")

def _mask(args):
    output = Path(args.output_dir)
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = read_rgb(args.input)
    base_path = config_path.parent / config["base_mask"] if config.get("base_mask") else None
    base = read_mask(base_path, (source.shape[1], source.shape[0])) if base_path else None
    paths = [output / name for name in MASK_FILES]
    _preflight(paths, args.overwrite, (args.input, config_path, base_path))
    mask, alpha, meta = build_masks(source, config, base)
    active = meta.pop("active_mask")
    overlay = source.astype(np.float32)
    overlay[mask] = overlay[mask] * 0.65 + np.array([60, 200, 110], dtype=np.float32) * 0.35
    save_gray(mask, output / "mask.png", args.overwrite)
    save_gray(active, output / "active_mask.png", args.overwrite)
    save_gray(alpha, output / "alpha.png", args.overwrite)
    save_rgb(np.uint8(np.rint(overlay)), output / "overlay.png", args.overwrite)
    meta.update({"input_sha256": file_sha256(args.input), "config_sha256": file_sha256(config_path),
                 "alpha_encoding": "8-bit grayscale / 255", "alpha_sha256": file_sha256(output / "alpha.png")})
    write_json(meta, output / "mask_report.json", args.overwrite)
    return meta

def _project(args):
    _preflight((args.output, args.report), args.overwrite, (args.input, args.proposal, args.mask, args.alpha))
    source, proposal = read_rgb(args.input), read_rgb(args.proposal)
    size = (source.shape[1], source.shape[0])
    mask = read_mask(args.mask, size)
    alpha = read_alpha(args.alpha, size) if args.alpha else None
    result = project_arrays(source, proposal, mask, alpha)
    save_rgb(result, args.output, args.overwrite)
    report = audit_arrays(source, read_rgb(args.output), mask)
    report.update({"input_sha256": file_sha256(args.input), "proposal_sha256": file_sha256(args.proposal),
                   "mask_sha256": file_sha256(args.mask), "alpha_sha256": file_sha256(args.alpha) if args.alpha else None,
                   "output_sha256": file_sha256(args.output), "new_training": False,
                   "geometry_operation": "same-size RGB projection; no resizing or warp"})
    if not report["scope_preservation_passed"]:
        raise RuntimeError("Saved PNG failed protected-pixel verification")
    if args.report:
        write_json(report, args.report, args.overwrite)
    return report

def _audit(args):
    reference, result = read_rgb(args.reference), read_rgb(args.image)
    mask = read_mask(args.mask, (reference.shape[1], reference.shape[0]))
    return audit_arrays(reference, result, mask)

def _compare(args):
    _preflight((args.output,), args.overwrite, args.images)
    if len(args.images) < 2:
        raise ValueError("Comparison requires at least two images")
    images = [Image.fromarray(read_rgb(path)) for path in args.images]
    labels = args.labels or [Path(p).name for p in args.images]
    if len(labels) != len(images):
        raise ValueError("Provide one label per comparison image")
    cell_width = 384
    cell_height = max(int(im.height * cell_width / im.width) for im in images)
    canvas = Image.new("RGB", (cell_width * len(images), cell_height + 40), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(args.font or "DejaVuSans.ttf", 18)
        safe_labels = labels
    except OSError:
        if args.font:
            raise ValueError("Could not open the requested comparison font")
        font = ImageFont.load_default()
        safe_labels = [label.encode("ascii", "backslashreplace").decode() for label in labels]
    resize = getattr(Image, "Resampling", Image).LANCZOS
    for index, (image, label) in enumerate(zip(images, safe_labels)):
        draw.text((index * cell_width + 8, 10), label, fill="black", font=font)
        thumbnail = image.copy()
        thumbnail.thumbnail((cell_width, cell_height), resize)
        canvas.paste(thumbnail, (index * cell_width + (cell_width - thumbnail.width) // 2, 40))
    save_rgb(np.asarray(canvas), args.output, args.overwrite)
    return {"images": len(images), "preview_only": True, "output_sha256": file_sha256(args.output)}

def _prompt(args):
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    if args.prompt:
        return args.prompt
    raise ValueError("Model editing requires --prompt-file or --prompt")

def _propose(args):
    from .backends import generate_klein, generate_openai
    report_path = args.report or str(Path(args.output).with_suffix(".json"))
    raw_path = Path(args.output).with_name(Path(args.output).stem + ".raw.png")
    _preflight((args.output, report_path, raw_path), args.overwrite, (args.input, args.prompt_file))
    source = read_rgb(args.input)
    prompt = _prompt(args)
    if args.crop:
        left, top, right, bottom = args.crop
        if left < 0 or top < 0 or right > source.shape[1] or bottom > source.shape[0] or right <= left or bottom <= top:
            raise ValueError("Crop must fit inside the input, with half-open coordinates")
    with tempfile.TemporaryDirectory(prefix="aigc-clear-") as temp:
        temp = Path(temp)
        local_input, local_proposal = temp / "input.png", temp / "proposal.png"
        crop = source if not args.crop else source[top:bottom, left:right]
        save_rgb(crop, local_input)
        if args.backend == "openai":
            metadata = generate_openai(local_input, prompt, local_proposal, model=args.model or "gpt-image-2", quality=args.quality, size=args.size)
        else:
            from .backends import KLEIN_MODEL, KLEIN_REVISION
            metadata = generate_klein(local_input, prompt, local_proposal, model_id=args.model or KLEIN_MODEL,
                revision=args.revision or KLEIN_REVISION, seed=args.seed, steps=args.steps,
                device=args.device, local_files_only=args.local_files_only)
        try:
            proposal = read_rgb(local_proposal)
            if proposal.shape != crop.shape:
                raise ValueError(f"Editor returned {proposal.shape[1]}x{proposal.shape[0]}, expected {crop.shape[1]}x{crop.shape[0]}")
        except (ValueError, OSError) as exc:
            # Preserve original bytes, including alpha, before the temporary directory closes.
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            with raw_path.open("wb" if args.overwrite else "xb") as stream:
                stream.write(local_proposal.read_bytes())
            raise ValueError(f"{exc}; raw saved as {raw_path}") from None
        if args.crop:
            full = source.copy()
            full[top:bottom, left:right] = proposal
            proposal = full
        save_rgb(proposal, args.output, args.overwrite)
    metadata["model_output_size"] = metadata["output_size"]
    metadata["output_size"] = [proposal.shape[1], proposal.shape[0]]
    metadata.update({"input_sha256": file_sha256(args.input), "output_sha256": file_sha256(args.output),
                     "crop": args.crop, "prompt": prompt, "new_training": False})
    write_json(metadata, report_path, args.overwrite)
    return metadata

def _run(args):
    out = Path(args.output_dir)
    if args.proposal and args.backend:
        raise ValueError("Choose an existing --proposal or a --backend")
    if not args.proposal and not args.backend:
        raise ValueError("Provide --proposal or --backend")
    if args.backend:
        _prompt(args)
    source = read_rgb(args.input)
    if args.proposal:
        proposal_array = read_rgb(args.proposal)
        if proposal_array.shape != source.shape:
            raise ValueError("Proposal geometry must match the input; no automatic resize or warp")
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_path = config_path.parent / config["base_mask"] if config.get("base_mask") else None
    files = [out / "masks" / name for name in MASK_FILES] + [out / "result.png", out / "report.json", out / "comparison.png"]
    if args.backend:
        files += [out / "proposal.png", out / "proposal.json", out / "proposal.raw.png"]
    _preflight(files, args.overwrite, (args.input, config_path, base_path, args.proposal, args.prompt_file))
    mask_args = argparse.Namespace(**vars(args))
    mask_args.output_dir = out / "masks"
    _mask(mask_args)
    proposal = args.proposal
    if args.backend:
        edit_args = argparse.Namespace(**vars(args))
        edit_args.output, edit_args.report = out / "proposal.png", out / "proposal.json"
        _propose(edit_args)
        proposal = edit_args.output
    project_args = argparse.Namespace(input=args.input, proposal=proposal, mask=out/"masks/mask.png",
        alpha=out/"masks/alpha.png", output=out/"result.png", report=out/"report.json", overwrite=args.overwrite)
    report = _project(project_args)
    _compare(argparse.Namespace(images=[args.input, out/"result.png"], labels=["Input", "Projected edit"],
                               output=out/"comparison.png", font=None, overwrite=args.overwrite))
    return report

def _editor_options(parser):
    parser.add_argument("--backend", choices=("openai", "klein"))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--prompt")
    group.add_argument("--prompt-file", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--quality", default="high", choices=("auto", "low", "medium", "high"))
    parser.add_argument("--size", default="auto")
    parser.add_argument("--revision")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--crop", nargs=4, type=int, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))

def make_parser():
    parser = argparse.ArgumentParser(prog="aigc-clear", description="Frozen editor proposals with exact protected pixels")
    parser.add_argument("--version", action="version", version="aigc-clear 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("mask", "project", "audit", "compare", "propose", "run"):
        sub = commands.add_parser(name)
        sub.set_defaults(handler=globals()["_" + name])
        if name != "audit":
            sub.add_argument("--overwrite", action="store_true")
        if name in ("mask", "project", "propose", "run"):
            sub.add_argument("--input", type=Path, required=True)
        if name in ("mask", "run"):
            sub.add_argument("--config", type=Path, required=True)
            sub.add_argument("--output-dir", type=Path, required=True)
        if name in ("project", "run"):
            sub.add_argument("--proposal", type=Path, required=name == "project")
        if name == "project":
            sub.add_argument("--mask", type=Path, required=True)
            sub.add_argument("--alpha", type=Path)
        if name in ("project", "propose", "compare"):
            sub.add_argument("--output", type=Path, required=True)
        if name in ("project", "propose"):
            sub.add_argument("--report", type=Path)
        if name == "audit":
            sub.add_argument("--reference", type=Path, required=True)
            sub.add_argument("--image", type=Path, required=True)
            sub.add_argument("--mask", type=Path, required=True)
        if name == "compare":
            sub.add_argument("--images", nargs="+", type=Path, required=True)
            sub.add_argument("--labels", nargs="+")
            sub.add_argument("--font")
        if name in ("propose", "run"):
            _editor_options(sub)
    return parser

def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.command == "propose" and not args.backend:
        parser.error("propose requires --backend openai or klein")
    try:
        result = args.handler(args)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
        return 0 if result.get("scope_preservation_passed", True) else 1
    except Exception as exc:
        message = str(exc)
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            message = message.replace(key, "[REDACTED]")
        print("error: " + message, file=sys.stderr)
        return 2
