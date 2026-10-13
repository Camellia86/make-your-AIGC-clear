"""Optional image-editing backends, imported only when requested.

These functions save the model's raw output. In particular, they never resize an
output to match its source; callers can check ``output_size`` before compositing.
"""

import base64
import binascii
import io
import os
from pathlib import Path
import time
from typing import Any


KLEIN_MODEL = "black-forest-labs/FLUX.2-klein-4B"
KLEIN_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"


def _output_path(source: Path, output: Path, overwrite: bool) -> Path:
    source, output = Path(source), Path(output)
    same_file = source.resolve() == output.resolve()
    if not same_file and source.exists() and output.exists():
        same_file = source.samefile(output)
    if same_file:
        raise ValueError("Output must not replace the source image.")
    if output.exists() and not overwrite:
        raise FileExistsError("Output already exists. Pass overwrite=True to replace it.")
    return output


def _openai_error(error: Exception, *, initializing: bool = False) -> RuntimeError:
    """Map SDK failures without retaining response text or credential values."""
    status = getattr(error, "status_code", None)
    if status == 401:
        message = "OpenAI authentication failed. Check OPENAI_API_KEY and your API account."
    elif status == 403:
        message = "OpenAI denied the request. Check model access and project permissions."
    elif status == 429:
        message = "OpenAI rate limit or quota reached. Check account quota and retry later."
    elif status == 404:
        message = "OpenAI image model was not found. Check the model name and account access."
    elif status in (400, 422):
        message = "OpenAI rejected the image-edit request. Check the image, model and generation options."
    elif isinstance(status, int) and status >= 500:
        message = "OpenAI service failed. Retry later."
    elif initializing:
        message = "Could not initialize the OpenAI client. Check OPENAI_API_KEY and SDK configuration."
    else:
        message = "OpenAI image-edit request failed. Check network connectivity and API configuration, then retry."
    return RuntimeError(message)


def _image_module() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Image backends require Pillow. Install it with: pip install Pillow") from exc
    return Image


def _metadata(
    backend: str,
    model: str,
    output_size: tuple[int, int],
    generation_seconds: float,
    *,
    revision: str = None,
    seed: int = None,
    steps: int = None,
) -> dict:
    return {
        "backend": backend,
        "model": model,
        "revision": revision,
        "seed": seed,
        "steps": steps,
        "generation_seconds": generation_seconds,
        "output_size": list(output_size),
    }


def generate_openai(
    source: Path,
    prompt: str,
    output: Path,
    *,
    model: str = "gpt-image-2",
    quality: str = "high",
    size: str = "auto",
    overwrite: bool = False,
) -> dict:
    """Edit with the public OpenAI SDK and save its decoded PNG response.

    The API model is selected by ``model``. This backend makes no claim that it
    reproduces an image produced by a separate, built-in image-generation tool.
    """
    output = _output_path(source, output, overwrite)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        raise RuntimeError("Set OPENAI_API_KEY to use the OpenAI backend.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the OpenAI Python SDK with: pip install openai") from exc

    image_module = _image_module()
    try:
        client = OpenAI(api_key=api_key)
    except Exception as exc:
        raise _openai_error(exc, initializing=True) from None
    with Path(source).open("rb") as source_file:
        started = time.perf_counter()
        try:
            response = client.images.edit(
                model=model,
                image=source_file,
                prompt=prompt,
                quality=quality,
                size=size,
                output_format="png",
                n=1,
            )
        except Exception as exc:
            raise _openai_error(exc) from None
        generation_seconds = time.perf_counter() - started

    data = getattr(response, "data", None)
    encoded = getattr(data[0], "b64_json", None) if data else None
    if not encoded:
        raise RuntimeError("The OpenAI image response did not contain b64_json image data.")
    try:
        png = base64.b64decode(encoded, validate=True)
    except (binascii.Error, TypeError, ValueError) as exc:
        raise RuntimeError("The OpenAI image response contained invalid base64 data.") from exc
    try:
        with image_module.open(io.BytesIO(png)) as generated:
            if generated.format != "PNG":
                raise RuntimeError("The OpenAI image response was not a PNG image.")
            generated.load()
            output_size = generated.size
    except (OSError, ValueError) as exc:
        raise RuntimeError("The OpenAI image response contained an unreadable PNG image.") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb" if overwrite else "xb") as stream:
        stream.write(png)
    return _metadata("openai", model, output_size, generation_seconds)


def generate_klein(
    source: Path,
    prompt: str,
    output: Path,
    *,
    model_id: str = KLEIN_MODEL,
    revision: str = KLEIN_REVISION,
    seed: int = 2026,
    steps: int = 4,
    device: str = "cuda",
    local_files_only: bool = False,
    overwrite: bool = False,
) -> dict:
    """Run frozen FLUX.2 Klein image editing at the source's native dimensions."""
    output = _output_path(source, output, overwrite)
    if steps < 1:
        raise ValueError("Klein inference steps must be at least 1.")
    if device != "cpu" and device != "cuda" and not device.startswith("cuda:"):
        raise ValueError("Klein supports 'cpu' or CUDA devices such as 'cuda:0'.")
    image_module = _image_module()
    with image_module.open(source) as opened:
        width, height = opened.size
        if width % 16 or height % 16:
            raise ValueError(
                "Klein requires source width and height to be multiples of 16 "
                "(received {} x {}). Choose a suitable crop; images are not resized automatically."
                .format(width, height)
            )
        source_image = opened.convert("RGB")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("The Klein backend requires PyTorch. Install torch for your device.") from exc
    uses_cuda = device.startswith("cuda")
    if uses_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable for the Klein backend. Use --device cpu or install "
            "a CUDA-enabled PyTorch build on a compatible GPU."
        )
    try:
        from diffusers import Flux2KleinPipeline
    except ImportError as exc:
        raise RuntimeError(
            "The Klein backend requires diffusers with Flux2KleinPipeline and its "
            "dependencies. Install the package's klein extra."
        ) from exc

    pipeline = Flux2KleinPipeline.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.bfloat16 if uses_cuda else torch.float32,
        local_files_only=local_files_only,
    )
    for component in pipeline.components.values():
        if isinstance(component, torch.nn.Module):
            component.eval()
            component.requires_grad_(False)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=device).manual_seed(seed)

    if uses_cuda:
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        response = pipeline(
            image=source_image,
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=1.0,
            generator=generator,
        )
    if uses_cuda:
        torch.cuda.synchronize(device)
    generation_seconds = time.perf_counter() - started
    images = getattr(response, "images", None)
    if not images:
        raise RuntimeError("The Klein pipeline returned no images.")
    generated = images[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb" if overwrite else "xb") as stream:
        generated.save(stream, format="PNG")
    return _metadata(
        "klein",
        model_id,
        generated.size,
        generation_seconds,
        revision=revision,
        seed=seed,
        steps=steps,
    )
