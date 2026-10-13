"""SDK boundary tests only: these tests do not run real model inference."""

import base64
from contextlib import contextmanager
import importlib
import io
import os
from pathlib import Path
import sys
import tempfile
import traceback
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from PIL import Image

from aigc_clear import backends


METADATA_KEYS = {
    "backend", "model", "revision", "seed", "steps", "generation_seconds", "output_size"
}


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.png"
        self.output = self.root / "nested" / "raw.png"
        Image.new("RGB", (32, 48), "blue").save(self.source)

    def png_bytes(self, size=(32, 48)):
        buffer = io.BytesIO()
        Image.new("RGB", size, "red").save(buffer, format="PNG")
        return buffer.getvalue()

    def openai_module(self, encoded):
        module = ModuleType("openai")
        client = mock.Mock()
        client.images.edit.return_value = SimpleNamespace(
            data=[SimpleNamespace(b64_json=encoded)]
        )
        module.OpenAI = mock.Mock(return_value=client)
        return module, client

    def test_module_import_does_not_load_optional_model_sdks(self):
        with mock.patch.dict(sys.modules, {"openai": None, "torch": None, "diffusers": None}):
            importlib.reload(backends)

    def test_openai_requires_api_key_before_loading_sdk(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.dict(
            sys.modules, {"openai": None}
        ):
            with self.assertRaisesRegex(RuntimeError, "Set OPENAI_API_KEY"):
                backends.generate_openai(self.source, "clean hair", self.output)
        self.assertFalse(self.output.exists())

    def test_openai_missing_sdk_has_actionable_error(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), mock.patch.dict(
            sys.modules, {"openai": None}
        ):
            with self.assertRaisesRegex(RuntimeError, "pip install openai"):
                backends.generate_openai(self.source, "clean hair", self.output)

    def test_openai_refuses_existing_output_before_loading_sdk(self):
        self.output.parent.mkdir()
        self.output.write_bytes(b"keep existing output")
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.dict(
            sys.modules, {"openai": None}
        ):
            with self.assertRaisesRegex(FileExistsError, "overwrite=True"):
                backends.generate_openai(self.source, "clean hair", self.output)
        self.assertEqual(self.output.read_bytes(), b"keep existing output")

    def test_openai_explicit_overwrite_replaces_existing_output(self):
        self.output.parent.mkdir()
        self.output.write_bytes(b"old output")
        png = self.png_bytes()
        module, _ = self.openai_module(base64.b64encode(png).decode("ascii"))
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), mock.patch.dict(
            sys.modules, {"openai": module}
        ):
            backends.generate_openai(self.source, "clean hair", self.output, overwrite=True)
        self.assertEqual(self.output.read_bytes(), png)

    def test_backends_always_refuse_source_path_even_with_overwrite(self):
        original = self.source.read_bytes()
        for generate in (backends.generate_openai, backends.generate_klein):
            for overwrite in (False, True):
                with self.subTest(backend=generate.__name__, overwrite=overwrite):
                    with mock.patch.dict(sys.modules, {"openai": None, "torch": None, "diffusers": None}):
                        with self.assertRaisesRegex(ValueError, "source image"):
                            generate(self.source, "clean hair", self.source, overwrite=overwrite)
                    self.assertEqual(self.source.read_bytes(), original)

    def test_backends_refuse_resolved_alias_of_source(self):
        alias = self.root / "unused" / ".." / self.source.name
        for generate in (backends.generate_openai, backends.generate_klein):
            with self.subTest(backend=generate.__name__):
                with self.assertRaisesRegex(ValueError, "source image"):
                    generate(self.source, "clean hair", alias, overwrite=True)

    def test_backends_refuse_hard_link_to_source(self):
        alias = self.root / "linked.png"
        try:
            os.link(self.source, alias)
        except OSError:
            self.skipTest("The test filesystem does not support hard links")
        for generate in (backends.generate_openai, backends.generate_klein):
            with self.subTest(backend=generate.__name__):
                with self.assertRaisesRegex(ValueError, "source image"):
                    generate(self.source, "clean hair", alias, overwrite=True)

    def test_openai_initialization_and_request_errors_hide_sdk_text_and_traceback(self):
        private_message = "credential-marker-test-key and original SDK response"
        categories = [
            (401, "authentication"), (403, "permissions"), (429, "quota"),
            (404, "model name"), (400, "generation options"), (422, "generation options"),
            (503, "Retry later"), (None, "network connectivity"),
        ]
        for stage in ("initialize", "request"):
            for status, expected in categories:
                with self.subTest(stage=stage, status=status):
                    module, client = self.openai_module(None)
                    sdk_error = RuntimeError(private_message)
                    sdk_error.status_code = status
                    if stage == "initialize":
                        module.OpenAI.side_effect = sdk_error
                        if status is None:
                            expected = "SDK configuration"
                    else:
                        client.images.edit.side_effect = sdk_error
                    with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), mock.patch.dict(
                        sys.modules, {"openai": module}
                    ):
                        try:
                            backends.generate_openai(self.source, "clean hair", self.output)
                        except RuntimeError as error:
                            self.assertIn(expected, str(error))
                            self.assertIsNone(error.__cause__)
                            self.assertTrue(error.__suppress_context__)
                            rendered = "".join(traceback.format_exception(type(error), error, error.__traceback__))
                            self.assertNotIn(private_message, rendered)
                            self.assertNotIn("test-key", rendered)
                            self.assertNotIn("During handling", rendered)
                        else:
                            self.fail("Expected a sanitized OpenAI error")
                    if stage == "initialize":
                        client.images.edit.assert_not_called()
                    self.assertFalse(self.output.exists())

    def test_openai_exclusive_write_preserves_file_created_during_request(self):
        png = self.png_bytes()
        module, client = self.openai_module(base64.b64encode(png).decode("ascii"))
        response = client.images.edit.return_value

        def edit(**kwargs):
            self.output.parent.mkdir()
            self.output.write_bytes(b"written by another process")
            return response

        client.images.edit.side_effect = edit
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), mock.patch.dict(
            sys.modules, {"openai": module}
        ):
            with self.assertRaises(FileExistsError):
                backends.generate_openai(self.source, "clean hair", self.output)
        self.assertEqual(self.output.read_bytes(), b"written by another process")

    def test_openai_sdk_request_and_raw_png_preserve_dimensions(self):
        png = self.png_bytes((64, 64))
        module, client = self.openai_module(base64.b64encode(png).decode("ascii"))
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), mock.patch.dict(
            sys.modules, {"openai": module}
        ), mock.patch.object(backends.time, "perf_counter", side_effect=[10.0, 12.5]):
            metadata = backends.generate_openai(
                self.source, "clean hair", self.output,
                model="custom-image-model", quality="medium", size="1024x1024",
            )
        module.OpenAI.assert_called_once_with(api_key="test-key")
        kwargs = client.images.edit.call_args.kwargs
        self.assertEqual(kwargs["model"], "custom-image-model")
        self.assertEqual(kwargs["prompt"], "clean hair")
        self.assertEqual(kwargs["quality"], "medium")
        self.assertEqual(kwargs["size"], "1024x1024")
        self.assertEqual(kwargs["output_format"], "png")
        self.assertEqual(kwargs["n"], 1)
        self.assertTrue(kwargs["image"].closed)
        self.assertEqual(self.output.read_bytes(), png)
        self.assertEqual(set(metadata), METADATA_KEYS)
        self.assertEqual(metadata["output_size"], [64, 64])
        self.assertEqual(metadata["generation_seconds"], 2.5)
        self.assertIsNone(metadata["seed"])
        self.assertNotIn("test-key", str(metadata))
        self.assertNotIn(str(self.root), str(metadata))

    def test_openai_rejects_missing_invalid_and_non_png_responses(self):
        jpeg = io.BytesIO()
        Image.new("RGB", (32, 48)).save(jpeg, format="JPEG")
        for encoded, expected in [
            (None, "did not contain b64_json"),
            ("not valid base64!", "invalid base64"),
            (base64.b64encode(b"broken png").decode("ascii"), "unreadable PNG"),
            (base64.b64encode(jpeg.getvalue()).decode("ascii"), "not a PNG"),
        ]:
            with self.subTest(expected=expected):
                module, _ = self.openai_module(encoded)
                with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), mock.patch.dict(
                    sys.modules, {"openai": module}
                ):
                    with self.assertRaisesRegex(RuntimeError, expected):
                        backends.generate_openai(self.source, "clean hair", self.output)
                self.assertFalse(self.output.exists())

    def fake_klein_modules(self, cuda_available=True, output_size=(32, 48)):
        state = {"inference": False, "call_in_inference": False}

        class FakeModule:
            def __init__(self):
                self.training = True
                self.parameter = SimpleNamespace(requires_grad=True)

            def eval(self):
                self.training = False
                return self

            def requires_grad_(self, value):
                self.parameter.requires_grad = value
                return self

        torch = ModuleType("torch")
        torch.nn = SimpleNamespace(Module=FakeModule)
        torch.bfloat16 = object()
        torch.float32 = object()
        torch.cuda = SimpleNamespace(
            is_available=mock.Mock(return_value=cuda_available), synchronize=mock.Mock()
        )
        generator = mock.Mock()
        generator.manual_seed.return_value = generator
        torch.Generator = mock.Mock(return_value=generator)

        @contextmanager
        def inference_mode():
            state["inference"] = True
            try:
                yield
            finally:
                state["inference"] = False

        torch.inference_mode = inference_mode
        neural_components = [FakeModule(), FakeModule(), FakeModule()]
        pipeline = mock.Mock()
        pipeline.components = dict(zip(["transformer", "vae", "text_encoder"], neural_components))
        pipeline.components["scheduler"] = object()
        pipeline.to.return_value = pipeline

        def run(**kwargs):
            state["call_in_inference"] = state["inference"]
            return SimpleNamespace(images=[Image.new("RGB", output_size, "red")])

        pipeline.side_effect = run
        diffusers = ModuleType("diffusers")
        diffusers.Flux2KleinPipeline = SimpleNamespace(
            from_pretrained=mock.Mock(return_value=pipeline)
        )
        return torch, diffusers, pipeline, neural_components, state

    def test_klein_native_dimensions_seed_frozen_modules_and_inference_mode(self):
        torch, diffusers, pipeline, components, state = self.fake_klein_modules()
        with mock.patch.dict(sys.modules, {"torch": torch, "diffusers": diffusers}), mock.patch.object(
            backends.time, "perf_counter", side_effect=[3.0, 4.25]
        ):
            metadata = backends.generate_klein(
                self.source, "clean hair", self.output,
                device="cuda:0", seed=7, steps=6, local_files_only=True,
            )
        diffusers.Flux2KleinPipeline.from_pretrained.assert_called_once_with(
            backends.KLEIN_MODEL, revision=backends.KLEIN_REVISION,
            torch_dtype=torch.bfloat16, local_files_only=True,
        )
        pipeline.to.assert_called_once_with("cuda:0")
        torch.Generator.assert_called_once_with(device="cuda:0")
        torch.Generator.return_value.manual_seed.assert_called_once_with(7)
        kwargs = pipeline.call_args.kwargs
        self.assertEqual(kwargs["image"].size, (32, 48))
        self.assertEqual(kwargs["width"], 32)
        self.assertEqual(kwargs["height"], 48)
        self.assertEqual(kwargs["num_inference_steps"], 6)
        self.assertEqual(kwargs["guidance_scale"], 1.0)
        self.assertTrue(state["call_in_inference"])
        self.assertFalse(state["inference"])
        for component in components:
            self.assertFalse(component.training)
            self.assertFalse(component.parameter.requires_grad)
        self.assertEqual(torch.cuda.synchronize.call_args_list, [mock.call("cuda:0"), mock.call("cuda:0")])
        self.assertEqual(set(metadata), METADATA_KEYS)
        self.assertEqual(metadata["revision"], backends.KLEIN_REVISION)
        self.assertEqual(metadata["seed"], 7)
        self.assertEqual(metadata["steps"], 6)
        self.assertEqual(metadata["generation_seconds"], 1.25)
        self.assertNotIn(str(self.root), str(metadata))
        with Image.open(self.output) as saved:
            self.assertEqual(saved.size, (32, 48))

    def test_klein_cpu_uses_float32_and_preserves_raw_output_size(self):
        torch, diffusers, _, _, _ = self.fake_klein_modules(
            cuda_available=False, output_size=(64, 64)
        )
        with mock.patch.dict(sys.modules, {"torch": torch, "diffusers": diffusers}):
            metadata = backends.generate_klein(
                self.source, "clean hair", self.output,
                model_id="custom-klein", revision="pinned-revision", device="cpu",
            )
        diffusers.Flux2KleinPipeline.from_pretrained.assert_called_once_with(
            "custom-klein", revision="pinned-revision", torch_dtype=torch.float32,
            local_files_only=False,
        )
        torch.cuda.is_available.assert_not_called()
        torch.cuda.synchronize.assert_not_called()
        self.assertEqual(metadata["output_size"], [64, 64])
        with Image.open(self.output) as saved:
            self.assertEqual(saved.size, (64, 64))

    def test_klein_unavailable_cuda_does_not_load_model(self):
        torch, diffusers, _, _, _ = self.fake_klein_modules(cuda_available=False)
        with mock.patch.dict(sys.modules, {"torch": torch, "diffusers": diffusers}):
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                backends.generate_klein(self.source, "clean hair", self.output)
        diffusers.Flux2KleinPipeline.from_pretrained.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_klein_refuses_existing_output_before_loading_model(self):
        self.output.parent.mkdir()
        self.output.write_bytes(b"keep existing output")
        with mock.patch.dict(sys.modules, {"torch": None, "diffusers": None}):
            with self.assertRaisesRegex(FileExistsError, "overwrite=True"):
                backends.generate_klein(self.source, "clean hair", self.output)
        self.assertEqual(self.output.read_bytes(), b"keep existing output")

    def test_klein_explicit_overwrite_replaces_existing_output(self):
        self.output.parent.mkdir()
        self.output.write_bytes(b"old output")
        torch, diffusers, _, _, _ = self.fake_klein_modules()
        with mock.patch.dict(sys.modules, {"torch": torch, "diffusers": diffusers}):
            backends.generate_klein(self.source, "clean hair", self.output, overwrite=True)
        with Image.open(self.output) as saved:
            self.assertEqual(saved.size, (32, 48))
            self.assertEqual(saved.getpixel((0, 0)), (255, 0, 0))

    def test_klein_exclusive_write_preserves_file_created_during_inference(self):
        torch, diffusers, pipeline, _, _ = self.fake_klein_modules()

        def run(**kwargs):
            self.output.parent.mkdir()
            self.output.write_bytes(b"written by another process")
            return SimpleNamespace(images=[Image.new("RGB", (32, 48), "red")])

        pipeline.side_effect = run
        with mock.patch.dict(sys.modules, {"torch": torch, "diffusers": diffusers}):
            with self.assertRaises(FileExistsError):
                backends.generate_klein(self.source, "clean hair", self.output)
        self.assertEqual(self.output.read_bytes(), b"written by another process")

    def test_klein_rejects_non_multiple_dimensions_before_loading_model(self):
        Image.new("RGB", (33, 48)).save(self.source)
        with mock.patch.dict(sys.modules, {"torch": None, "diffusers": None}):
            with self.assertRaisesRegex(ValueError, "multiples of 16"):
                backends.generate_klein(self.source, "clean hair", self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
