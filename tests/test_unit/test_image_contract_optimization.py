"""Frozen identity parity and adversarial ownership for image construction."""
import importlib.util
from pathlib import Path
import pickle
import sys
import threading

import numpy as np
import pytest

from ColorCorrectionPipeline.contracts import ImageData
import ColorCorrectionPipeline.contracts as contracts


@pytest.fixture(scope="module")
def baseline():
    # The baseline is frozen prior source, never the candidate implementation
    # used to calculate expected identities. Local audits may use the complete
    # independently qualified wheel instead of the extracted CI fixture.
    import os
    path = os.environ.get("CCP_IMAGE_BASELINE_CONTRACTS")
    if path is None:
        path = Path(__file__).parents[1] / "fixtures/image_contract_p9.py"
    path = Path(path)
    if not path.is_file():
        pytest.fail("missing frozen patch-9 image contract fixture")
    name = "ColorCorrectionPipeline._frozen_image_contracts"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.ImageData


def cases():
    rng = np.random.default_rng(612)
    yield rng.integers(0, 256, (4, 6, 3), dtype=np.uint8), None
    yield rng.integers(0, 65536, (4, 6, 3), dtype=np.uint16), None
    yield rng.integers(0, 65536, (4, 6, 3), dtype=np.uint16).astype(">u2"), None
    for dtype in ("float16", "float32", "float64", ">f8"):
        yield rng.random((4, 6, 3)).astype(dtype), None
    yield np.asfortranarray(rng.random((4, 6, 3))), None
    yield rng.random((8, 12, 3))[::2, ::2, ::-1], None
    yield rng.uniform(-2, 4, (4, 6, 3)), (-2, 4)
    yield np.array([[[-0., +0., 1.]]]), None
    yield np.array([[[-0., +0., 1.]]]), (-0., 1.)
    yield np.array([[[-0., +0., 1.]]], dtype=np.float32), (-0., 1.)


@pytest.mark.parametrize("orientation", range(1, 9))
@pytest.mark.parametrize("order", ["RGB", "BGR"])
def test_every_small_conversion_is_byte_and_identity_equal_to_frozen_wheel(baseline, orientation, order):
    for pixels, value_range in cases():
        kwargs = dict(channel_order=order, orientation=orientation, value_range=value_range,
                      provenance={"case": [1, {"flags": [True, None]}]})
        old, new = baseline.from_array(pixels, **kwargs), ImageData.from_array(pixels, **kwargs)
        assert new.pixels.tobytes() == old.pixels.tobytes()
        assert new.to_metadata() == old.to_metadata()
        assert new.digest == old.digest and new.pixel_digest == old.pixel_digest


@pytest.mark.parametrize("dtype", ["uint8", "uint16", ">u2", "float16", "float32", "float64", ">f8"])
def test_large_chunked_normalization_exact_for_layout_endianness_and_range(baseline, dtype):
    rng = np.random.default_rng(613)
    if np.dtype(dtype).kind == "u":
        native = np.dtype(dtype).newbyteorder("=")
        image = rng.integers(0, np.iinfo(dtype).max+1, (513, 683, 3), dtype=native).astype(dtype)
        value_range = None
    else:
        image = rng.uniform(-3, 10, (513, 683, 3)).astype(dtype)
        value_range = (-3, 10)
    # Non-contiguous permutation crosses normalization chunk boundaries and
    # rotates dimensions while retaining immutable output identity.
    image = image[::-1, ::-1]
    kwargs = dict(channel_order="BGR", orientation=7, value_range=value_range)
    old, new = baseline.from_array(image, **kwargs), ImageData.from_array(image, **kwargs)
    assert new.pixels.tobytes() == old.pixels.tobytes()
    assert new.to_metadata() == old.to_metadata()


@pytest.mark.parametrize("construction", [lambda array: ImageData(array), ImageData.from_array])
def test_caller_mutation_after_snapshot_validation_cannot_poison_owned_pixels(monkeypatch, construction):
    source = np.full((3, 4, 3), .4)
    original = contracts._image_bounds
    def mutate(array):
        result = original(array)
        source.fill(float("nan"))
        return result
    monkeypatch.setattr(contracts, "_image_bounds", mutate)
    image = construction(source)
    np.testing.assert_array_equal(image.pixels, np.full((3, 4, 3), .4))
    assert not image.pixels.flags.writeable
    with pytest.raises(ValueError):
        image.pixels.setflags(write=True)


def test_readonly_flag_on_mutable_owner_is_never_a_trust_signal():
    owner = np.full((3, 4, 3), .5)
    view = owner.view()
    view.setflags(write=False)
    direct, adapted = ImageData(view), ImageData.from_array(view)
    owner.fill(1)
    np.testing.assert_array_equal(direct.pixels, np.full((3, 4, 3), .5))
    np.testing.assert_array_equal(adapted.pixels, direct.pixels)
    buffer = bytearray(np.zeros((3, 4, 3)).tobytes())
    readonly = np.frombuffer(memoryview(buffer).toreadonly(), dtype=np.float64).reshape(3, 4, 3)
    captured = ImageData(readonly)
    buffer[:] = np.ones((3, 4, 3)).tobytes()
    assert not captured.pixels.any()


def test_reused_immutable_storage_has_independent_headers_and_no_large_slice_retention():
    original = ImageData.from_array(np.zeros((3, 4, 3)))
    copied = ImageData(original.pixels)
    assert copied.pixels is not original.pixels
    original.pixels.shape = (2, 6, 3)
    assert copied.shape == (3, 4, 3)
    pixel = copied.pixels[:1, :1]
    small = ImageData(pixel)
    root = small.pixels
    while isinstance(root, np.ndarray) and root.base is not None:
        root = root.base
    assert type(root) is bytes and len(root) == 24


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -.1, 1.1])
def test_even_immutable_byte_arrays_are_revalidated_on_public_entries(bad):
    values = np.array([[[.1, .2, bad]]])
    frozen = np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape)
    for construction in (ImageData, ImageData.from_array):
        with pytest.raises(ValueError):
            construction(frozen)
    payload = ImageData.from_array(np.zeros((1, 1, 3))).to_dict()
    payload["pixels"] = frozen
    with pytest.raises(ValueError):
        ImageData.from_dict(payload)


def test_pending_orientation_pickle_and_claimed_digests_are_still_checked(baseline):
    source = np.arange(72, dtype=np.uint16).reshape(4, 6, 3)
    options = dict(orientation=6, apply_orientation=False)
    current, old = ImageData.from_array(source, **options), baseline.from_array(source, **options)
    assert current.with_orientation_applied().to_metadata() == old.with_orientation_applied().to_metadata()
    restored = pickle.loads(pickle.dumps(current))
    assert restored.digest == current.digest and not restored.pixels.flags.writeable
    data = current.to_dict()
    data["pixel_digest"] = "0"*64
    with pytest.raises(ValueError, match="digest"):
        ImageData.from_dict(data)


def test_small_images_need_no_executor(monkeypatch):
    monkeypatch.setattr(contracts, "ThreadPoolExecutor", lambda **kwargs: pytest.fail("small image spawned SHA worker"))
    ImageData.from_array(np.zeros((4, 6, 3)))


@pytest.mark.parametrize("where", ["worker", "main"])
def test_parallel_hash_error_propagates_and_joins_workers(monkeypatch, where):
    original = contracts._hash_parts
    def failing(metadata, arrays=()):
        target = ("shape" not in metadata) if where == "worker" else ("shape" in metadata)
        if target:
            raise RuntimeError("injected hash failure")
        return original(metadata, arrays)
    monkeypatch.setattr(contracts, "_hash_parts", failing)
    with pytest.raises(RuntimeError, match="injected hash failure"):
        ImageData.from_array(np.zeros((600, 600, 3)))
    assert not any(thread.name.startswith("ccp-image-sha256") for thread in threading.enumerate())


def test_missing_thread_resources_preserve_sequential_identities(monkeypatch, baseline):
    class NoWorker:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, *args, **kwargs):
            raise RuntimeError("can't start new thread")
    monkeypatch.setattr(contracts, "ThreadPoolExecutor", NoWorker)
    pixels = np.zeros((600, 600, 3))
    assert ImageData.from_array(pixels).to_metadata() == baseline.from_array(pixels).to_metadata()


@pytest.mark.parametrize("value_range", [(-1e308, 1e308), (-0., 1.), (0., np.nextafter(0., 1.))])
def test_extreme_ranges_preserve_old_result_or_rejection(baseline, value_range):
    array = np.array([[[value_range[0], 0., value_range[1]]]])
    with np.errstate(all="ignore"):
        try:
            old = baseline.from_array(array, value_range=value_range)
        except ValueError:
            with pytest.raises(ValueError):
                ImageData.from_array(array, value_range=value_range)
        else:
            new = ImageData.from_array(array, value_range=value_range)
            assert new.pixels.tobytes() == old.pixels.tobytes()
            assert new.to_metadata() == old.to_metadata()
