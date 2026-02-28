import os
from pathlib import Path


def test_import_package():
    import vggt_qwen3  # noqa: F401


def test_checkpoint_loader_behaviour(tmp_path: Path):
    from vggt_qwen3.inference.qa_inference import load_checkpoint_if_available

    class Dummy:
        def __init__(self):
            import torch

            self._param = torch.nn.Parameter(torch.zeros(1))

        def parameters(self):
            return [self._param]

    model = Dummy()

    # No checkpoint directory: allowed, uses base weights.
    info = load_checkpoint_if_available(model, None, allow_base_fallback=False)
    assert info["used_base_weights"] is True

    # Missing directory without fallback: should raise.
    missing = tmp_path / "missing_ckpt"
    try:
        load_checkpoint_if_available(model, str(missing), allow_base_fallback=False)
    except FileNotFoundError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected FileNotFoundError for missing checkpoint dir.")

    # Missing directory with fallback: no exception, but base weights used.
    info2 = load_checkpoint_if_available(model, str(missing), allow_base_fallback=True)
    assert info2["used_base_weights"] is True

