import importlib.metadata
from types import SimpleNamespace

import pytest
import torch

from miles.utils import nvcomp


@pytest.fixture
def cuda13(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.version, "cuda", "13.0")


def test_a_missing_wheel_is_refused_with_the_command_that_installs_it(
    cuda13: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(name: str):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", missing)

    with pytest.raises(RuntimeError, match=r"pip install --no-deps nvidia-libnvcomp-cu13==5\.3\.0\.16"):
        nvcomp.nvcomp_distribution()


@pytest.mark.parametrize(
    "version, accepted", [("5.3.0.16", True), ("5.4.1", True), ("5.2.0.13", False), ("6.0.0.12", False)]
)
def test_only_the_5_3_c_api_is_accepted(
    cuda13: None, monkeypatch: pytest.MonkeyPatch, version: str, accepted: bool
) -> None:
    """6.0 renamed the temp-size query and gave it a stream argument; the binding calls the 5.3 names."""
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: SimpleNamespace(name=name, version=version))

    if accepted:
        assert nvcomp.nvcomp_distribution().version == version
    else:
        with pytest.raises(RuntimeError, match=version):
            nvcomp.nvcomp_distribution()


def test_a_cpu_build_of_torch_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.version, "cuda", None)

    with pytest.raises(RuntimeError, match="CUDA build"):
        nvcomp.nvcomp_distribution()


@pytest.mark.parametrize(
    "algorithm, device, message",
    [("lz4", "cuda:0", "algorithm must be one of"), ("zstd", "cuda", "explicit CUDA device"), ("zstd", "cpu", "CUDA")],
)
def test_an_unsupported_algorithm_or_device_is_refused_before_loading_nvcomp(
    algorithm: str, device: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        nvcomp.NvcompCompressor(algorithm, device)
