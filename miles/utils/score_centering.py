import math
from array import array
from collections.abc import Sequence
from dataclasses import InitVar, dataclass, field

import torch


def score_centering_enabled(args) -> bool:
    return bool(getattr(args, "use_score_centering", False))


@dataclass(frozen=True, eq=False)
class RolloutScoreCenteringHead:
    """Sampler top-probability rows used by modeled-tail score centering.

    Rows are stored in CSR form so forced environment tokens can carry their
    exact singleton distribution while sampled rows retain the configured
    top-k head. Logprobs remain normalized over the sampler's full vocabulary;
    they are never renormalized over the retained head.
    """

    ids: InitVar[Sequence[int] | torch.Tensor]
    offsets: InitVar[Sequence[int] | torch.Tensor]
    logprobs: InitVar[Sequence[float] | torch.Tensor]
    _ids: torch.Tensor = field(init=False, repr=False)
    _offsets: torch.Tensor = field(init=False, repr=False)
    _logprobs: torch.Tensor = field(init=False, repr=False)

    def __post_init__(
        self,
        ids: Sequence[int] | torch.Tensor,
        offsets: Sequence[int] | torch.Tensor,
        logprobs: Sequence[float] | torch.Tensor,
    ) -> None:
        owned_ids = _owned_integer_tensor(ids, torch.int32)
        owned_offsets = _owned_integer_tensor(offsets, torch.long)
        owned_logprobs = _owned_float_tensor(logprobs)
        _validate_head(owned_ids, owned_offsets, owned_logprobs)
        object.__setattr__(self, "_ids", owned_ids)
        object.__setattr__(self, "_offsets", owned_offsets)
        object.__setattr__(self, "_logprobs", owned_logprobs)

    @classmethod
    def from_rows(
        cls,
        id_rows: Sequence[Sequence[int]],
        logprob_rows: Sequence[Sequence[float]],
    ) -> "RolloutScoreCenteringHead":
        if len(id_rows) != len(logprob_rows):
            raise ValueError("score-centering id and logprob rows must align")
        ids = []
        logprobs = []
        offsets = [0]
        for row_ids, row_logprobs in zip(id_rows, logprob_rows, strict=True):
            if len(row_ids) != len(row_logprobs):
                raise ValueError("score-centering ids and logprobs must align within every row")
            ids.extend(row_ids)
            logprobs.extend(row_logprobs)
            offsets.append(len(ids))
        return cls(ids=ids, offsets=offsets, logprobs=logprobs)

    @classmethod
    def concatenate(
        cls,
        heads: Sequence["RolloutScoreCenteringHead"],
    ) -> "RolloutScoreCenteringHead":
        if not heads:
            return cls(ids=[], offsets=[0], logprobs=[])
        if len(heads) == 1:
            return heads[0]

        ids = torch.cat([head._ids for head in heads])
        logprobs = torch.cat([head._logprobs for head in heads])
        offsets = [torch.zeros(1, dtype=torch.long)]
        id_count = 0
        for head in heads:
            offsets.append(head._offsets[1:] + id_count)
            id_count += head._ids.numel()
        return cls(ids=ids, offsets=torch.cat(offsets), logprobs=logprobs)

    def __len__(self) -> int:
        return self._offsets.numel() - 1

    def prefix(self, response_length: int) -> "RolloutScoreCenteringHead":
        if not 0 <= response_length <= len(self):
            raise ValueError(f"score-centering prefix length must be in [0, {len(self)}]")
        if response_length == len(self):
            return self
        id_count = self._offsets[response_length]
        return type(self)(
            ids=self._ids[:id_count],
            offsets=self._offsets[: response_length + 1],
            logprobs=self._logprobs[:id_count],
        )

    def _as_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Borrow the private CSR tensors for immediate read-only transport."""
        return self._ids, self._offsets, self._logprobs

    def _select_rows(
        self,
        token_indices: Sequence[int] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = _integer_indices(token_indices)
        if torch.any(indices < 0) or torch.any(indices >= len(self)):
            raise ValueError(f"score-centering response indices must be in [0, {len(self)})")
        lengths = self._offsets[indices + 1] - self._offsets[indices]
        if indices.numel() == 0:
            return self._ids.new_empty(0), self._logprobs.new_empty(0), lengths
        id_parts = [self._ids[self._offsets[index] : self._offsets[index + 1]] for index in indices]
        logprob_parts = [self._logprobs[self._offsets[index] : self._offsets[index + 1]] for index in indices]
        ids = id_parts[0] if len(id_parts) == 1 else torch.cat(id_parts)
        logprobs = logprob_parts[0] if len(logprob_parts) == 1 else torch.cat(logprob_parts)
        return ids, logprobs, lengths


def _owned_integer_tensor(values: Sequence[int] | torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        _validate_integer_tensor(values)
        return values.detach().to(device="cpu", dtype=dtype, copy=True)
    if len(values) == 0:
        return torch.empty(0, dtype=dtype, device="cpu")
    if dtype == torch.int32 and len(values) > 0:
        storage = array("i", values)
        return torch.frombuffer(storage, dtype=torch.int32)
    tensor = torch.as_tensor(values, device="cpu")
    _validate_integer_tensor(tensor)
    return tensor.to(dtype=dtype, copy=True)


def _owned_float_tensor(values: Sequence[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        if values.ndim != 1 or not torch.is_floating_point(values):
            raise ValueError("score-centering logprobs must be a one-dimensional floating-point tensor")
        return values.detach().to(device="cpu", dtype=torch.float32, copy=True)
    if len(values) > 0:
        storage = array("f", values)
        return torch.frombuffer(storage, dtype=torch.float32)
    return torch.empty(0, dtype=torch.float32, device="cpu")


def _integer_indices(values: Sequence[int] | torch.Tensor) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        tensor = values.detach().cpu()
    elif isinstance(values, range):
        tensor = torch.arange(values.start, values.stop, values.step, device="cpu")
    elif len(values) == 0:
        tensor = torch.empty(0, dtype=torch.long, device="cpu")
    else:
        tensor = torch.as_tensor(values, device="cpu")
    _validate_integer_tensor(tensor)
    return tensor.to(torch.long)


def _validate_integer_tensor(tensor: torch.Tensor) -> None:
    if tensor.ndim != 1 or tensor.dtype == torch.bool or torch.is_floating_point(tensor) or torch.is_complex(tensor):
        raise ValueError("score-centering ids, offsets, and response indices must be one-dimensional integers")


def _validate_head(ids: torch.Tensor, offsets: torch.Tensor, logprobs: torch.Tensor) -> None:
    if offsets.numel() == 0 or offsets[0] != 0 or offsets[-1] != ids.numel():
        raise ValueError("score-centering offsets must start at zero and end at the flattened id count")
    if torch.any(offsets[1:] <= offsets[:-1]):
        raise ValueError("every score-centering row must contain at least one token")
    if logprobs.numel() != ids.numel():
        raise ValueError("score-centering logprobs must align with flattened token ids")
    if torch.any(ids < 0):
        raise ValueError("score-centering token ids must be nonnegative")
    if not torch.isfinite(logprobs).all() or torch.any(logprobs > 1e-6):
        raise ValueError("score-centering logprobs must be finite nonpositive values")
    for start, end in zip(offsets[:-1].tolist(), offsets[1:].tolist(), strict=True):
        row_ids = ids[start:end]
        if torch.unique(row_ids).numel() != row_ids.numel():
            raise ValueError("score-centering token ids must be distinct within each row")
        mass = logprobs[start:end].double().exp().sum().item()
        if not 0.0 < mass <= 1.0001 or not math.isfinite(mass):
            raise ValueError("score-centering logprobs must describe a probability subdistribution")
