"""Tensor storage lifetime contracts for persistent runtime structures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import mlx.core as mx
import numpy as np


class TensorOwnership(str, Enum):
    OWNED = "owned"
    BORROWED_STABLE = "borrowed-stable"
    BORROWED_EPHEMERAL = "borrowed-ephemeral"


class TensorLayout(str, Enum):
    UNKNOWN = "unknown"
    ROW_MAJOR_CONTIGUOUS = "row-major-contiguous"


class TensorOwnershipError(ValueError):
    """Raised when a storage lifetime cannot satisfy a resident boundary."""


@dataclass(frozen=True)
class TensorLease:
    value: mx.array
    ownership: TensorOwnership
    layout: TensorLayout = TensorLayout.UNKNOWN
    owner: object | None = None

    def __post_init__(self) -> None:
        if self.ownership is TensorOwnership.BORROWED_STABLE and self.owner is None:
            raise TensorOwnershipError(
                "borrowed-stable storage requires an owner retained for its lifetime"
            )


@dataclass(frozen=True)
class ResidentTensor:
    value: mx.array
    ownership: TensorOwnership
    layout: TensorLayout
    owner: object | None = None

    def __post_init__(self) -> None:
        if self.ownership is TensorOwnership.BORROWED_EPHEMERAL:
            raise TensorOwnershipError(
                "resident structures cannot retain borrowed-ephemeral storage"
            )
        if self.layout is not TensorLayout.ROW_MAJOR_CONTIGUOUS:
            raise TensorOwnershipError(
                "resident tensors require the row-major contiguous layout contract"
            )
        if self.ownership is TensorOwnership.BORROWED_STABLE and self.owner is None:
            raise TensorOwnershipError(
                "borrowed-stable resident storage must retain its owner"
            )


def owned_tensor(
    value: mx.array, *, layout: TensorLayout = TensorLayout.UNKNOWN
) -> TensorLease:
    """Declare storage created and lifetime-controlled by this runtime."""
    return TensorLease(value, TensorOwnership.OWNED, layout)


def borrowed_stable_tensor(
    value: mx.array,
    *,
    owner: object,
    layout: TensorLayout = TensorLayout.UNKNOWN,
) -> TensorLease:
    """Borrow storage whose owner is retained for the resident lifetime."""
    return TensorLease(value, TensorOwnership.BORROWED_STABLE, layout, owner)


def borrowed_ephemeral_tensor(
    value: mx.array,
    *,
    owner: object | None = None,
    layout: TensorLayout = TensorLayout.UNKNOWN,
) -> TensorLease:
    """Borrow staging/scratch storage that may be reused after the load step."""
    return TensorLease(value, TensorOwnership.BORROWED_EPHEMERAL, layout, owner)


_COPY_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    if (index >= N) return;
    output[index] = input[index];
"""


_copy_kernel = (
    mx.fast.metal_kernel(
        name="glm53_owned_tensor_copy",
        input_names=["input"],
        output_names=["output"],
        source=_COPY_SOURCE,
    )
    if mx.metal.is_available()
    else None
)


def _row_major(value: mx.array) -> mx.array:
    return mx.contiguous(value, allow_col_major=False)


def _owned_copy(value: mx.array) -> mx.array:
    value = _row_major(value)
    if _copy_kernel is None:
        # CPU fallback is used only where MLX/Metal is unavailable. The private
        # NumPy owner is retained by the resulting MLX array and is not exposed
        # to the caller that supplied the borrowed storage.
        private = np.array(np.asarray(value), copy=True, order="C")
        output = mx.array(private)
    else:
        count = int(value.size)
        threads = 256
        output = _copy_kernel(
            inputs=[value],
            template=[("N", count)],
            grid=((count + threads - 1) // threads * threads, 1, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[value.shape],
            output_dtypes=[value.dtype],
        )[0]
    mx.eval(output)
    return output


def require_resident(lease: TensorLease | ResidentTensor) -> ResidentTensor:
    """Validate a resident boundary without silently copying ephemeral data."""
    if isinstance(lease, ResidentTensor):
        return lease
    if not isinstance(lease, TensorLease):
        raise TypeError(
            "resident tensors require an explicit TensorLease; bare arrays "
            "have no storage lifetime contract"
        )
    if lease.ownership is TensorOwnership.BORROWED_EPHEMERAL:
        raise TensorOwnershipError(
            "borrowed-ephemeral storage must call materialize_owned() before "
            "crossing a resident boundary"
        )
    value = _row_major(lease.value)
    return ResidentTensor(
        value,
        lease.ownership,
        TensorLayout.ROW_MAJOR_CONTIGUOUS,
        lease.owner,
    )


def materialize_owned(lease: TensorLease | ResidentTensor) -> ResidentTensor:
    """Create storage independent of every borrowed source and source owner."""
    if isinstance(lease, ResidentTensor) and lease.ownership is TensorOwnership.OWNED:
        return lease
    if not isinstance(lease, (TensorLease, ResidentTensor)):
        raise TypeError("materialize_owned() requires an explicit storage contract")
    if lease.ownership is TensorOwnership.OWNED:
        value = _row_major(lease.value)
        mx.eval(value)
    else:
        value = _owned_copy(lease.value)
    return ResidentTensor(
        value,
        TensorOwnership.OWNED,
        TensorLayout.ROW_MAJOR_CONTIGUOUS,
    )


def resident_concatenate(
    leases: Iterable[TensorLease | ResidentTensor], *, axis: int
) -> ResidentTensor:
    """Build an owned, evaluated fusion from non-ephemeral source tensors."""
    residents = [require_resident(lease) for lease in leases]
    if not residents:
        raise ValueError("cannot concatenate an empty resident tensor list")
    value = _row_major(mx.concatenate([item.value for item in residents], axis=axis))
    mx.eval(value)
    return ResidentTensor(
        value,
        TensorOwnership.OWNED,
        TensorLayout.ROW_MAJOR_CONTIGUOUS,
    )


def storage_descriptor(value: TensorLease | ResidentTensor) -> dict[str, str]:
    return {
        "ownership": value.ownership.value,
        "layout": value.layout.value,
    }
