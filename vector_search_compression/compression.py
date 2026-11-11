"""Packed weight representation, factor construction, and byte allocation."""

from dataclasses import dataclass
import math

import numpy as np


GROUP_SIZE = 64
WIDTHS = (2, 3, 4, 8)
METHODS = ("rtn", "gptq", "gptaq", "awq")


def matrix_payload(shape, bits, method, rank=0):
    """Bytes for the original matrix, or for both factors when rank > 0."""
    rows, columns = shape
    if method not in METHODS or bits not in (*WIDTHS, 16):
        raise ValueError("Unknown method or weight precision")
    if min(rows, columns) < 1 or not 0 <= rank <= min(rows, columns):
        raise ValueError("Invalid matrix shape or rank")

    parts = ((rank, columns), (rows, rank)) if rank else (shape,)
    if bits == 16:
        return sum(2 * rows * columns for rows, columns in parts)

    if any(columns % GROUP_SIZE for _, columns in parts):
        raise ValueError("Packed weights require complete groups of 64 input channels")
    return sum(
        rows * columns * bits // 8
        + 4 * rows * (columns // GROUP_SIZE)
        + (4 * columns if method == "awq" else 0)
        for rows, columns in parts
    )


@dataclass(frozen=True)
class PackedWeight:
    shape: tuple[int, int]
    bits: int
    codes: np.ndarray
    scales: np.ndarray
    input_scale: np.ndarray | None = None

    @property
    def nbytes(self):
        inputs = 0 if self.input_scale is None else self.input_scale.nbytes
        return self.codes.nbytes + self.scales.nbytes + inputs

    def decode(self):
        """Decode integer codes and fold any permanent input scale into the weights."""
        rows, columns = self.shape
        if self.bits == 8:
            values = self.codes
        else:
            chunks = self.codes.astype(np.uint32).reshape(rows, -1, self.bits)
            offsets = np.arange(self.bits, dtype=np.uint32) * 8
            words = np.bitwise_or.reduce(chunks << offsets, axis=-1)
            shifts = np.arange(8, dtype=np.uint32) * self.bits
            values = (words[..., None] >> shifts) & ((1 << self.bits) - 1)
        signed = values.astype(np.float32).reshape(rows, -1, GROUP_SIZE)
        signed -= 1 << (self.bits - 1)
        signed *= self.scales[..., None]
        weights = signed.reshape(rows, columns)
        if self.input_scale is not None:
            weights /= self.input_scale[None, :]
        return weights


def pack_weight(codes, scales, bits, input_scale=None):
    """Pack row-major integer codes and the fitted scales from a quantizer."""
    codes = np.asarray(codes)
    scales = np.asarray(scales, dtype=np.float32)
    if codes.ndim != 2 or not np.issubdtype(codes.dtype, np.integer):
        raise ValueError("Expected a two-dimensional integer code matrix")
    rows, columns = codes.shape
    method = "awq" if input_scale is not None else "rtn"
    expected = matrix_payload(codes.shape, bits, method)
    if bits not in WIDTHS:
        raise ValueError("Packing applies only to integer weights")
    if np.any(codes < 0) or np.any(codes >= (1 << bits)):
        raise ValueError("Codes outside the specified integer grid")
    if (scales.shape != (rows, columns // GROUP_SIZE)
            or not np.all(np.isfinite(scales)) or np.any(scales <= 0)):
        raise ValueError("Expected a positive finite scale per row and input group")
    if input_scale is not None:
        input_scale = np.asarray(input_scale, dtype=np.float32)
        if (input_scale.shape != (columns,) or not np.all(np.isfinite(input_scale))
                or np.any(input_scale <= 0)):
            raise ValueError("Expected a positive finite scale per input channel")

    if bits == 8:
        packed = codes.astype(np.uint8)
    else:
        # Eight codes occupy exactly 'bits' bytes, including at 3 bits.
        shifts = np.arange(8, dtype=np.uint32) * bits
        chunks = codes.astype(np.uint32).reshape(rows, -1, 8)
        words = np.bitwise_or.reduce(chunks << shifts, axis=-1)
        offsets = np.arange(bits, dtype=np.uint32) * 8
        packed = (words[..., None] >> offsets).astype(np.uint8).reshape(rows, -1)
    result = PackedWeight(codes.shape, bits, packed, scales.copy(),
                          None if input_scale is None else input_scale.copy())
    if result.nbytes != expected:
        raise RuntimeError("Packed payload does not match the byte formula")
    return result


def round_to_nearest(weight, bits):
    """Quantize a linear weight using the study's symmetric grouped range rule."""
    if bits not in WIDTHS:
        raise ValueError("Invalid integer precision")
    weight = np.asarray(weight, dtype=np.float32)
    matrix_payload(weight.shape, bits, "rtn")
    if not np.isfinite(weight).all():
        raise ValueError("Nonfinite weights")
    groups = weight.reshape(weight.shape[0], -1, GROUP_SIZE)
    low = np.minimum(groups.min(axis=-1), 0.0)
    high = np.maximum(groups.max(axis=-1), -low)
    low = np.where(low < 0, -high, low)
    zero_group = (low == 0) & (high == 0)
    low[zero_group], high[zero_group] = -1.0, 1.0
    maximum_code = (1 << bits) - 1
    scales = (high - low) / np.float32(maximum_code)
    codes = np.rint(groups / scales[..., None]) + (1 << (bits - 1))
    codes = np.clip(codes, 0, maximum_code).astype(np.uint8).reshape(weight.shape)
    return pack_weight(codes, scales, bits)


def nv_candidate_ranks(shape):
    """Storage-saving ranks in the NV-Embed-v2 grid, aligned to 64 channels."""
    rows, columns = shape
    width = min(shape)
    grid = {max(GROUP_SIZE, GROUP_SIZE * round(width * f / GROUP_SIZE))
            for f in (1 / 16, 1 / 8, 1 / 4, 1 / 2)}
    return sorted(rank for rank in grid if rank * (rows + columns) < rows * columns)


def weighted_factors(weight, whitening, channel_weights, rank):
    """Return (input factor, output factor) using the original weighted SVD algebra.

    Supply the lower triangular input whitening factor from training activations
    and the positive, normalized weights for output channels.
    """
    weight = np.asarray(weight, dtype=np.float64)
    whitening = np.asarray(whitening, dtype=np.float64)
    channel_weights = np.asarray(channel_weights, dtype=np.float64)
    rows, columns = weight.shape
    if (whitening.shape != (columns, columns) or channel_weights.shape != (rows,)
            or not 0 < rank <= min(weight.shape)
            or not np.all(np.isfinite(channel_weights) & (channel_weights > 0))):
        raise ValueError("Incompatible weight, training statistics, or rank")
    root = np.sqrt(channel_weights)
    left, values, right = np.linalg.svd(root[:, None] * (weight @ whitening),
                                        full_matrices=False)
    split = np.sqrt(values[:rank])
    first = np.linalg.solve(whitening.T, (split[:, None] * right[:rank]).T).T
    second = left[:, :rank] * split[None, :] / root[:, None]
    return first.astype(np.float32), second.astype(np.float32)


def output_damage(reference, replacement, channel_weights, mask=None):
    """Channel-weighted squared output change, averaged over valid tokens.

    Pass linear outputs for floating-point scoring or block endpoints for
    quantized-block scoring. Only the candidate site should differ.
    """
    reference = np.asarray(reference, dtype=np.float32)
    replacement = np.asarray(replacement, dtype=np.float32)
    weights = np.asarray(channel_weights, dtype=np.float64)
    if (reference.shape != replacement.shape or reference.ndim < 2
            or weights.shape != (reference.shape[-1],)
            or not np.all(np.isfinite(weights) & (weights >= 0))):
        raise ValueError("Incompatible outputs or channel weights")
    delta = replacement - reference
    if mask is None:
        delta = delta.reshape(-1, reference.shape[-1])
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != reference.shape[:-1]:
            raise ValueError("Invalid token mask")
        delta = delta[mask]
    if not len(delta) or not np.isfinite(delta).all():
        raise ValueError("Output damage requires finite values for valid tokens")
    return float(np.sum(delta.astype(np.float64) ** 2 * weights) / len(delta))


def allocate_layout(profiles, budget_bytes, fixed_bytes, score):
    """Choose one option per matrix by minimum total damage within the byte cap.

    Equal damage favors lower payload and then the original option order.
    Each option contains 'rank', 'bytes', and the requested damage score.
    """
    if budget_bytes <= 0 or fixed_bytes < 0 or fixed_bytes > budget_bytes:
        raise ValueError("Invalid storage limits")
    if not profiles or any(not options for options in profiles.values()):
        raise ValueError("Every matrix needs a nonempty list of options")

    costs = []
    for name, options in profiles.items():
        ranks = set()
        for option in options:
            cost, value, rank = option["bytes"], option[score], option["rank"]
            if (not isinstance(cost, int) or cost <= 0 or not math.isfinite(value)
                    or value < 0 or not isinstance(rank, int) or rank < 0 or rank in ranks):
                raise ValueError(f"Invalid option for {name}")
            costs.append(cost)
            ranks.add(rank)
        if 0 not in ranks:
            raise ValueError(f"Missing intact-matrix option for {name}")

    unit = math.gcd(*costs)
    limit = (budget_bytes - fixed_bytes) // unit
    states = {0: (0.0, ())}
    for name, options in profiles.items():
        following = {}
        for used, (damage, choices) in states.items():
            for index, option in enumerate(options):
                total = used + option["bytes"] // unit
                if total > limit:
                    continue
                proposed = damage + option[score], choices + (index,)
                previous = following.get(total)
                if previous is None or proposed < previous:
                    following[total] = proposed
        if not following:
            raise ValueError(f"No layout fits the budget through matrix {name}")
        states = following

    loss, used, choices = min((damage, used, choices) for used, (damage, choices) in states.items())
    layout = {name: options[index]["rank"] for (name, options), index
              in zip(profiles.items(), choices, strict=True)}
    return {"ranks": layout, "payload_bytes": fixed_bytes + used * unit,
            "budget_bytes": budget_bytes, "damage": loss}


def choose_layout(floating_cross_entropy, block_cross_entropy, floating_margin_mse, block_margin_mse):
    """Apply the training holdout rule to the two separately allocated layouts."""
    values = (floating_cross_entropy, block_cross_entropy, floating_margin_mse, block_margin_mse)
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("Invalid held-out loss or margin error")
    return ("block" if block_cross_entropy < floating_cross_entropy
            and block_margin_mse <= floating_margin_mse else "floating")

