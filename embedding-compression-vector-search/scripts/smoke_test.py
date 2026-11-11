"""Check compression and ranking on a small deterministic input."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vector_search_compression.compression import (
    WIDTHS, allocate_layout, choose_layout, matrix_payload, output_damage,
    pack_weight, round_to_nearest, weighted_factors,
)
from vector_search_compression.retrieval import exact_topk, metrics


def main():
    rng = np.random.default_rng(12)
    weight = rng.normal(size=(128, 128)).astype(np.float32)
    for bits in WIDTHS:
        maximum = 1 << bits
        indices = rng.integers(0, maximum, size=weight.shape, dtype=np.uint8)
        scales = np.ones((128, 2), dtype=np.float32)
        packed = pack_weight(indices, scales, bits)
        np.testing.assert_array_equal(packed.decode(), indices.astype(np.float32) - maximum // 2)
        assert packed.nbytes == matrix_payload(weight.shape, bits, "rtn")
        rtn = round_to_nearest(weight, bits)
        assert rtn.nbytes == packed.nbytes
        np.testing.assert_allclose(round_to_nearest(np.zeros_like(weight), bits).decode(), 0)

    awq = pack_weight(indices, scales, 8, np.full(128, 2, dtype=np.float32))
    np.testing.assert_array_equal(awq.decode(), (indices.astype(np.float32) - 128) / 2)
    assert awq.nbytes == matrix_payload(weight.shape, 8, "awq")

    first, second = weighted_factors(weight, np.eye(128), np.ones(128), rank=64)
    assert (first.shape, second.shape) == ((64, 128), (128, 64))
    damage = output_damage(weight, second @ first, np.ones(128))
    profiles = {
        "layer.0": [{"rank": 0, "bytes": 8, "damage": 0.0},
                    {"rank": 64, "bytes": 4, "damage": damage}],
        "layer.1": [{"rank": 0, "bytes": 8, "damage": 0.0},
                    {"rank": 64, "bytes": 4, "damage": damage}],
    }
    layout = allocate_layout(profiles, budget_bytes=16, fixed_bytes=4, score="damage")
    assert layout["ranks"] == {"layer.0": 0, "layer.1": 64}
    assert layout["payload_bytes"] == 16
    assert choose_layout(2, 1, 1, 1) == "block"
    assert choose_layout(2, 2, 1, 0.5) == "floating"
    assert choose_layout(2, 1, 1, 1.00001) == "floating"

    documents = np.array([[1, 0], [1, 0], [0, 1], [0.5, 0]], dtype=np.float32)
    queries = np.array([[1, 0], [0, 1]], dtype=np.float32)
    _, order = exact_topk(queries, documents, [-1, 2], document_chunk=1, query_batch=1)
    np.testing.assert_array_equal(order, [[0, 1, 3, 2], [0, 1, 3, -1]])
    summary, rows = metrics(order, ["q0", "q1"], ["d0", "d1", "q1", "d3"],
                            {"q0": {"d1": 1}, "q1": {"d0": 1, "d3": 1}})
    assert len(rows) == 2 and summary["recall_at_10"] == 1 and summary["mrr_at_10"] == 0.75
    assert abs(rows[0]["ndcg_at_10"] - 1 / np.log2(3)) < 1e-12
    print("Compression, byte allocation, layout selection, and retrieval checks passed.")


if __name__ == "__main__":
    main()

