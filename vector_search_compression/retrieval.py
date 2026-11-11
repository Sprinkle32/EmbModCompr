"""Exhaustive inner-product search over prepared embeddings."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def exact_topk(queries, documents, self_indices, k=10, document_chunk=65536, query_batch=16):
    """Return corpus offsets, breaking equal scores by corpus row order."""
    if (queries.ndim != 2 or documents.ndim != 2 or queries.shape[1] != documents.shape[1]
            or not len(queries) or not len(documents) or not 0 < k
            or document_chunk <= 0 or query_batch <= 0):
        raise ValueError("Invalid embedding shapes or ranking parameters")
    if queries.dtype != np.float32 or documents.dtype != np.float32:
        raise ValueError("Embeddings must be stored in float32")
    exclude = np.asarray(self_indices)
    if (exclude.shape != (len(queries),) or not np.issubdtype(exclude.dtype, np.integer)
            or np.any(exclude < -1) or np.any(exclude >= len(documents))):
        raise ValueError("Expected a valid self-match offset or -1 for each query")
    if not np.isfinite(queries).all():
        raise ValueError("Nonfinite query embeddings")

    k = min(k, len(documents))
    best_scores = np.full((len(queries), k), -np.inf, dtype=np.float32)
    best_ids = np.full((len(queries), k), len(documents), dtype=np.int64)
    for start in range(0, len(documents), document_chunk):
        end = min(start + document_chunk, len(documents))
        block = documents[start:end]
        if not np.isfinite(block).all():
            raise ValueError("Nonfinite document embeddings")
        ids = np.arange(start, end)
        for qb in range(0, len(queries), query_batch):
            qe = min(qb + query_batch, len(queries))
            scores = queries[qb:qe] @ block.T
            if not np.isfinite(scores).all():
                raise ValueError("Nonfinite inner product")
            local = exclude[qb:qe] - start
            active = (local >= 0) & (local < len(block))
            scores[np.flatnonzero(active), local[active]] = -np.inf
            # The block is already in corpus order, so stable ties stay in order.
            chosen = np.argsort(-scores, axis=1, kind="stable")[:, :k]
            candidates = np.concatenate((best_ids[qb:qe], ids[chosen]), axis=1)
            scores = np.concatenate((best_scores[qb:qe],
                                     np.take_along_axis(scores, chosen, axis=1)), axis=1)
            keep = np.lexsort((candidates, -scores), axis=1)[:, :k]
            best_ids[qb:qe] = np.take_along_axis(candidates, keep, axis=1)
            best_scores[qb:qe] = np.take_along_axis(scores, keep, axis=1)
    best_ids[~np.isfinite(best_scores)] = -1
    return best_scores, best_ids


def metrics(rankings, query_ids, document_ids, qrels, k=10):
    """Calculate linearly graded nDCG, recall, and reciprocal rank."""
    if len(rankings) != len(query_ids) or k <= 0:
        raise ValueError("Ranking/query mismatch or invalid cutoff")
    discount = 1 / np.log2(np.arange(2, k + 2))
    rows = []
    for query_id, ranking in zip(query_ids, rankings, strict=True):
        relevant = qrels.get(query_id)
        if not relevant:
            continue
        if any(not np.isfinite(gain) or gain <= 0 for gain in relevant.values()):
            raise ValueError("Relevance judgments must have positive finite grades")
        gains = np.array([relevant.get(document_ids[i], 0.0) if i >= 0 else 0.0
                          for i in ranking[:k]], dtype=np.float64)
        ideal = np.array(sorted(relevant.values(), reverse=True)[:k], dtype=np.float64)
        hits = np.flatnonzero(gains > 0)
        rows.append({
            "query_id": query_id,
            "ndcg_at_10": float((gains * discount[:len(gains)]).sum()
                                / (ideal * discount[:len(ideal)]).sum()),
            "recall_at_10": len(hits) / len(relevant),
            "mrr_at_10": 1 / (1 + int(hits[0])) if len(hits) else 0.0,
        })
    if not rows:
        raise ValueError("No queries have positive relevance judgments")
    if k != 10:
        rows = [{key.replace("_at_10", f"_at_{k}"): value for key, value in row.items()}
                for row in rows]
    return {key: float(np.mean([row[key] for row in rows]))
            for key in rows[0] if key != "query_id"}, rows


def read_ids(path):
    values = path.read_text(encoding="utf-8").splitlines()
    if any(not value for value in values) or len(set(values)) != len(values):
        raise ValueError(f"Expected one unique, nonempty identifier per line in {path}")
    return values


def read_qrels(path):
    qrels = {}
    with path.open(encoding="utf-8", newline="") as file:
        rows = csv.DictReader(file, delimiter="\t")
        required = {"query_id", "document_id", "relevance"}
        if not required.issubset(rows.fieldnames or ()):
            raise ValueError(f"Missing qrels columns: {required}")
        for row in rows:
            gain = float(row["relevance"])
            if not np.isfinite(gain):
                raise ValueError("Nonfinite relevance grade")
            if gain <= 0:
                continue
            relevant = qrels.setdefault(row["query_id"], {})
            document_id = row["document_id"]
            if document_id in relevant:
                raise ValueError("Duplicate positive relevance judgment")
            relevant[document_id] = gain
    return qrels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--query-ids", type=Path, required=True)
    parser.add_argument("--document-ids", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--exclude-self", action="store_true")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:<index>")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    queries = np.load(args.queries, mmap_mode="r", allow_pickle=False)
    documents = np.load(args.documents, mmap_mode="r", allow_pickle=False)
    query_ids, document_ids = read_ids(args.query_ids), read_ids(args.document_ids)
    if len(query_ids) != len(queries) or len(document_ids) != len(documents):
        raise ValueError("Identifiers must align one-to-one with the embedding rows")
    qrels = read_qrels(args.qrels)
    document_index = {name: i for i, name in enumerate(document_ids)}
    missing = next((name for query in query_ids for name in qrels.get(query, ())
                    if name not in document_index), None)
    if missing is not None:
        raise ValueError(f"Judged relevant document is outside the corpus: {missing}")
    offsets = [document_index.get(query, -1) if args.exclude_self else -1 for query in query_ids]
    if args.device == "cpu":
        _, order = exact_topk(queries, documents, offsets)
    elif args.device == "cuda" or args.device.startswith("cuda:"):
        from .retrieval_cuda import exact_topk_cuda

        _, order = exact_topk_cuda(queries, documents, offsets, device=args.device)
    else:
        raise ValueError("Expected cpu, cuda, or cuda:<index>")
    mean, rows = metrics(order, query_ids, document_ids, qrels)
    output = {"queries": len(rows), "documents": len(documents), **mean, "per_query": rows}
    result = json.dumps(output, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()

