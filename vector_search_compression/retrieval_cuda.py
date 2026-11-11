"""GPU full-corpus top-k in single-precision arithmetic."""

import numpy as np
import torch


def _stable_select(scores, ids, k):
    """Order candidates by score descending, then original row ascending."""
    k = min(k, scores.shape[1])
    # Sorting the small merged candidate set is deterministic even at ties.
    by_id = torch.argsort(ids, dim=1, stable=True)
    ids = ids.gather(1, by_id)
    scores = scores.gather(1, by_id)
    by_score = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]
    return scores.gather(1, by_score), ids.gather(1, by_score)


@torch.inference_mode()
def exact_topk_cuda(queries, documents, self_indices, k=10, document_chunk=65536,
               query_batch=16, device='cuda'):
    """Return top-k scores and corpus offsets using the full-corpus ranking kernel."""
    if (queries.ndim != 2 or documents.ndim != 2 or queries.shape[1] != documents.shape[1]
            or not len(queries) or not len(documents) or len(self_indices) != len(queries)
            or k <= 0 or document_chunk <= 0 or query_batch <= 0):
        raise ValueError("Invalid embedding shapes or ranking parameters")
    if queries.dtype != np.float32 or documents.dtype != np.float32:
        raise ValueError("Embeddings must be stored in float32")
    if not np.isfinite(queries).all():
        raise ValueError("Nonfinite query embeddings")
    if any(index < -1 or index >= len(documents) for index in self_indices):
        raise ValueError("Expected a valid self-match offset or -1 for each query")
    k = min(k, len(documents))
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    all_scores = torch.full((len(queries), k), -torch.inf, dtype=torch.float32, device=device)
    all_ids = torch.full((len(queries), k), len(documents), dtype=torch.long, device=device)
    q_all = torch.from_numpy(np.ascontiguousarray(queries, dtype=np.float32)).to(device)
    for start in range(0, len(documents), document_chunk):
        end = min(start + document_chunk, len(documents))
        d = torch.from_numpy(np.array(documents[start:end], dtype=np.float32, copy=True)).to(device)
        for qb in range(0, len(queries), query_batch):
            qe = min(qb + query_batch, len(queries))
            q = q_all[qb:qe]
            scores = q @ d.T
            if not torch.isfinite(scores).all():
                raise ValueError("Nonfinite inner product")
            for row, index in enumerate(self_indices[qb:qe]):
                if start <= index < end:
                    scores[row, int(index)-start] = -torch.inf
            n = min(k, end-start)
            values, local = torch.topk(scores, min(n+1, end-start), dim=1, sorted=True)
            chosen = local[:, :n]
            if end-start > n:
                tied = values[:, n-1] == values[:, n]
                rows = torch.nonzero(tied, as_tuple=False).flatten()
                if len(rows):
                    chosen[rows] = torch.argsort(scores[rows], dim=1, descending=True, stable=True)[:, :n]
            ids = chosen + start
            values = scores.gather(1, chosen)
            best_scores, best_ids = _stable_select(torch.cat((all_scores[qb:qe], values), dim=1),
                                                   torch.cat((all_ids[qb:qe], ids), dim=1), k)
            all_scores[qb:qe] = best_scores
            all_ids[qb:qe] = best_ids
        del d, scores
    all_ids.masked_fill_(~torch.isfinite(all_scores), -1)
    return all_scores.cpu().numpy(), all_ids.cpu().numpy()
