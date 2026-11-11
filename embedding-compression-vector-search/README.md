# Compressing Embedding Models for Vector Search

This repository accompanies *Compressing Embedding Models for Vector Search: Quantization and Low-Rank Factorization*. The study measures how post-training weight compression affects vector-search quality. It compares direct quantization with selective low-rank factorization followed by quantization, using the measured parameter size of each direct model as the storage limit for the corresponding factorized model.

The evaluation covers four quantizers and nine embedding models. The full 2-, 3-, 4-, and 8-bit sweep uses BGE-base and NV-Embed-v2. The experiments use 16 datasets from NanoBEIR and community question answering (CQA), with full-corpus evaluations on Natural Questions (NQ) and FiQA. The central budget comparisons use direct and factorized precisions of 2 and 4 bits, 4 and 8 bits, and 8 bits and FP16, respectively. Factorization improves quality at the first limit, while direct quantization leads at the two higher limits tested.

## Reproduce the results

Run these commands from the repository root with Python 3.10 or later. A GPU and model downloads are unnecessary for this part.

```sh
python -m pip install -r requirements.txt
python scripts/check_results.py --output results
python scripts/plot_direct_quantization.py
python scripts/plot_matched_budgets.py
python scripts/plot_dataset_deltas.py
```

The first script checks task coverage, the storage caps, the held-out layout-selection rule, and the quantizer choices used in the follow-up experiments. It writes the task-group averages and the full-corpus and ablation tables to `results/`. The plotting scripts write Figures 3–5 to `figures/` as PDF and PNG files.

| Result | Source tables |
| --- | --- |
| Direct quantization by bit width | `data/direct_per_task.csv` |
| Direct and factorized weights at the same storage limit | `data/matched_budget_per_task.csv`, `data/matched_budget_group.csv`, `data/infeasible_designs.csv` |
| Factor weighting, candidate scoring, and held-out layout selection | `data/factor_weighting_ablation.csv`, `data/scoring_comparison_*.csv`, `data/layout_selection.csv` |
| Additional embedding models and the Qwen3-8B versus Qwen3-4B storage comparison | `data/supplement_per_task.csv`, `data/supplement_reference_group.csv`, `data/method_selection.csv` |
| Full-corpus NQ and FiQA | `data/full_corpus.csv` |
| NV-Embed-v2 with a fixed factorized layout at two precisions | `data/fixed_layout.csv` |

Metrics are stored as fractions. The figures multiply nDCG@10 and its differences by 100. Each NanoBEIR or CQA mean weights its datasets equally. Empty cells indicate unavailable values. `both_quantized` uses the same evaluated model for queries and documents, including when the evaluated model is a floating-point reference. `original_docs_quantized_queries` reuses document vectors from the original model. `shared_original` identifies the common original-model reference. The supplement's case identifiers distinguish comparisons that can share a physical result.

## Run the compression and evaluation components

`vector_search_compression/compression.py` contains the grouped weight format, round-to-nearest quantization (RTN), weighted singular value decomposition (SVD), candidate damage, exact-byte layout allocation, and held-out layout selection. The functions take explicit weights, training statistics, or measured candidate outputs. `pack_weight` also accepts the fitted codes and scales from GPTQ, GPTAQ, and activation-aware weight quantization (AWQ) adapters. It preserves the permanent AWQ input scale in the byte count.

```sh
python scripts/smoke_test.py
```

To evaluate prepared embeddings, supply single-precision NumPy arrays and text files with one identifier per row, in embedding order. Supply a tab-separated relevance file with the header `query_id`, `document_id`, `relevance`. Enable self-match exclusion for datasets where a query can appear in the corpus.

```sh
python -m vector_search_compression.retrieval --queries queries.npy --documents documents.npy --query-ids query_ids.txt --document-ids document_ids.txt --qrels qrels.tsv --exclude-self --output results/retrieval.json
```

The command uses exhaustive inner-product search and reports normalized discounted cumulative gain (nDCG@10), recall (Recall@10), and mean reciprocal rank (MRR@10). Equal scores follow corpus row order. To use the full-corpus graphics processing unit (GPU) ranking kernel, install a CUDA-enabled build of [PyTorch](https://pytorch.org/get-started/locally/) and add `--device cuda`. The full-corpus kernel uses document chunks of 65,536 and query batches of 16. Use the released task scores to reproduce the paper's NanoBEIR figures, which include the original benchmark evaluator's handling of ties.

## Experiment inputs

Use the checkpoint revisions in [data/models.csv](data/models.csv) and the query and corpus sizes in [data/datasets.csv](data/datasets.csv). NQ and FiQA are full-corpus counterparts of NanoNQ and NanoFiQA. The encoding and compression controls are summarized in [PROTOCOL.md](PROTOCOL.md).

The result tables and plotting scripts are self-contained. Rerunning all models from raw text requires the model-specific encoders, the original calibration and held-out examples, and the method-specific quantization adapters. This repository provides the compression components and prepared-embedding evaluator. Checkpoints, benchmark text, calibration text, and cached embeddings are obtained separately under their respective terms.

