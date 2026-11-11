# Experimental protocol

## Weights and storage

- Compare Round-to-Nearest (RTN), [GPTQ](https://arxiv.org/abs/2210.17323), [GPTAQ](https://proceedings.mlr.press/v267/li25dn.html), and [Activation-aware Weight Quantization (AWQ)](https://arxiv.org/abs/2306.00978) at 2, 3, 4, and 8 bits. Original single-precision and full half-precision (FP16) models serve as references. The method-specific routines retain their calibration, range fitting, scaling, clipping, and compensation as applicable. The study uses the upstream GPTQ implementation at revision 2d65066eeb06a5c9ff5184d8cebdf33662c67faf, GPTAQ at d6bd2e41e9abc8f22a76d6ea789c450b6c143ced, and AWQ at d6e797a42b9ef7778de8ee2352116e0f48a78d61.
- Group 64 consecutive input channels within each row. Store the packed codes and a single-precision scale for each row group. The zero point is the implicit midpoint of the unsigned code grid. Store permanent AWQ input scales when used. Count the fixed-precision remainder once, including the vocabulary weights. Report parameter payload in bytes separately from checkpoint container overhead.
- A direct model sets the byte limit for the corresponding factorized model. Its factorized counterpart uses the specified precision for both factors and for retained complete matrices. Full FP16 model references and selective FP16 factorization are distinct representations. Retrieval executes with decoded weights and single-precision arithmetic. TensorFloat-32 (TF32) is disabled. These experiments measure quality and parameter storage.

## Factorization and model selection

1. Construct a basis for each eligible matrix from original training-side statistics. The NV-Embed-v2 implementation uses a full input Gram matrix with a diagonal ridge and smoothed output-channel weights. Compute the weighted singular value decomposition and split singular values symmetrically between the two unrotated factors. Keep the original matrix as one of the options. The rank grid for NV-Embed-v2 is supplied in the code. Other models retain their own candidate sets.
2. Profile each candidate on original calibration activations. Floating-point scoring measures the output change of the linear operator. Quantized-block scoring substitutes only the candidate and measures the change at the containing block's output. Both use channel-weighted squared error over valid tokens. For methods that calibrate activations, the second factor receives the output of the actual quantized first factor. GPTAQ retains the floating-point input as its teacher.
3. Allocate one option per matrix by dynamic programming within the measured byte limit. Apply each scoring rule separately. The held-out rule selects the quantized-block layout when it strictly reduces ranking cross-entropy and does not increase the mean squared error of score margins relative to the original model. Otherwise it selects the floating-point layout.
4. The main calibration set has 512 training-side examples. The layout choice uses 128 separate training-side examples. Evaluation datasets supply no calibration, candidate scoring, or layout selection examples. Follow-up quantizers are chosen by summing unrounded nDCG@10 across complete comparable model–dataset results for the relevant configuration. The follow-up results do not change these choices.

## Retrieval

Preserve each embedding model's tokenizer, truncation, prompts, attention, pooling, and vector normalization. Keep dataset order and relevance judgments fixed. Evaluate with the same model for queries and documents or with original-model document embeddings, as recorded in each table. Rank by exhaustive similarity, exclude self matches where applicable, and average over eligible queries. In the full-corpus experiments, rank equal scores by their corpus row. The paper's NanoBEIR results use the original benchmark implementation for its exact and near-tie cases.

- [BEIR](https://arxiv.org/abs/2104.08663) and [Massive Text Embedding Benchmark (MTEB)](https://arxiv.org/abs/2210.07316) describe the underlying retrieval benchmarks. The NanoBEIR dataset revision is beb106fbcfaa599c508c667041bf8c85fd78736b.
- Full-corpus NQ uses the BEIR NQ test split, with 3,452 queries and 2,681,468 documents. Full-corpus FiQA uses the MTEB FiQA test split, with 648 queries and 57,638 documents. The CQA datasets use the query subsets and full corpora listed in the dataset table.

