# PetitGPT model card

**Author:** Yang Qi · **Checkpoint:** alpha075 · **Release:** research-v1

PetitGPT is a 124.6M-parameter language model pretrained from scratch on one RTX 4090, then adapted for instruction following. The released weights interpolate two post-training checkpoints. Multiple-choice results are competitive on some tasks, while reliable factual answers, rewriting, and complete code generation remain limited.

[Download the model](https://huggingface.co/yqi0/petitgpt) · [Run guide](RUN_GUIDE.md) · [Technical report](TECHNICAL_REPORT.md) · [Reproducibility](../../TRAINING_AND_REPRODUCIBILITY.md)

## Identity

| Field | Value |
|---|---|
| Status | released research-v1 checkpoint; native PyTorch CUDA inference |
| Unique parameters | 124,635,456 (124.6M) |
| Layers / width / FFN | 30 / 576 / 1,536 |
| Attention | 9 query heads, 3 key/value heads (GQA), head dim 64 |
| Vocabulary / context | 32,000 / 2,048 |
| Embeddings | tied input/output |
| Normalization | RMSNorm, epsilon 1e-6 |
| Positions | RoPE, theta 10,000, full head rotation |
| Dropout | 0.0 |
| Stored weights | FP32 |

Complete checkpoint-derived settings ship in the bundle's `config.json` ([source copy](../../inference_native/config.json)). Checkpoint, tokenizer, and archive hashes are in [MODEL_PROVENANCE.json](MODEL_PROVENANCE.json).

## Provenance

The selected weights are a **parameter interpolation**, not a training step:

> `theta = theta_P2_step750 + 0.75 · (theta_P3_step320 − theta_P2_step750)`

The training path is Stage A pretraining (updates 1–38,146), Stage B continuation (38,147–49,590, with full optimizer state resumed), P2 concise-instruction SFT (750 updates, initialized from Base weights), then P3 basic-instruction adaptation with replay. The blend uses **P3 step 320**; step 640 is a later comparison point. Interpolation and export add no optimizer updates.

Later DPO, response-distillation, LoRA, and unified Base-SFT experiments are **not ancestors of these weights**. Some brought local gains alongside losses in other capabilities; none was selected to replace alpha075. The separate soft-logit distillation lab used external models. See the [branch map](TECHNICAL_REPORT.md#6-selected-ancestry-versus-research-branches) and [experiment ledger](tables/RESEARCH_EXPERIMENT_LEDGER.csv) for initialization and outcomes.

## Training data

The pretraining selection contained 13,755,731 documents. Packing retained 13,000,005,634 tokens; optimizer updates actually traversed 12,999,720,960 model-input positions, with one exposure per executed block. The [token accounting](TECHNICAL_REPORT.md#42-four-token-quantities-reconciled) explains the differences between selected, packed, and processed quantities.

**Stage A (10,000,003,234 selected serialized tokens, 4 sources):** FineWeb-Edu dedup 71.11%, DCLM-Edu 20.32%, Wikipedia (FineWiki EN) 5.08%, Python-Edu 3.50%.

**Stage B (3,000,004,240 selected serialized tokens, 7 sources):** FineWeb-Edu dedup 40.10%, DCLM-Edu 22.92%, structured tutorial content 11.46%, Python-Edu 8.33%, Wikipedia 5.73%, PES2O 5.73%, StackExchange 5.73%.

Dataset revisions and recorded source notices are in the [mixture table](tables/PRETRAIN_SOURCE_MIXTURE.csv) and [source notice](SOURCE_NOTICE.md). Source shares describe selected serialized tokens, not separately measured per-source optimizer consumption.

P2 used 12,000 selected conversations from seven row-level `source` subsets of `HuggingFaceTB/smol-smoltalk` at revision `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`, config `default`, train split. Loss supervised every assistant turn's content and trailing EOS. P3 combined procedural instruction tasks with replay examples from the earlier instruction data. See [post-training](TECHNICAL_REPORT.md#5-post-training-and-the-selected-weights) for the training settings and selection trade-off. Component revisions and licence questions that remain unresolved are documented in the source notice.

## Intended use, and use it is not intended for

**Intended:** research and experimentation with small language models, including reproducing measurements, inspecting the pipeline, and analysing failure modes.

Treat generated factual claims and code as unverified. The model has not been validated for production or high-stakes use, tool use, multilingual capability, long-context work, extended dialogue, or safety and refusal behavior. IFEval and the historical assistant review measure specific generation tasks; they do not establish broad assistant reliability.

## Evaluation — public multiple-choice likelihood

The project measured all three models under the same task protocols. Alpha075 leads SmolLM-135M-Instruct and SmolLM2-135M-Instruct on ARC-Easy and ARC-Challenge, and trails both on PIQA and HellaSwag. Its results are:

| Dataset | Split / documents | acc | acc_norm |
|---|---|---:|---:|
| ARC-Easy | test / 2,376 | 57.74% | 52.36% |
| ARC-Challenge | test / 1,172 | 28.16% | 32.68% |
| PIQA | validation / 1,838 | 63.49% | 62.30% |
| HellaSwag | validation / 10,042 | 31.28% | 35.60% |

ARC-Easy and PIQA come from **FP32 V2**; ARC-Challenge and HellaSwag come from **benchmark extension V1**. Full-precision values, correct counts, revisions, and comparator results are in [PUBLIC_BENCHMARK_RESULTS.csv](tables/PUBLIC_BENCHMARK_RESULTS.csv) and the [report](TECHNICAL_REPORT.md#81-public-multiple-choice-likelihood-frozen-fp32).

Each model uses its own tokenizer with no chat template, inserted BOS/EOS, or text generation. `acc` ranks candidates by summed continuation log-likelihood; `acc_norm` divides that sum by the candidate's Unicode character count, excluding the leading delimiter. This is the original answer text for ARC/PIQA and the pinned task-preprocessed ending for HellaSwag. It is character normalization, not token normalization, and may differ from externally reported `acc_norm` scores.

<details>
<summary>Likelihood protocol and interpretation</summary>

ARC and PIQA use raw `Question: …\nAnswer:` prompts; HellaSwag uses the pinned task's preprocessed context. Ties take the first candidate. Both campaigns use FP32 parameters, forward computation, log-softmax and sums, with autocast and TF32 disabled, MATH scaled dot-product attention, batch size 1, no padding, no KV cache, and no compilation. The evaluator follows pinned lm-evaluation-harness conventions; it is not a full installed-harness run. See [EVALUATION_PROTOCOLS.json](provenance/EVALUATION_PROTOCOLS.json).

ARC-Easy and PIQA were reused during development. No contamination audit was performed for these benchmarks. Training data, compute, tokenizers, and architectures differ across the models, and no significance test was run. The scores describe these measurements and do not establish reliable free-form generation.

</details>

## Evaluation — IFEval instruction following

**IFEval completion V2** evaluated 541 prompts containing 834 instructions across 25 instruction types. The dataset has a single split named `train`; that name does not mean the prompts were used to train PetitGPT.

| Model | Prompt strict | Instruction strict | Prompt loose | Instruction loose | 1,280-token cap hits |
|---|---:|---:|---:|---:|---:|
| PetitGPT-alpha075 | 17.19% | 28.54% | 17.74% | 29.98% | 40/541 |
| SmolLM-135M-Instruct | 10.35% | 21.82% | 12.01% | 24.10% | 290/541 |
| SmolLM2-135M-Instruct | 21.63% | 35.85% | 22.55% | 37.29% | 233/541 |

Each model received the same user prompts through its native chat formatter. SmolLM2's template inserts default system text; PetitGPT's does not. Generation was zero-shot and greedy with a 1,280-new-token cap, using a separate evaluator rather than the released CLI (whose cap is 384). Strict/loose scores use programmatic verifiers with no LLM judge. No input prompts were dropped or truncated; cap-hit responses were scored as generated.

Prompt accuracy requires all instructions in a prompt to pass; instruction accuracy counts each instruction separately. Full-precision results and integer counts are in [IFEVAL_RESULTS.csv](tables/IFEVAL_RESULTS.csv); see the [report](TECHNICAL_REPORT.md#84-ifeval-instruction-following) for numerical settings and interpretation.

## Evaluation — historical full-answer assistant review

A separate evaluation family reviewed generated text across 189 prompts per model over three models. **Two named versions exist and must not be combined in one table:** `assistant_review_fable_v1` (the original 567 final records) and `assistant_owner_clarification_4_v1` (four explicit final-content decisions, every other axis preserved). The version shown below is `assistant_owner_clarification_4_v1`.

| Slice | Content / joint (true / false / unknown) | Other axes |
|---|---|---|
| old_qa41 | 6 / 33 / 2 | format 0/1 |
| old_practical38 | 10 / 26 / 2 | explicit format 20/21 |
| old_python14 | 0 / 14 / 0 | interface 11/14; finite execution `not_recorded` |
| new_natural64 | 3 / 61 / 0 | format 7/18 |
| new_python32 | 0 / 32 / 0 | interface 31/32 |

Practical joint bounds: **26/81 .. 10/27** (= 52/162 .. 60/162). This denominator is **27 equally weighted dialogue groups over 38 scored turns** — rows are averaged inside a group first — not an ordinary row average. These are exact unknown-retention bounds, **not confidence intervals**.

**Qualifications.** The recorded reviewing model was Claude Fable 5.1 through Claude Code at high effort; these are model-assisted judgments, not independent human gold labels. An initial pass masked model metadata; a later pass with the mapping visible produced 17 consistency edits, followed by four owner clarifications. Review was therefore only partly blinded, and the development sets were reused across many runs.

`not_recorded` means the field is unavailable in this imported view, not proof that a test never ran. Results involving builtins unavailable in the restricted execution environment remain unknown rather than being counted as model failures.

In these specific Python diagnostics the model produced a correct function **interface** in 42 of 46 prompts and a correct **whole answer** in 0 of 46. The 0/46 figure is a full-answer review result, not training-set accuracy or a uniform unit-test pass rate. [Success and failure cases](TECHNICAL_REPORT.md#85-qualitative-cases-successes-and-failures) show Python outputs, summaries, rewrites, and dialogue responses with complete stored answers and score-version notes.

Ordinary QA and complete natural-task generation remained limited across the evaluated configurations. Individual results differ by suite and label version and are reported with their source rather than reduced to a cross-version maximum; see the [versioned results](tables/ASSISTANT_RESULTS_VERSIONED.csv) and [review method](TECHNICAL_REPORT.md#83-review-method-and-development-set-reuse).

## Export parity

Eight frozen fixture pairs (four `bf16_native`, four `fp32_math`) matched exactly on prompt IDs, boundaries, full-shape logits, greedy output IDs and stop reason, with a maximum absolute logit difference of **0 within each profile**. All 213 named state entries and 60 non-persistent rotary buffers matched after strict load and safetensors reload.

This is a **numerical parity check between source and export under the same profile**. It is not a quality test, not a semantic evaluation, and it does not assert that the two profiles agree with each other. Local import closure was demonstrated once in a fresh isolated process on the measured environment; that is **not** a clean-machine, CPU, cross-hardware or fresh-installation test.

## Format support

Native PyTorch CUDA inference only. **No** Transformers `AutoModel`, GGUF, ONNX, vLLM or llama.cpp compatibility is implemented or tested. `special_tokens_map.json` is descriptive native metadata. A CUDA GPU is required.

## Licence and distribution

Author-controlled code, the model weights, and the tokenizer are released under [Apache-2.0](LICENSE). Author-written documentation is licensed separately under [CC BY 4.0](DOCUMENTATION_LICENSE.md). Third-party content retains its own terms and notices; see the [source notice](SOURCE_NOTICE.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

The research-use description adds no noncommercial or research-only restriction to the Apache-licensed artifacts. Cite the project using [CITATION.cff](../../CITATION.cff).

## Recorded runtime

Python 3.10.12, torch 2.11.0+cu126, numpy 2.2.6, tokenizers 0.22.2, safetensors 0.8.0, NVIDIA GeForce RTX 4090. Matching versions do not guarantee bit-identical results on other hardware or untested software; recorded driver versions differ across project phases, which is a recorded difference rather than a resolved equivalence.

## Files and report

See [native run guide](RUN_GUIDE.md), [source notice](SOURCE_NOTICE.md), [third-party notices](THIRD_PARTY_NOTICES.md) and [documentation manifest](SHA256SUMS). The downloaded HF bundle has its own `SHA256SUMS` covering the model and runtime files.

[Complete report](TECHNICAL_REPORT.md). Model destination: https://huggingface.co/yqi0/petitgpt .
