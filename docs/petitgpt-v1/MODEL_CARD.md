# petitgpt

Author: Yang Qi. Selected checkpoint: alpha075.

## Identity

| Field | Value |
|---|---|
| Status | accepted native research inference artifact |
| Unique parameters | 124,635,456 (124.6M) |
| Layers / width / FFN | 30 / 576 / 1536 |
| Attention | 9 query heads, 3 key/value heads (GQA), head dim 64 |
| Vocabulary / context | 32,000 / 2,048 |
| Embeddings | tied input/output |
| Normalization | RMSNorm, epsilon 1e-6 |
| Positions | RoPE, theta 10000, full head rotation |
| Dropout | 0.0 |
| Stored weights | FP32 |

Complete checkpoint-derived settings ship in the bundle's `config.json`. The three core modules (model, chat template, token contract) are byte-identical to the project originals. Checkpoint and archive hashes are in MODEL_PROVENANCE.json.

## Provenance

The selected weights are a **parameter interpolation**, not a training step:

> `theta = theta_P2_step750 + 0.75 · (theta_P3_step320 − theta_P2_step750)`

Ancestry: tokenizer release → Stage A pretraining (steps 0–38,146) → Stage B continued pretraining (steps 38,146–49,590, exact full-state resume, accepted as Base) → P2 concise-instruction SFT (750 updates, weights-only initialization from Base) → P3 basic-instruction adaptation (parent B is **step 320**, not the step-640 endpoint) → this interpolation, executed with **zero optimizer updates and zero backward passes** → a numerically unchanged FP32 export.

This checkpoint **does not contain** later DeepSeek-response-KD, unified Base-SFT, DPO, soft-KD or LoRA branch updates. Several later branches were initialized *from* it (a one-pass behaviour mix, a DPO pilot, a chosen-answer CE control, and two response-distillation runs); others were not — a unified SFT curve started from the accepted pretrained Base, a loss-allocation A/B split from that curve's step 403, and a shared-tokenizer soft-KD lab ran entirely on external models. A preference-data build used this model's generations but produced no checkpoint. Branch exposures must not be summed into this model's training history.

## Training data

Pretraining consumed 13,000,005,634 retained packed tokens over 13,755,731 documents, of which the optimizer stepped over 12,999,720,960 model-input positions, one exposure per block, with no replay of the executed token-position traversal.

**Stage A (10,000,003,234 selected serialized tokens, 4 sources):** FineWeb-Edu dedup 71.11%, DCLM-Edu 20.32%, Wikipedia (FineWiki EN) 5.08%, Python-Edu 3.50%.

**Stage B (3,000,004,240 selected serialized tokens, 7 sources):** FineWeb-Edu dedup 40.10%, DCLM-Edu 22.92%, structured tutorial content 11.46%, Python-Edu 8.33%, Wikipedia 5.73%, PES2O 5.73%, StackExchange 5.73%.

Upstream datasets, pinned revisions and the licence string recorded at each pinned revision are in `tables/PRETRAIN_SOURCE_MIXTURE.csv`. Those recorded strings are evidence of what the builder captured at that revision; they are not a legal determination, are not asserted to be today's terms, and do not by themselves determine the licence of trained weights.

Post-training used seven instruction subsets. These are values of a row-level `source` column inside one pinned collection, `HuggingFaceTB/smol-smoltalk` at revision `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`, config `default`, train split, established by digest joins through the project's own census and cleanup records. The publisher's card at that pinned revision carries a flat Apache-2.0 badge; its parent collection limits that grant to four newly generated subsets and refers readers to the original dataset for each incorporated public dataset. Four of the seven labels correspond to the newly generated subsets. Of the three incorporated components, one declares `apache-2.0`, one declares `odc-by`, and one declares no licence in its card metadata. Component notices were read from current publisher pages, not from revisions contemporaneous with this training run, and **no component revision is established**. No licence determination is made here.

## Intended use, and use it is not intended for

**Intended:** research and engineering study of a small from-scratch language model — reproducing the recorded measurements, inspecting the pipeline, and analysing failure modes.

**Not intended:** a general assistant, a production system, anything safety- or correctness-certified, or a source of factual answers. Do not execute code it generates without independent review.

**Not evaluated at all:** long-context work, multilingual behaviour, tool use, extended multi-turn dialogue, safety and refusal behaviour, factual currency, retrieval, and any public generative benchmark.

## Evaluation — public multiple-choice likelihood

Frozen FP32 results, copied byte-identically from the accepted measurement and **not recomputed**:

| Dataset | Split / documents | acc | acc_norm |
|---|---|---|---|
| ARC-Easy | test / 2,376 | 1372/2376 = 0.5774410774410774 | 1244/2376 = 0.5235690235690236 |
| PIQA | validation / 1,838 | 1167/1838 = 0.6349292709466812 | 1145/1838 = 0.6229597388465724 |

Protocol: zero-shot raw `Question: …\nAnswer:` completion scored by candidate-answer likelihood. No chat template, no role tokens, no few-shot examples, no BOS insertion, no scored EOS, no generation, no cleanup. FP32 parameters and forward with autocast disabled, TF32 off for matmul and cuDNN, MATH SDPA, batch size 1 unpadded, no KV cache, no compile. `acc` is the first argmax of summed continuation log-likelihood; `acc_norm` divides by `len()` of the **original** answer text in Unicode characters, not tokenizer length; ties take the first index.

Comparators measured under the identical protocol on the same rows: SmolLM-135M-Instruct 0.4924 / 0.6708 and SmolLM2-135M-Instruct 0.5400 / 0.6670 (acc, ARC-Easy / PIQA).

**Qualifications.** The evaluator is a native protocol-compatible implementation pinned to a specific lm-evaluation-harness commit; the harness package was not installed and a full installed-harness run is not claimed. Both datasets are prior project diagnostics with no contamination audit — they are **not untouched final tests**. Training data, compute, tokenizers and architectures are unmatched across the three models. This is a protocol-bounded descriptive comparison; no significance test was run. Multiple-choice accuracy does not establish free-generation reliability.

## Evaluation — historical full-answer assistant review

A separate evaluation family scored generated text across 189 prompts per model (567 answers, 565 distinct prompt/output/contract units) over three models. **Two named versions exist and must not be combined in one table:** `assistant_review_fable_v1` (the original 567 final records) and `assistant_owner_clarification_4_v1` (four explicit final-content decisions, every other axis preserved). The version shown below is `assistant_owner_clarification_4_v1`.

| Slice | Content / joint (true / false / unknown) | Other axes |
|---|---|---|
| old_qa41 | 6 / 33 / 2 | format 0/1 |
| old_practical38 | 10 / 26 / 2 | explicit format 20/21 |
| old_python14 | 0 / 14 / 0 | interface 11/14; finite execution `not_recorded` |
| new_natural64 | 3 / 61 / 0 | format 7/18 |
| new_python32 | 0 / 32 / 0 | interface 31/32 |

Practical joint bounds: **26/81 .. 10/27** (= 52/162 .. 60/162). This denominator is **27 equally weighted dialogue groups over 38 scored turns** — rows are averaged inside a group first — not an ordinary row average. These are exact unknown-retention bounds, **not confidence intervals**.

**Qualifications.** Judgments are model-assisted, not human adjudication. The chronology was: an initial pass over the 565 units with model metadata masked and its own recorded limitations; then a pass with the mapping visible that produced 17 consistency edits; then four owner clarifications forming a separate version. It was therefore neither strictly blinded throughout nor fully label-visible throughout. Development sets were reused across many runs. `not_recorded` means the field is unavailable in this imported view — it does not establish that no historical function test was ever run. Unsupported-builtin results stay unknown and are never converted into demonstrated failures.

In these specific Python diagnostics the model produced a correct function **interface** in 42 of 46 prompts and a correct **whole answer** in 0 of 46. This does not establish that it can never write correct code.

Ordinary QA and complete natural-task generation remained limited across the evaluated configurations. Individual results differ by suite and label version and are reported with their source rather than reduced to a cross-version maximum; see the technical report §10.1.

## Export parity

Eight frozen fixture pairs (four `bf16_native`, four `fp32_math`) matched exactly on prompt IDs, boundaries, full-shape logits, greedy output IDs and stop reason, with a maximum absolute logit difference of **0 within each profile**. All 213 named state entries and 60 non-persistent rotary buffers matched after strict load and safetensors reload.

This is a **numerical parity check between source and export under the same profile**. It is not a quality test, not a semantic evaluation, and it does not assert that the two profiles agree with each other. Local import closure was demonstrated once in a fresh isolated process on the measured environment; that is **not** a clean-machine, CPU, cross-hardware or fresh-installation test.

## Format support

Native PyTorch CUDA inference only. **No** Transformers `AutoModel`, GGUF, ONNX, vLLM or llama.cpp compatibility is implemented or tested. `special_tokens_map.json` is descriptive native metadata. A CUDA GPU is required.

## Licence and distribution

Copyright 2026 Yang Qi. Owner-controlled code and the selected model/tokenizer are licensed under the standard Apache License 2.0 in LICENSE, only for rights Yang Qi is entitled to grant.

As an explicit exception to the root code licence, author-written reports and documentation (including README, native run guide and versioned report) are licensed under Creative Commons Attribution 4.0 International (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/ and https://creativecommons.org/licenses/by/4.0/legalcode.en . Attribute Yang Qi and petitgpt, link the licence, and indicate changes. Existing third-party content and notices retain their applicable terms; they are not relicensed.

Research describes intended use; it adds no noncommercial or research-only restriction to Apache-licensed artifacts. This grant was approved by the owner through the explicit research-release execution instruction. Earlier PENDING_OWNER_DECISION records remain historical evidence; this does not claim an earlier licence choice.

Source metadata is not rights clearance. Weights are not the raw corpus; neither automatic inheritance nor automatic non-application of all dataset terms is asserted. No infringement guarantee or legal certification is given. A disclaimer does not replace applicable permission.

## Recorded runtime

Python 3.10.12, torch 2.11.0+cu126, numpy 2.2.6, tokenizers 0.22.2, safetensors 0.8.0, NVIDIA GeForce RTX 4090. Matching versions do not guarantee bit-identical results on other hardware or untested software; recorded driver versions differ across project phases, which is a recorded difference rather than a resolved equivalence.

## Files and report

See [native run guide](RUN_GUIDE.md), [source notice](SOURCE_NOTICE.md), [third-party notices](THIRD_PARTY_NOTICES.md) and [file manifest](SHA256SUMS).

[Complete report](TECHNICAL_REPORT.md). Model destination: https://huggingface.co/yqi0/petitgpt .
