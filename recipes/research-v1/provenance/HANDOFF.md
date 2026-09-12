# research-v1 source handoff

This is an implemented, publication-candidate source/config handoff for local integration. No model, dataset or optimizer payload is included. It does not execute training. Each stage below identifies its actual implementation, recorded command or explicit command gap, effective semantics, input/output identities and environment. The original path/hash map is private; SOURCE_MAP records original-content and candidate-content hashes separately.

## Release architecture and lineage

124,635,456 unique parameters;30 layers,width576,FFN1536,9Q/3KV heads,vocabulary32000,context2048,tied embeddings,RoPE/RMSNorm/SwiGLU/GQA. Do not substitute the older 16-layer/137M-era trainer specification.

Tokenizer F/G → selection I → packing M → pretrain A → full-state transition/bridge → pretrain B Base49590 → P2step750 → P3step320 → alpha075 = P2 +0.75*(P3step320-P2) → native FP32 export. Later DPO/responseKD/LoRA/unifiedSFT are separate research branches; the soft-logit lab uses external models.

## Source layout and integration contract

Each `sources/<stage>/` is an independent source root preserving its original package layout. Run modules with that root on PYTHONPATH; do not combine Stage A/B/posttraining `src` modules simply because names match. Input locations in configs are logical relative paths. A laptop integration adapter must bind them to acquired/restored artifacts. Sources preserve original fail-closed identity/scope checks; path sanitization changes source/contract bytes, so projected configs are evidence, not drop-in replacements for original signed/frozen contracts.

SOURCE_MAP includes actual byte comparisons against locally available public commit157ef969. The requested tidy318bc90 commit is unavailable locally and was not fetched. Ten source comparisons differ, including canonical_loss/canonical_schedule, P2 trainer and chat_template. `configs/AVAILABLE_PUBLIC_SOURCE_DIFFS.patch` exposes those differences. Matching trainer filenames alone is not source equality. The recorded Stage B compatibility bridge must not be redesigned as a fresh optimizer transition.

## Practical order before running any future training

1. Read STAGE_RECIPE and the exact original-hash match fields. Choose stage-specific source roots; preserve license notices.
2. Restore/acquire the exact tokenizer/data/parent identities. F/I raw parents, D2/D3 selection outputs and L1 reserve exclusions are not all in the selected backup. Do not silently substitute a current dataset revision.
3. Supply owner-approved frozen P2 data/review inputs and private P3 generator/checker contract; they are intentionally absent here. Recovered P2 builder and finalizer match their recorded digests; the dynamic earlier pilot helper is included but its execution-era digest is not separately established.
4. On the laptop, port only path/runtime binding and required integration boundaries, preserving training semantics. Reconcile altered hashes explicitly; never disable identity checks to force a launch. Inspect differences against318bc90 when that object is available there.
5. Use the recorded environment as a compatibility target. Installations, smoke tests, training and publication require a later task; none ran here. The extracted source is a faithful review input with explicit gaps, not a promised one-command historical replay.

## Recorded stage-by-stage implementation

### F_TOKENIZER_CORPUS

**Implementation:** `sources/tokenizer/tokenizer/data_preparation/build_tokenizer_training_corpus.py`.

**Supporting record:** `configs/tokenizer_corpus_selection.json`.

**Recorded command / scope:** Recorded F builder identity and frozen contract recovered; complete historical shell argv is UNKNOWN. See exact argparse default expressions and frozen contract; do not synthesize a command and call it recorded..

**Settings and defaults:** Static expression table, plus builder source; workers=6 recorded in F implementation.

**Input:** Frozen eligible source occurrences after D2/D3 and L1 exclusion; source release/file identities in tokenizer_corpus_selection input_bindings.

**Output:** Five canonical JSONL buckets; logical output schema and recorded SHA/bytes in frozen F selection.

**Effective semantics:** No new dedup; preserve eligible occurrences; L1 exclusion uses normalized identity view only while F text bytes remain lossless; rank BLAKE2b128 with frozen full rank tuple; two-pass radix cutoff/exact bin sort..

**Environment / identity:** F build commit recorded 90e14a34, but builder file absent at that local commit; current file recovered by exact 81fc2a63 digest, not guessed Git membership..

### G_TOKENIZER

**Implementation:** `sources/tokenizer/tokenizer/tokenizer_training/train_tokenizer.py`, `sources/tokenizer_validation/runs/g_production_2026-08-21/tools/verify_g_release.py`.

**Supporting record:** `configs/tokenizer_environment_argv.json`, `configs/tokenizer_release.json`.

**Recorded command / scope:** Exact ordered argv in tokenizer_environment_argv; --data five files in recorded order --fields text --no_messages --vocab_size 32000 --min_freq 2 --exclude_hash_manifest ... --corpus_release_manifest ... --corpus_release_manifest_sha256 ... --full_corpus_validation --out_dir ....

**Settings and defaults:** No normalizer/postprocessor; ByteLevel add_prefix_space=false, seven special IDs. Exact copied trainer defaults, not a current alternative..

**Input:** F five JSONL files, 2,511,569 occurrences / 10,000,043,658 UTF8 text bytes; L1 exclusion manifest.

**Output:** tokenizer.json SHA d8f84df58928023edebd809e152b3b38a0dac53b9f887bd2455f427661e9b9ce; shipped tokenizer files verified.

**Effective semantics:** 32000=7+256+31737; full-corpus roundtrip/UNK/special-ID validation historically zero failures. No fresh tokenizer execution..

**Environment / identity:** Python3.10.12/tokenizers0.22.2; trainer hash7bd6b759 at e0408a34.

### I_SELECTION

**Implementation:** `sources/data_selection/pretrain/stage_i_realize_v1.py`.

**Supporting record:** `configs/data_selection_plan.json`.

**Recorded command / scope:** python pretrain/stage_i_realize_v1.py run --plan <exact candidate_i_plan_v7> --expected-plan-sha256 3effe6300c3f383dd788d22b4146064b98f620bed85e5073a5229f5fb26d5e56 --out-dir <I realization> --repo-root <source root>.

**Settings and defaults:** Recorded work/pass1 defaults in original implementation; seed, node_order and selection_rules from exact plan..

**Input:** Frozen graph, source provenance bindings, accepted H token accounting, canonical tokenizer; no source exhaustion replay.

**Output:** Native immutable selected plaintext realization, Stage A10,000,003,234 and B3,000,004,240 serialized tokens.

**Effective semantics:** Original native stage-I module family and implementation_file hashes matched against local 3e6994a8 sources. Exact corpus cannot be reconstructed from tokenizer alone..

**Environment / identity:** environment_contract in plan; no current environment substituted.

### M_PACKING

**Implementation:** `sources/packing/pretrain/stage_m_realize_v1.py`.

**Supporting record:** `configs/packing_plan.json`.

**Recorded command / scope:** python pretrain/stage_m_realize_v1.py run --plan <exact candidate_m_plan_v6> --expected-plan-sha256 6c7af2bdaf00b12fd62e8af0f26960ed2c12317cd87ce290f9b1de055e7b1d21 --out-dir <packed release> --repo-root <source root>.

**Settings and defaults:** ordering_contract, packing_semantics, stage_streams and expected_totals recorded in plan; copied producer defaults separately indexed..

**Input:** Exact accepted I realization; text field and stream order fixed, source disjointness/replay rules preserved.

**Output:** uint16 arrays: A10,000,003,073 tokens, B3,000,002,561; 6,347,659 sequences, one final lookahead per stage.

**Effective semantics:** [BOS] content [EOS], empty text separator; no padding;1840 tail tokens removed; not equivalent to raw text bytes or optimizer input count.

**Environment / identity:** Bound implementation ea081327 and plan environment_contract.

### PRETRAIN_A

**Implementation:** `sources/pretrain_stage_a/pretrain/train_pretrain_with_bench.py`.

**Supporting record:** `configs/pretrain_stage_a_effective.json`, `configs/pretraining_exact_plan.json`, `configs/pretrain_contract_semantics.json`.

**Recorded command / scope:** Exact recorded cmd/args in pretrain_stage_a_effective; not a newly generated launch.

**Settings and defaults:** Effective args override parser. 30/576/1536,9Q/3KV,2048,32000,tied=True; micro8*accum16; seed20260831; samplerA20260832, samplerB20260833, val20260834.

**Input:** Stage-A packed arrays plus full G2 validation and original launch/plan bindings.

**Output:** Transition step38146; final canonical Base is produced later outside runs scope.

**Effective semantics:** Muon matrices + auxiliary AdamW embeddings/norms; lr6e-4,wd.1,clip1,500 warmup,WSD absolute49590,decay44631..49590,minratio.1,bf16,compile. BOS-loss masking/default boundary behavior remains exact trainer/dataset contract, see defaults and model-input masking code; no invented reset at stage boundary..

**Environment / identity:** Private trainer commit6d80423a; recorded run_meta config, dirty flag kept; source Git version separately hash-identified.

### PRETRAIN_B

**Implementation:** `sources/pretrain_stage_b/pretrain/train_pretrain_with_bench.py`.

**Supporting record:** `configs/pretrain_stage_b_effective.json`, `configs/pretrain_contract_semantics.json`, `configs/pretraining_environment.json`.

**Recorded command / scope:** Recorded cmd/args with --resume_full, --resume_step38146, --strict_resume_contract; consumes compatibility-bridge checkpoint, not weights-only init.

**Settings and defaults:** micro8,accum16,absolute schedule retained; data_stage_start_step38146 and stop49590; replay disabled; fail on branch mismatch.

**Input:** Exact Stage-A endpoint and recorded bridge state, Stage-B packed arrays; bridge validation needs original immutable contracts.

**Output:** canonical Base step49590 SHA95ae1201a5ce8a6f838e8e131c59b8779d639f4ec4cf4dbc553928b9b754b86e.

**Effective semantics:** No optimizer or schedule reset. Resume model/optimizer/scaler/RNG and stage sampler cursor under recorded compatibility bridge; 213 tensor equivalence is historical evidence. Across A/B optimizer processed12,999,720,960 positions..

**Environment / identity:** Successor trainer7686fd81; Python3.10.12,torch2.11.0+cu126,numpy2.2.6,RTX4090; isolated source copy distinct from Stage A.

### P2

**Implementation:** `sources/posttraining/sft/train_sft.py`, `sources/posttraining/sft/p2_loss.py`, `sources/posttraining/sft/p2_plan.py`, `sources/posttraining/sft/p2_evaluation.py`, `sources/posttraining/runs/sft_p2_concise_instruction_20260906/data/prepare_p2_data.py`, `sources/posttraining/runs/sft_p2_concise_instruction_20260906/data/finalize_p2_data.py`.

**Supporting record:** `configs/p2_effective_launch.json`, `configs/p2_input_identities.json`.

**Recorded command / scope:** Exact argv retained: python -u -m sft.train_sft with --preserve_messages --loss_normalization effective_batch_tokens --ce_precision selected_fp32 --p2_consumption_plan ...; all other explicit arguments in p2_effective_launch.

**Settings and defaults:** Initial model config comes from accepted Base, NOT resolved_args tie_embeddings=false fallback. Fresh AdamW, betas.9/.95,eps1e-8,weight decay by ndim>=2,lr5e-5,wd.1,warmup38,clip1,micro2*accum16,seq2048,seed20260906,workers2,750updates..

**Input:** Frozen12k/500 rows from exact reviewed smol-smoltalk-derived pools; pinned collection f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc, default/train. Frozen training SHA37bc8aa8765c2ec11521cec4797a9621cfc198f5f6557c4e19962686c41559a8 and validation9acca1a4f94cd15ce1e3c22dbc2de70f16eaf7405e068e078219a2ef52a94796.

**Output:** step750 SHA20afc40096568343c59d8a55e9606667c279f5f1c976908c57de18a0107f4e4e; midpoint375, metrics retained.

**Effective semantics:** Every assistant turn content+EOS supervised; user/system/roles/BOS masked. Preserve original messages; no truncation/injected system. Buffer effective batch, divide each CE sum by total shifted targets across all microbatches; backward then one clip/step. selected FP32 logits CE with BF16 forward.24k row exposures /1,994,854 target observations, no inference quality promotion..

**Environment / identity:** Exact trainer997201f1 and p2_lossce269097 recovered, including untracked code; launch runtime environment retained through original records; no environment installation.

### P3

**Implementation:** `sources/posttraining/runs/p3_basic_instruction_generalization_20260907/runtime/execute.py`, `sources/posttraining/runs/p3_basic_instruction_generalization_20260907/runtime/prepare.py`.

**Supporting record:** `configs/p3_effective.json`, `configs/p3_frozen_bindings.json`, `configs/p3_launch.json`.

**Recorded command / scope:** Recorded train argv in p3_launch; execute.py train uses frozen RUN_CONFIG and UPDATE_PLAN.

**Settings and defaults:** FreshAdamW(.9,.95),eps1e-8,wd0,lr5e-5,32 warmup,cosine640,minratio.1; micro2*accum16,seq512 within2048architecture; seed20260907,workers0,CPUthreads4;2passes;7200s cap,no automatic restart.

**Input:** P2step750 weights only;7168 procedural +3072replay rows; full plan frozen; no private prompts shipped.

**Output:** step320 SHA b11cb018d9f7382e46088f587dc62db9bc25c7d696d399e6b8a001f7ca9cb4e2 and adverse endpoint640; both deliberately retained.

**Effective semantics:** 7 procedural updates then3replay per block; effective-update target-token mean includingEOS; selectedFP32CE/BF16forward. Checkpoint function records model,optimizer,config,step,scheduler,RNG,plan/data hashes. Step320 is alpha blend parent; step640 documents later regression, not silently discarded..

**Environment / identity:** Frozen accepted_source_files bound; private curriculum/checker data source withheld explicitly.

### INTERPOLATION

**Implementation:** `sources/posttraining/runs/p3_retention_two_point_interpolation_20260907/runtime/blend.py`.

**Supporting record:** `configs/interpolation_launch.json`, `configs/p3_frozen_bindings.json`.

**Recorded command / scope:** Recorded argv calls runtime/blend.py; script constants bind two parents and exactly alpha.50/.75.

**Settings and defaults:** CPU FP32 tensor-by-tensor A+alpha*(B-A), no optimizer/backward; verify tied parameter aliasing, same config, finite values and fixed buffers.

**Input:** P2step750 and P3step320, neverstep640; complete input SHA checks in original code.

**Output:** alpha075 SHA1da85cc329d55e92dacf51c36623779558c4c6c9a39d78a34a064f61fcddbe97; alpha050 control retained.

**Effective semantics:** Working reference selected on recorded transfer/retention tradeoff, not best-general-assistant claim; inference-only weight artifact, no fabricated resume optimizer.

**Environment / identity:** Existing recorded CPU FP32 materialization; original argv retained.

### NATIVE_EXPORT

**Implementation:** `sources/posttraining/runs/alpha075_native_inference_export_v1/evidence/scripts/export_weights.py`, `sources/native_inference/inference.py`, `sources/native_inference/src/model.py`.

**Supporting record:** `configs/export_bindings.json`, `configs/native/config.json`, `configs/native/requirements-inference-tested.txt`.

**Recorded command / scope:** Original export_weights.py entrypoint; build_bundle excerpt documents only actual accepted-generation precision adaptation. Private publication/review-copy orchestration excluded..

**Settings and defaults:** FP32 safetensors export with strict state/config validation and tied embeddings; nativeCLI CUDA only,greedy,bf16_native/fp32_math,1..384new tokens,total<=2048.

**Input:** alpha075 and bound model/chat/special/generation implementations.

**Output:** native weight SHA4396efb7a52b047e7fdf513e46d1b401dfc70582d3aca1f9cb5a07e97d426ef1; no tensor bytes included here.

**Effective semantics:** 213 named state entries preserved; historical8source/export parity pairs. Whole standalone on Pod protected; previously deleted staging weights are not restored..

**Environment / identity:** Python3.10.12,torch2.11.0+cu126,numpy2.2.6,tokenizers.22.2,safetensors.8.0.

### PUBLIC_BENCHMARK

**Implementation:** `sources/benchmark/runs/petitgpt_frozen_reference_public_benchmark_fp32_v2/runtime/evaluate_v2.py`, `sources/benchmark/runs/petitgpt_frozen_reference_public_benchmark_fp32_v2/runtime/aggregate_report.py`.

**Supporting record:** `configs/benchmark_protocol.json`, `configs/benchmark_models.json`.

**Recorded command / scope:** Actual evaluator/aggregator recovered; input requests, run manifests and original raw outcomes must be bound externally. Full recorded launch argv not claimed if absent..

**Settings and defaults:** FP32params/forward/logsoftmax,autocast+TF32off,MATHSDPA,unpaddedbatch1; zero-shot raw completion; no chat/BOS/scoredEOS; firstmax ties;acc_norm Unicode-character normalization.

**Input:** ARC-Easy2376test and PIQA1838validation; three frozen model/tokenizer identities; pinned harness methods loaded via AST from bundled licensed helper files.

**Output:** Published versioned table: alpha075 ARC57.74/52.36,PIQA63.49/62.30; no new scores computed.

**Effective semantics:** Native protocol-compatible evaluation, not full installed harness; request boundary/per-token checks and finite numerical validation; aggregation preserves incomplete status and does not merge score versions.

**Environment / identity:** Exact environment guards in runtime/common.py; see handoff dependency table.

## Inputs and data construction boundary

Tokenizer inputs are JSONL objects with lossless text, canonical_source, canonical_release_id, physical_row_index and cleaned_text_sha256; ranking and byte budgets use the frozen F contract. Selected I documents are not interchangeable with raw pool documents; Stage M consumes exact selected ordering. Packed uint16 data has 2048-position blocks plus stage-final lookahead, while sampler accumulation drops alignment remainder. The4token-accounting boundaries are preserved in the projected plan.

P2 rows preserve original message lists and supervision metadata; builders consume previously reviewed eligible pools and immutable review/defer records. The seven labels belong to the pinned smol-smoltalk collection, not seven independently pinned repository configs. The released source notice describes unresolved third-party terms. Private rows and held-out messages are not included in this handoff. P3 private procedural templates and checks are deliberately withheld; absence is an explicit dependency gap, not a license to invent a replacement curriculum.

For a frozen artifact, use its original SHA rather than a candidate projected-config SHA. Restoring a path does not change its identity requirements. Logs do not replace removed optimizer/sampler/RNG states. Equal model weights do not prove full checkpoint equivalence.

## Significant non-release research track

| Track | Retained scientific distinction | Relationship to release |
|---|---|---|
| P4 micro / QA / dose6 / R1 | Controlled acquisition and retention failures; dose156/468 trajectory | Downstream alpha075 line, not ancestry |
| DP1 DPO and KD1 chosen-CE | Matched preference/CE comparison; preserve negative results | Fresh branches, not alpha075 updates |
| RKD1 response KD and RKD2 LoRA | Repaired response data, full-vs-adapter behavior; adapter requires exact base/tokenizer | Downstream branches |
| KD2 soft-logit | CE20 control + KD20 treatment, exact teacher/student/tokenizer/logit alignment | Independent external SmolLM2 lab |
| Unified Base SFT |403 branch/lowest observed NLL,806 trajectory,1209 endpoint | Starts from Base, not alpha075 |
| Unified A/B and interpolation grid | Both arms and five-point grid controls retained | Separate loss-allocation branch |

No new checkpoint quality selection was performed. See the published technical report §§6–7 for original conclusions. Private run-retention decisions are intentionally excluded from this source package.

## Precise unresolved gaps

- **G_PUBLIC_BASELINE** (integration): Requested tidy commit unavailable in both local repositories; comparison uses explicitly labeled 157ef969 ancestor only.
- **G_PRIVATE_P3_CONSTRUCTION** (p3): Private procedural curriculum/checker modules may reveal held-out templates; omitted. Frozen data IDs and actual trainer retained; acquire owner-approved generator/data contract on laptop, never substitute prompts.
- **G_RELOCATED_IMMUTABLE_BINDINGS** (all): Path-sanitized sources/configs differ from original frozen hashes; require a reviewed public path/binding adapter. Do not silently weaken existing fail-closed checks or treat projected records as original contracts.
- **G_DATA_ACQUISITION** (data): Raw source and instruction selection are not distributable here; exact historical input access/licensing and upstream acquisition needed. Hash identities do not grant permission or guarantee bit-identical redownload.
- **G_DYNAMIC_RUNTIME_ARTIFACTS** (export_and_benchmark): Runtime frozen input/code/parity manifests and benchmark request materialization are external artifacts; handoff includes actual algorithms and protocol, not private prompts or executable publication orchestration. Local integration must bind these inputs explicitly.

The F builder missing-at-commit issue was resolved by finding the exact current81fc2a63 bytes; it is not counted as a remaining gap. P2 top-level builders were also recovered by exact recorded digests. The independent G verifier is supplied as available source, with its current byte identity, not falsely assigned a historical commit.

## Validation and licensing

All packaged source/config bytes are hashed and Python syntax is parsed without imports. No torch/transformers/tokenizer module was imported, no model was loaded, no corpus was rehashed. Static argparse defaults are extracted from the copied implementation; effective launch arguments and inherited checkpoint configuration take precedence. Library versions alone do not guarantee cross-hardware bit identity.

Author-controlled code/model/tokenizer licensing and documentation exceptions remain as in `notices/`. Pinned harness helper licensing is included beside that source. This package is a local publication-review input, not legal clearance or a grant to redistribute omitted data.

## Final execution-source binding verification

Stage A reports `git_dirty=true`. Its complete 12-file AST execution closure recovered from commit `6d80423a` hashes to `bbd49b9d73d3cb2fa18aacb3eee861a901e5a7511ed334b85b37239ab1d50043`, exactly the historical runtime binding. Stage B's 13-file closure at `7686fd81` hashes to `1086af0b6821b2fdc4b2850371845c992f831dfcd84a6d504d2938fad003e75d`, also matching. The Stage B successor-head compatibility bridge and package initializers are included. These assertions concern original bytes; relocated candidate contracts still require the documented integration adapter. See `configs/TRAINER_EXECUTION_BINDINGS.json`.

Inputs are toks[:-1], targets toks[1:]. Effective mask_bos_in_loss=true; mask_repeated_eos_in_loss=true (mask the later target in consecutive EOS targets); mask_last_label_in_loss=false, so valid final labels remain. EOS weight=1.0. These are the recorded trainer/dataset settings, not a redesigned boundary policy.

Interpolation runtime and the benchmark core/likelihood/prepare source originals also match their recorded file manifests. Report aggregation without a historical code-freeze entry is identified by its current original-source hash, without claiming a historical code binding.

Stage A initial training is recorded separately in `configs/pretrain_stage_a_initial_effective.json`: no resume, stop at38146. Its original governed-run contract binds the same verified source bundle. The previously named `pretrain_stage_a_effective.json` is the accepted endpoint recovery invocation (resume38146/stop38146); it must not be mistaken for a second training pass.