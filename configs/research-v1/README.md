# Research-v1 execution evidence

Start with [Training and reproducibility](../../TRAINING_AND_REPRODUCIBILITY.md).
These JSON files preserve the handoff's projected effective launch/config records.
**They are not YAML configurations accepted by a new training loader, and are not
byte-identical original frozen contracts.** Original/candidate hashes and source
identities are in [SOURCE_MAP](../../recipes/research-v1/provenance/SOURCE_MAP.json);
[INTEGRATION](../../recipes/research-v1/provenance/INTEGRATION.json) maps them to this checkout.

- `STAGE_RECIPE.json`: source/evidence/semantics index for all release stages.
- `STATIC_ARGUMENT_DEFAULTS.json`: source-extracted defaults; recorded explicit
  launch arguments and inherited checkpoint config take precedence.
- `pretrain_stage_a_initial_effective.json`: actual initial A launch, no resume.
  `pretrain_stage_a_effective.json` is endpoint recovery at 38146, not a new pass.
- `pretrain_stage_b_effective.json`, `pretrain_contract_semantics.json`,
  `pretraining_exact_plan.json`, `TRAINER_EXECUTION_BINDINGS.json`: full-state
  transition, optimizer/loss/schedule and verified historical source closures.
- `p2_effective_launch.json`, `p2_input_identities.json`: actual P2 argv and data
  identities. The fallback `tie_embeddings=false` parser value is superseded by
  the tied Base checkpoint configuration.
- `p3_effective.json`, `p3_frozen_bindings.json`, `p3_launch.json`: P3 recipe and
  identities of omitted curriculum/checker/frozen inputs.
- `interpolation_launch.json`, `export_bindings.json`: fixed parents and native export.
- `benchmark_protocol.json`, `benchmark_models.json`: actual pinned likelihood
  protocol, task revisions, canonical row hashes and model identities.
- `tokenizer_*`, `data_selection_plan.json`, `packing_plan.json`: historical input
  contracts, revisions and ordering. The tokenizer payload is kept only at its
  canonical versioned directory, not duplicated here.

`AVAILABLE_PUBLIC_SOURCE_DIFFS.patch` records the handoff's older comparison to
157ef969. The integration record adds comparisons against the actual 318bc90 base;
it does not overwrite that historical evidence. Public data distribution was
limited to source/config identities and authored synthetic fixtures; no run-specific
training or held-out rows, checkpoints, raw outcomes or private retention inventory
were added.
