# Historical ARC-Challenge, HellaSwag and IFEval source

These are the archived programs used for **benchmark extension V1**
(ARC-Challenge and HellaSwag) and **IFEval completion V2**, the campaigns reported
in the [technical report](../../../../docs/petitgpt-v1/TECHNICAL_REPORT.md#8-evaluation).
Python and YAML files are copied byte-for-byte from the original evidence
archives. This publication adds source visibility; it does not rerun evaluation.

## Which code produced which results?

| Campaign | Actual execution source | Published results |
|---|---|---|
| ARC-Challenge and HellaSwag | [likelihood.py](run/runtime/likelihood.py); independent aggregation in [aggregate_likelihood.py](run/runtime/aggregate_likelihood.py) | [PUBLIC_BENCHMARK_RESULTS.csv](../../../../docs/petitgpt-v1/tables/PUBLIC_BENCHMARK_RESULTS.csv) |
| Final IFEval completion V2 | [generate_ifeval_v2.py](run/ifeval/completion_v2/runtime/generate_ifeval_v2.py), [effective_config.py](run/ifeval/completion_v2/runtime/effective_config.py), [verify_ifeval_v2.py](run/ifeval/completion_v2/runtime/verify_ifeval_v2.py) | [IFEVAL_RESULTS.csv](../../../../docs/petitgpt-v1/tables/IFEVAL_RESULTS.csv) |
| Earlier ARC-Easy and PIQA FP32 V2 | [evaluate_v2.py](../benchmark/runs/petitgpt_frozen_reference_public_benchmark_fp32_v2/runtime/evaluate_v2.py), in the existing snapshot | ARC-Easy/PIQA rows in the same benchmark CSV |

[prepare.py](run/runtime/prepare.py) prepares task rows, token boundaries and
chat inputs. Likelihood evaluation reuses the original FP32 scorer methods from
the existing [common.py](../benchmark/runs/petitgpt_frozen_reference_public_benchmark_fp32_v2/runtime/common.py).
IFEval uses the pinned [programmatic verifiers](run/upstream/lm_eval/tasks/ifeval)
and per-model native chat formatting.

The older [generate_ifeval.py](run/runtime/generate_ifeval.py) and completion V1
files preserve intermediate history. The final published IFEval scores come
from **completion V2**. Its effective-config helper resolves installed Transformers
defaults and verifies frozen greedy generation settings. The V2 driver retains
the original verified PetitGPT smoke-test bridge.

The separate [reader_evaluate.py](../../reader_evaluate.py) is the later reader
workflow for supplied local ARC-Easy/PIQA inputs. It does not run these three
additional tasks.

## Layout and execution requirements

`run/` preserves the original relative layout, combining the extension, canonical
template continuation, and final V2 archives. Later versions of the same path
take precedence. The initial extension generator remains in
`run/ifeval/completion_v1/generate_ifeval_pre_continuation.py`.

These are historical execution snapshots, **not a self-contained runnable
benchmark package**. A fresh clone does not contain all their runtime inputs:

- Scripts expect a sibling `source/` checkout at commit
  `b82ede26bff93423bd05fa08b8cae6ebd62d1dee`, including the existing benchmark and
  native-inference trees. Shared source hashes were checked when publishing.
- Model checkpoints, tokenizer/model snapshots, official datasets, normalized
  task rows, token encodings and per-model IFEval generation inputs are external.
- Historical protocol/identity manifests, code-freeze records, fixture receipts,
  smoke results and continuation records are required by the original assertions.
  Some bind absolute paths from the original environment.
- Inference requires the recorded CUDA environment. IFEval verification also
  expects isolated `evaluator_deps/` and `nltk_data/` directories. Recorded package
  versions are in [ENVIRONMENT_SUMMARY.json](ENVIRONMENT_SUMMARY.json).

Restoring only directory paths is insufficient: original input hashes and gate
records must also match. Scripts retain their original checks and output behavior.
Preparation, routing, config regression, freeze/bridge and delivery scripts may
perform work at import time; they are not import-safe libraries. Delivery scripts
refer to historical local packaging and audit context.

For an independently configured new run, use separate outputs and record new
identities; do not treat a changed run as the original measurement. Public
[evaluation protocols](../../../../docs/petitgpt-v1/provenance/EVALUATION_PROTOCOLS.json)
describe task revisions, formatting, numeric policies and scoring.

## Provenance and third-party code

[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) records archive SHA-256 values,
member paths, byte counts and per-file hashes, plus published source bindings
from the original IFEval V2 code freeze. It is separate from the older reader
source manifests, which remain unchanged. Raw datasets, model outputs, private
requests and machine environment dumps are outside this source publication.

Vendored task files come from lm-evaluation-harness commit
`b954108c9baaaa934b4ad842033b31a97ee30816`; original URLs and hashes are in
[SOURCE_IDENTITIES.json](run/protocol/SOURCE_IDENTITIES.json).
See [upstream notices](run/upstream/NOTICE.md) for retained EleutherAI and Google
Research attribution and license texts.

Publication checks verify archive/source hashes, V2 code-freeze bindings, Python
syntax and documentation links. They do not establish a new model run or a
clean-machine reproduction of the published scores.
