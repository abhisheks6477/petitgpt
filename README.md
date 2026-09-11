# petitgpt

Yang Qi

Research v1 describes the selected alpha075 native model: 124,635,456 parameters, 30 layers, width 576, 9 query / 3 KV heads. Lineage: Base -> P2 -> P3 step320 -> interpolation. Later distillation, unified Base-SFT, DPO and LoRA updates are absent.

[Complete report](docs/petitgpt-v1/TECHNICAL_REPORT.md) · [Model card](docs/petitgpt-v1/MODEL_CARD.md) · [Native run guide](docs/petitgpt-v1/RUN_GUIDE.md) · [Hugging Face model](https://huggingface.co/yqi0/petitgpt)

Frozen FP32 zero-shot likelihood: ARC-Easy acc 0.5774410774410774 / acc_norm 0.5235690235690236; PIQA acc 0.6349292709466812 / acc_norm 0.6229597388465724. These protocol-bounded diagnostics do not establish reliable free generation. Whole-answer Python content passed 0/46 while interface passed 42/46; see the versioned report for protocols and limitations.

Author-controlled code/model/tokenizer: [Apache-2.0](LICENSE). Author-written documentation: [CC BY 4.0](docs/petitgpt-v1/DOCUMENTATION_LICENSE.md), an explicit exception to the root code licence. [Source notice](SOURCE_NOTICE.md) and [third-party notices](THIRD_PARTY_NOTICES.md) preserve upstream terms and uncertainty. Research is intended use, not an extra licence restriction.

## Historical experiments

The [historical README](docs/petitgpt-v1/HISTORICAL_README.md) preserves the earlier project narrative; its architecture and run results must not be substituted for the current release.
