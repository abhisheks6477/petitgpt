# Research outside the released weights

The released ancestry is Base49590 → P2step750 → P3step320 → alpha075.
These experiments are separate branches. Start with the unchanged
[experiment ledger](../docs/petitgpt-v1/tables/RESEARCH_EXPERIMENT_LEDGER.csv)
and [technical report](../docs/petitgpt-v1/TECHNICAL_REPORT.md) for measured results,
protocol versions and negative findings.

| Track | Implementation and evidence status |
|---|---|
| [GRPO](grpo/README.md) | Implementation and unit tests only; no research-v1 measured run established. |
| DPO / chosen-response CE | Reusable [DPO](../dpo/) and [SFT](../sft/) engines remain in functional directories. DP1/KD1 measurements exist in the ledger; these public engines are not claimed to be exact run-specific executables. |
| Response KD / LoRA | [Distillation utilities](../distill/) support response-data work. RKD1/RKD2 measured outcomes are in the ledger; a generic wrapper does not reproduce their private run contracts. |
| Soft-logit KD | KD2 used external SmolLM2 teacher/student models, with distinct tokenizer/logit alignment. No claim that the response-CE wrapper implements this lab. |
| P4, dose6, R1, unified Base-SFT, A/B/grid | Measured research in the ledger; not alpha075 stages. Their retained negative results remain public. Full run-specific closures/data are not all distributed. |

`src/model_moe.py` is also a tested implementation, with no released-weight or
measured research-v1 claim. Shared modules are kept in their functional directories.
Earlier superseded run snapshots are indexed under [legacy/](../legacy/README.md).
