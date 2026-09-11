# Source notice

petitgpt by Yang Qi; selected checkpoint alpha075. Parameter lineage: Base -> P2 step750 -> P3 step320 -> interpolation (0.75 toward P3). Later DeepSeek-response-KD, unified Base-SFT, DPO, soft-KD and LoRA updates are absent.

Pretraining source and tokenizer linkage have been established by the existing digest records. The tokenizer corpus uses six of the eight frozen releases; PES2O and StackExchange occur in pretraining, not tokenizer training. Selected source counts are not per-source consumed-token measurements. No new corpus audit is claimed.

## Pretraining and tokenizer sources

The following terms are builder-recorded metadata at pinned revisions, not independent verification of every licensor's rights. Dataset authors and contributors retain their rights. Exact mixture and transport revisions are preserved in the companion PRETRAIN_SOURCE_MIXTURE.csv.

| Dataset / config | Pinned revision | Recorded terms |
|---|---|---|
| HuggingFaceTB/smollm-corpus / fineweb-edu-dedup | `3ba9d605774198c5868892d7a8deda78031a781f` | odc-by-1.0 |
| HuggingFaceTB/dclm-edu / default | `dbad8ad71224482740cd9c9d353591adbf62fe04` | cc-by-4.0 |
| HuggingFaceFW/finewiki / en | `8bd13e72e6a002407649b3e898535f42ceb1aeb9` | cc-by-sa-4.0 |
| common-pile/stackv2_edu_filtered / default | `c354dbe88469a1153e97c6a63ac50591849654de` | per-record metadata.license (Software Heritage permissive subset) |
| HuggingFaceTB/smollm-corpus + HuggingFaceFW/finephrase / cosmopedia-v2 + tutorial | `3ba9d605774198c5868892d7a8deda78031a781f + 78cf4a5ed0099214979c094c963e699c19163838` | odc-by-1.0 (both) |
| allenai/dolmino-mix-1124 / pes2o | `a319f19eef1e257417b11ea8c30da266ae175557` | odc-by-1.0 |
| allenai/dolmino-mix-1124 / stackexchange | `a319f19eef1e257417b11ea8c30da266ae175557` | cc-by-sa |

Publisher datasets: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus ; https://huggingface.co/datasets/HuggingFaceTB/dclm-edu ; https://huggingface.co/datasets/HuggingFaceFW/finewiki ; https://huggingface.co/datasets/common-pile/stackv2_edu_filtered ; https://huggingface.co/datasets/HuggingFaceFW/finephrase ; https://huggingface.co/datasets/allenai/dolmino-mix-1124 .

## Instruction components

The collection is HuggingFaceTB/smol-smoltalk, revision `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`, default/train. Its pinned card has an Apache-2.0 badge. P2's seven labels are source-column values, not separate repositories. Parent/component correspondence is documentary, not a per-row join; exact component revisions remain unestablished. Parent and component notices below are CURRENT_ONLY observations retrieved on 2026-09-10, not terms proven contemporaneous with training.

- **openhermes-50k**: teknium/OpenHermes-2.5. Recorded metadata/grant: no licence in component-card metadata. component dataset card metadata at head b82037821055c377bed0d495e72e46de3bc72e84 (retrieved 2026-09-10T17:53:40Z)
- **smol-contraints**: Smol-contraints. Recorded metadata/grant: Apache-2.0. parent collection card at head 5feaf2fd3ffca7c237fc38d1861bc30365d48ffa (retrieved 2026-09-10T17:52:48Z): "All the new datasets (Smol-Magpie-Ultra, Smol-contraints, Smol-rewrite, Smol-summarize) are licensed under Apache 2.0."
- **smollm-rewrite-30k**: Smol-rewrite. Recorded metadata/grant: Apache-2.0. parent collection card at head 5feaf2fd3ffca7c237fc38d1861bc30365d48ffa (retrieved 2026-09-10T17:52:48Z): "All the new datasets (Smol-Magpie-Ultra, Smol-contraints, Smol-rewrite, Smol-summarize) are licensed under Apache 2.0."
- **smol-magpie-ultra-short**: Smol-Magpie-Ultra. Recorded metadata/grant: Apache-2.0. parent collection card at head 5feaf2fd3ffca7c237fc38d1861bc30365d48ffa (retrieved 2026-09-10T17:52:48Z): "All the new datasets (Smol-Magpie-Ultra, Smol-contraints, Smol-rewrite, Smol-summarize) are licensed under Apache 2.0."
- **self-oss-instruct**: bigcode/self-oss-instruct-sc2-exec-filter-50k. Recorded metadata/grant: odc-by. component dataset card metadata at head 356bb069eee815daa6e23e9a282eeefe1490ad44 (retrieved 2026-09-10T17:53:40Z)
- **smol-summarize-20k**: Smol-summarize. Recorded metadata/grant: Apache-2.0. parent collection card at head 5feaf2fd3ffca7c237fc38d1861bc30365d48ffa (retrieved 2026-09-10T17:52:48Z): "All the new datasets (Smol-Magpie-Ultra, Smol-contraints, Smol-rewrite, Smol-summarize) are licensed under Apache 2.0."
- **everyday-conversations**: HuggingFaceTB/everyday-conversations-llama3.1-2k. Recorded metadata/grant: apache-2.0. component dataset card metadata at head 14f543216b9ba42b6b951dc5bd199460d193b162 (retrieved 2026-09-10T17:53:41Z)

The parent publisher limits its Apache grant to its newly generated subsets and points to component-specific terms for incorporated datasets. The self-oss-instruct ODC-By metadata differs from the collection badge; a declaration difference is not proof of legal incompatibility. OpenHermes-2.5 refers to component-specific licences; its empty card licence metadata does not resolve those terms, and exact upstream-component rows for this subset remain unresolved. Ordinary historical unknowns have not been newly resolved.

Sources: https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk/tree/f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc ; https://huggingface.co/datasets/HuggingFaceTB/smoltalk ; https://huggingface.co/datasets/teknium/OpenHermes-2.5 ; https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k ; https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k .

No raw corpus, frozen evaluation prompts or model answers are distributed. Author licensing does not relicense upstream datasets or clear third-party rights. See DOCUMENTATION_LICENSE.md for the scoped grant.
