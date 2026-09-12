# Historical tokenizer versions

These are byte-preserved historical artifacts, not current defaults. The current
[seven-special-token release](../../tokenizer/README.md) has its own metadata.

| Directory | Version evidence |
|---|---|
| [12-layer-four-special/](12-layer-four-special/) | `tokenizer_12layers.json`, added in `7c9e55d7778a9a65d6d47e59c7a345d3b8e7d0c1` ("old tokenizer"); four specials, 32k vocabulary, prefix space false. |
| [pretrain-prefix-space-four-special/](pretrain-prefix-space-four-special/) | `tokenizer_pretrain_nospecial.json`, added in `7d9b0216d3854fd7506b9ae38e1c18059ba51735` ("continue pretraining"); four registered specials despite its name, prefix space true, different BPE model. |
| [four-special-root-metadata/](four-special-root-metadata/) | Original `tokenizer_config.json` and `special_tokens_map.json`: only PAD/UNK/BOS/EOS, no chat role tokens. Preserved as shared historical root metadata; the files do not uniquely identify which old BPE bytes they accompanied. |

The metadata predates the current release and cannot describe its seven-token
contract. Its exact attachment to either old tokenizer is unresolved; it is not
silently assigned to one. No files were resaved. The [migration map](../MIGRATIONS.csv)
records original paths, Git blobs and SHA256. Historical diagnostics/configs retain
their old path strings; use the recorded original checkout to replay them.
