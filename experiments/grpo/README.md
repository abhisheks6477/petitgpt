# GRPO implementation (not a measured research-v1 stage)

This code and its active tests implement GRPO, prompt-data preparation and reward
functions. The release report does not establish a completed GRPO experiment.
There are no GRPO scores to add to the release and no GRPO update in alpha075.

From any working directory, help is available without constructing a model:

```bash
python /path/to/petitgpt/experiments/grpo/grpo.py --help
python /path/to/petitgpt/experiments/grpo/prepare_grpo_data.py --help
```

Use the explicit tokenizer path
`/path/to/petitgpt/tokenizer/releases/tokenizer_v1/tokenizer.json` and caller-selected
data/output paths for new experiments. Read the CLI/module contracts before any
training. Imports now use `experiments.grpo`; shared model, token and optimizer
components still come from the repository's `src/`. Tests remain under `tests/`.
See the [experiment index](../README.md) for the measured/unmeasured distinction.
