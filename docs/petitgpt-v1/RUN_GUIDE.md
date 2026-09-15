# PetitGPT native run guide

Run the released **alpha075** checkpoint with the native PyTorch CLI or Python API. A CUDA GPU is required. For training your own model, see [Reproducibility](../../TRAINING_AND_REPRODUCIBILITY.md).

## 1. Download the model

With the Hugging Face CLI installed, download the complete bundle at the same fixed revision used by the project README:

```sh
hf download yqi0/petitgpt \
    --revision 7bf3df96e6880b242b2907d1e68093435feacd75 \
    --local-dir ./artifacts/petitgpt-research-v1

(cd ./artifacts/petitgpt-research-v1 && sha256sum -c SHA256SUMS)
```

In the examples below, replace `/path/to/bundle` with the downloaded directory. Keep all files together, including `src/`: `inference.py` imports from it. The bundle's `SHA256SUMS` verifies its downloaded files; the [documentation manifest](SHA256SUMS) in this GitHub directory covers a different file set.

## 2. Requirements

A **CUDA GPU is required** by this CLI. The tested configuration is Python 3.10.12 with:

```
torch==2.11.0+cu126
numpy==2.2.6
tokenizers==0.22.2
safetensors==0.8.0
```

on an NVIDIA GeForce RTX 4090 (driver 580.178.04). The bundle includes `requirements-inference-tested.txt` ([source copy](../../inference_native/requirements-inference-tested.txt)). Prepare this environment before inference. If you have a local wheelhouse containing the matching packages and their dependencies, an offline installation is:

```sh
python -m pip install --no-index --find-links /path/to/local/wheelhouse \
    -r /path/to/bundle/requirements-inference-tested.txt
```

The wheelhouse path is a placeholder; the model download does not supply these packages. Matching versions do not guarantee bit-identical results on arbitrary hardware or untested software.

## 3. Command line

Choose either `--prompt` for one user message or `--messages-json` for a conversation:

```sh
python /path/to/bundle/inference.py \
    --model-directory /path/to/bundle \
    --prompt "Say hello in one sentence." \
    --profile bf16_native \
    --max-new-tokens 32

python /path/to/bundle/inference.py \
    --model-directory /path/to/bundle \
    --messages-json /path/to/messages.json \
    --profile fp32_math \
    --max-new-tokens 32
```

For the second command, save this example as `/path/to/messages.json`:

```json
[{"role":"system","content":"Use short sentences."},
 {"role":"user","content":"My name is Lin."},
 {"role":"assistant","content":"Hello, Lin."},
 {"role":"user","content":"What name did I give you?"}]
```

These prompts illustrate the input format. Recorded model outputs and their review labels are available in the report's [success and failure cases](TECHNICAL_REPORT.md#85-qualitative-cases-successes-and-failures).

## 4. Python API

Add the downloaded bundle to Python's import path, then load the model and move it to CUDA:

```python
import sys
from pathlib import Path

bundle = Path("/path/to/bundle").resolve()
sys.path.insert(0, str(bundle))
from inference import load_bundle, generate

model, tokenizer = load_bundle(bundle)
model = model.to("cuda").eval()
result = generate(
    model, tokenizer,
    [{"role": "user", "content": "Say hello."}],
    cap=32,
    profile="fp32_math",
)
print(result["output_text_raw_including_terminal_eos"])
```

## 5. Input contract

`messages` must be a non-empty JSON array of objects with **exactly** the fields `role` and `content`. Each content value must be a string containing non-whitespace text. A conversation is an optional initial system turn followed by alternating user/assistant turns, **ending in user**. A plain `--prompt` becomes one user message.

Rejected, by design, with an explicit error rather than a repair:

| Case | Recorded rejection |
|---|---|
| Missing `content` | `Each message must contain only role and content` |
| Unsupported role (e.g. `tool`) | `message 0 has invalid role 'tool'` |
| Any extra message field | `Each message must contain only role and content` |
| Conversation ending in `assistant` | `chat must end with a non-empty user turn; got 'assistant'` |
| Prompt + budget over 2,048 | `Context overflow: prompt plus token budget exceeds 2048` |
| `max_new_tokens` outside 1..384 | `max_new_tokens must be 1..384` |

`default_system=None`: a supplied system turn and full history are retained, with **no injected default and no content-text normalization**. Role labels are stripped and lowercased before validation. Literal special-token spellings inside content — for example the literal spelling `[EOS]` — are encoded as ordinary text and cannot inject control IDs.

Native token structure:

```
[BOS] <|system|> system <|user|> user <|assistant|> assistant [EOS] … <|user|> user <|assistant|>
```

The system segment is omitted when absent. `[BOS]` occurs once; `[EOS]` closes completed assistant turns; no duplicate assistant prefix is added. IDs are fixed: `[PAD]=0`, `[UNK]=1`, `[BOS]=2`, `[EOS]=3`, `<|system|>=4`, `<|user|>=5`, `<|assistant|>=6`.

## 6. Precision profiles

Both profiles store FP32 parameters. They differ only in the forward numerical path, and **they are not asserted to agree with each other**.

| | `bf16_native` | `fp32_math` |
|---|---|---|
| Forward | CUDA BF16 autocast | FP32, autocast off |
| matmul TF32 | off | off |
| cuDNN TF32 | **on** | off |
| `float32_matmul_precision` | highest | highest |
| SDPA backend | native backends enabled | MATH |

## 7. Decoding and stopping

Greedy only: `temperature=0`, `top_k=0`, `top_p=1`, `EOS=3`. Default `max_new_tokens` is **384**; an explicit lower integer in `1..384` is supported. Prompt length plus budget must fit 2,048 — there is no cropping.

The CLI prints a JSON object; the Python API returns the same fields as a dictionary. Key fields are:

| Field | Meaning |
|---|---|
| `prompt_token_ids` | Full encoded conversation, including control tokens |
| `generated_token_ids` | Newly generated tokens, including a terminal EOS when produced |
| `output_text_raw_including_terminal_eos` | Decoded generated tokens with special tokens visible |
| `output_text_for_scoring` | Decoded output with only the terminal EOS removed |
| `stop_reason` | `eos` or `max_new_tokens` |
| `generated_tokens_including_eos` | Number of newly generated tokens |

Generation applies no answer cleanup or retries. Reaching the token cap can leave an answer unfinished. The IFEval results use a separate evaluation program with a 1,280-token generation cap; that is not a supported value for this CLI.

## 8. Format support and limits

Native PyTorch CUDA inference only. There is **no implemented or tested** Transformers `AutoModel`, GGUF, ONNX, vLLM or llama.cpp path. `special_tokens_map.json` is descriptive native metadata, not a loader contract.

The recorded import check ran in a fresh process with network and out-of-bundle repository access denied, and completed without denied access. Eight source/export fixture pairs matched within their numerical profiles. These checks establish behavior in the measured environment; they do not establish portability to other hardware or a fresh installation. See [export validation](TECHNICAL_REPORT.md#9-reproducibility-and-export) for scope and evidence.

## 9. Troubleshooting

| Symptom | What to check |
|---|---|
| Missing `src` or model file | Download the complete bundle and preserve its directory layout. For the API, add that directory to `sys.path` as shown above. |
| CUDA unavailable | Use a CUDA-enabled PyTorch installation and a visible compatible NVIDIA GPU; the released generator requires CUDA. |
| Context overflow | Shorten the supplied conversation or lower the generation budget. Both together must fit 2,048 tokens. |
| Invalid message fields, content, or role order | Follow the input contract in §5; history must end with a non-empty user turn. |
| Answer stops mid-sentence | Inspect `stop_reason`. If it is `max_new_tokens`, increase the cap only within the CLI and context limits. |

## 10. Interpreting output

Factual answers and generated code require independent review. Correct formatting or a plausible function signature does not establish a correct answer; see the [model card](MODEL_CARD.md) for measured capabilities and limitations.

Author: Yang Qi. Documentation: [CC BY 4.0](DOCUMENTATION_LICENSE.md).
