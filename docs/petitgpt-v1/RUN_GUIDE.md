# petitgpt native run guide

Author: Yang Qi. Documentation: CC BY 4.0. Download the loose model repository files together, preserving the src/ directory. The exact release file list and hashes are in SHA256SUMS. Model weights, tokenizer, config and executable code retain the accepted export bytes.

## 2. Requirements

A **CUDA GPU is required** by this CLI. The tested configuration is Python 3.10.12 with:

```
torch==2.11.0+cu126
numpy==2.2.6
tokenizers==0.22.2
safetensors==0.8.0
```

on an NVIDIA GeForce RTX 4090 (driver 580.178.04). Prepare the environment separately, from locally supplied wheels:

```sh
python -m pip install --no-index --find-links /path/to/local/wheelhouse \
    -r /path/to/bundle/requirements-inference-tested.txt
```

No packages were installed during the export itself. Matching versions do not guarantee bit-identical results on arbitrary hardware or untested software.

## 3. Command line

Replace `/path/to/bundle` with the real extracted location.

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

A synthetic `messages.json`:

```json
[{"role":"system","content":"Use short sentences."},
 {"role":"user","content":"My name is Lin."},
 {"role":"assistant","content":"Hello, Lin."},
 {"role":"user","content":"What name did I give you?"}]
```

> **Illustrative and unexecuted.** The greeting above is an authored example showing command syntax only. It was written for this guide, was not run, and is not a demonstration of model quality. Do not put frozen evaluation prompts or their outputs into a public demo.

## 4. Python API

With the extracted bundle directory on `sys.path`:

```python
from inference import load_bundle, generate

model, tokenizer = load_bundle("/path/to/bundle")
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

`messages` must be a JSON array of objects with **exactly** the fields `role` and `content`. A conversation is an optional initial system turn followed by alternating user/assistant turns, **ending in user**. A plain `--prompt` becomes one user message.

Rejected, by design, with an explicit error rather than a repair:

| Case | Recorded rejection |
|---|---|
| Missing `content` | `Each message must contain only role and content` |
| Unsupported role (e.g. `tool`) | `message 0 has invalid role 'tool'` |
| Any extra message field | `Each message must contain only role and content` |
| Conversation ending in `assistant` | `chat must end with a non-empty user turn; got 'assistant'` |
| Prompt + budget over 2,048 | `Context overflow: prompt plus token budget exceeds 2048` |
| `max_new_tokens` outside 1..384 | `max_new_tokens must be 1..384` |

`default_system=None`: a supplied system turn and full history are retained, with **no injected default and no text normalization**. Literal special-token spellings inside content — for example the literal spelling `[EOS]` — are encoded as ordinary text and cannot inject control IDs.

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

Output JSON preserves the generated token IDs, the raw text **including the terminal EOS**, and a stop reason of `eos` or `max_new_tokens`. There is no answer cleanup, no fact fixing, no best-of-N, and no retry. A token cap can truncate an answer mid-sentence; that is the recorded behaviour, not a defect.

## 8. Format support and limits

Native PyTorch CUDA inference only. There is **no implemented or tested** Transformers `AutoModel`, GGUF, ONNX, vLLM or llama.cpp path. `special_tokens_map.json` is descriptive native metadata, not a loader contract.

Local import closure was demonstrated once, in a fresh process with a temporary working directory and an empty `PYTHONPATH`, under an audit hook that denied network access and out-of-bundle repository access; no denied access occurred. That establishes closure **on the measured environment only**. It is not a clean-machine test, not a CPU test, not a cross-hardware test, and not a fresh dependency-installation test.

The recorded export parity is likewise **profile-specific**: within each of the two profiles the export reproduced its source exactly, and no claim is made that the two profiles agree with each other, nor that any Transformers, GGUF, ONNX, vLLM or llama.cpp path exists.

## 9. Before you use output

This is a research artifact, not a safety- or correctness-certified assistant. In the recorded Python diagnostics the model produced a valid function interface far more often than a correct whole answer. **Do not execute generated code without independent review.** Backend details and the bounded parity evidence live in a separate private evidence archive and are not part of this bundle.
