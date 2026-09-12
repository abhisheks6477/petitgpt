"""Local ARC-Easy/PIQA likelihood evaluation with the pinned raw-completion protocol."""

from __future__ import annotations

import ast
from pathlib import Path

from adapter_common import HERE, TOKENIZER_SHA, canonical, read, rows, sha, write
from reader_contracts import checkpoint, config, module, source, tokenizer

BENCH = HERE / "sources/benchmark/runs/petitgpt_frozen_reference_public_benchmark_fp32_v2"


def arguments(p, stage):
    p.add_argument(
        "--model", type=Path, required=True, help="New checkpoint or native bundle directory"
    )
    p.add_argument("--tasks", type=Path, required=True, help="Local task inventory JSON")


class Encoding:
    backend = "causal"

    def __init__(self, tok):
        self.tokenizer = tok
        path = BENCH / "source/harness/lm_eval/api/model.py"
        cls = next(
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "TemplateLM"
        )
        method = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_encode_pair"
        )
        ns = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
        self.pair = ns["_encode_pair"].__get__(self)

    def tok_encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise ValueError("Raw likelihood forbids added BOS/EOS")
        return self.tokenizer.encode(text, add_special_tokens=False).ids


def request(task, row, index):
    if task == "arc_easy":
        labels = row["choices"]["label"]
        choices = row["choices"]["text"]
        if len(labels) != len(choices) or len(set(labels)) != len(labels):
            raise ValueError("ARC labels mismatch")
        gold = labels.index(str(row["answerKey"]))
        question = row["question"]
    elif task == "piqa":
        choices = [row["sol1"], row["sol2"]]
        gold = row["label"]
        question = row["goal"]
    else:
        raise ValueError("Only arc_easy and piqa are supported")
    if (
        not isinstance(question, str)
        or not question
        or len(choices) < 2
        or any(not isinstance(c, str) or not c for c in choices)
    ):
        raise ValueError("Nonempty raw question/choice strings required")
    if type(gold) is not int or not 0 <= gold < len(choices):
        raise ValueError("Invalid gold choice")
    return {
        "row_index": index,
        "id": str(row.get("id", f"{task}:{index}")),
        "prompt": f"Question: {question}\nAnswer:",
        "choices": choices,
        "gold": gold,
    }


def inputs(path, tok):
    manifest = read(path)
    if manifest.get("schema") != "petitgpt-tasks-new-v3" or manifest.get("scope") not in (
        "synthetic",
        "subset",
        "full-local",
    ):
        raise ValueError(
            "Task manifest needs schema and explicit synthetic/subset/full-local scope"
        )
    enc = Encoding(tok)
    all_rows = []
    identities = []
    seen = set()
    for task in manifest["tasks"]:
        name = task["task"]
        if (
            name in seen
            or not isinstance(task["revision"], str)
            or not task["revision"]
            or not task["split"]
        ):
            raise ValueError("Unique task names and explicit revision/split required")
        seen.add(name)
        p = Path(task["path"])
        if not p.is_absolute():
            p = path.parent / p
        if sha(p) != task["sha256"]:
            raise ValueError("Task file hash mismatch")
        raw = rows(p)
        if type(task["rows"]) is not int or task["rows"] != len(raw) or not raw:
            raise ValueError("Task row count mismatch")
        normalized = [request(name, r, i) for i, r in enumerate(raw)]
        if len({r["id"] for r in normalized}) != len(normalized):
            raise ValueError("Duplicate task row IDs")
        for r in normalized:
            pairs = []
            for c in r["choices"]:
                ctx, cont = enc.pair(r["prompt"], " " + c)
                if (
                    not ctx
                    or not cont
                    or len(ctx) + len(cont) > 2048
                    or ctx + cont != enc.tok_encode(r["prompt"] + " " + c)
                ):
                    raise ValueError(
                        "Context/continuation token boundary or context length mismatch"
                    )
                pairs.append((ctx, cont))
            all_rows.append({"task": name, **r, "encodings": pairs})
        identities.append({
            **task,
            "path": str(p.resolve()),
            "normalized_rows_sha256": __import__("hashlib")
            .sha256(canonical(normalized).encode())
            .hexdigest(),
        })
    if not all_rows:
        raise ValueError("No tasks")
    return manifest["scope"], identities, all_rows


def run(a):
    tok = tokenizer(a.tokenizer)
    scope, identities, requests = inputs(a.tasks, tok)
    if a.model.is_dir():
        cfg = config(read(a.model / "config.json"))
        provenance = read(a.model / "MODEL_PROVENANCE.json")
        if provenance["new_export_sha256"] != sha(a.model / "model.safetensors"):
            raise ValueError("Native weight hash mismatch")
        if sha(a.model / "tokenizer.json") != TOKENIZER_SHA:
            raise ValueError("Native tokenizer mismatch")
        model_id = {"sha256": sha(a.model / "model.safetensors"), "config": cfg, "format": "native"}
    else:
        model_id = checkpoint(a.model)
    evidence = {
        "schema": "petitgpt-likelihood-new-v3",
        "policy": "new-run",
        "scope": scope,
        "tasks": identities,
        "model": model_id,
        "tokenizer_sha256": TOKENIZER_SHA,
        "request_count": len(requests),
        "execution": "NOT_RUN",
        "published_results_reproduced": False,
        "profile": "FP32 parameters/forward/logsoftmax/sum, MATH SDPA, batch1, no padding/chat/BOS/EOS",
        "acc_norm": "Unicode length of original answer excluding leading delimiter; first maximum",
    }
    if a.execute:
        execute(a, requests, evidence)
    return evidence


def execute(a, requests, evidence):
    source()
    from reader_contracts import CONFIG, load, state
    from safetensors.torch import load_model
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from src.model import GPT, gpt_config_from_checkpoint_dict

    helpers = module(BENCH / "runtime/likelihood_checks.py", "reader_likelihood_helpers")
    if not torch.cuda.is_available():
        raise RuntimeError("Recorded likelihood profile requires CUDA")
    model = GPT(gpt_config_from_checkpoint_dict(CONFIG)).eval().requires_grad_(False)
    if a.model.is_dir():
        load_model(model, str(a.model / "model.safetensors"), strict=True, device="cpu")
    else:
        model.load_state_dict(load(a.model)["model"], strict=True)
    state(model.state_dict())
    model.to("cuda")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    a.out_dir.mkdir(parents=True)
    write(a.out_dir / "INPUTS.json", evidence)
    results = []
    with (a.out_dir / "ROWS.jsonl").open("x") as f:
        for r in requests:
            scores = []
            for ctx, cont in r["encodings"]:
                x = torch.tensor([(ctx + cont)[:-1]], device="cuda")
                with (
                    torch.inference_mode(),
                    torch.autocast("cuda", enabled=False),
                    sdpa_kernel([SDPBackend.MATH]),
                ):
                    logits = model(x)
                    score = float(helpers.continuation_logprobs(logits, len(ctx), cont).sum())
                scores.append(score)
            result = {k: v for k, v in r.items() if k != "encodings"}
            result.update(scores=scores, **helpers.metrics(scores, r["choices"], r["gold"]))
            f.write(canonical(result) + "\n")
            f.flush()
            results.append(result)
    summary = {}
    for task in sorted({r["task"] for r in results}):
        subset = [r for r in results if r["task"] == task]
        summary[task] = {
            "rows": len(subset),
            **{key: sum(r[key] for r in subset) / len(subset) for key in ("acc", "acc_norm")},
        }
    write(
        a.out_dir / "RESULTS.json",
        {
            **evidence,
            "execution": "COMPLETED",
            "results": summary,
            "rows_sha256": sha(a.out_dir / "ROWS.jsonl"),
        },
    )
