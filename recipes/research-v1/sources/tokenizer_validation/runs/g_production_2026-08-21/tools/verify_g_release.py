#!/usr/bin/env python3
"""Independent closeout verification for the canonical PetitGPT tokenizer release.

Deliberately does not import tokenizer/tokenizer_training/train_tokenizer.py. It re-derives
every claim from the published bytes and from the frozen upstream corpus release, and it
verifies the already-produced full-stream validation evidence rather than duplicating the
10 GB round-trip.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

EXPECTED_SPECIAL_IDS = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[BOS]": 2,
    "[EOS]": 3,
    "<|system|>": 4,
    "<|user|>": 5,
    "<|assistant|>": 6,
}
EXPECTED_VOCAB_SIZE = 32_000
CORPUS_SCHEMA = "petitgpt-f-tokenizer-corpus-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--corpus_manifest", type=Path, required=True)
    ap.add_argument("--corpus_manifest_sha256", type=str, required=True)
    ap.add_argument("--out_json", type=Path, required=True)
    args = ap.parse_args()

    import tokenizers
    from tokenizers import Tokenizer

    release = args.release.resolve()
    checks: list[dict[str, object]] = []

    def check(name: str, ok: bool, detail: object = None) -> None:
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    # ---------------------------------------------------------------- artifact integrity
    required = [
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "manifest.json",
        "tokenizer_release_manifest.json",
        "environment.json",
        "validation.json",
        "SHA256SUMS",
    ]
    check("required_files_present", all((release / n).is_file() for n in required), required)

    listed: dict[str, str] = {}
    for line in (release / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        listed[name] = digest
    mismatched = [n for n, d in listed.items() if sha256_file(release / n) != d]
    check(
        "sha256sums_match_published_bytes",
        not mismatched,
        {"files": len(listed), "bad": mismatched},
    )

    on_disk = {
        str(p.relative_to(release))
        for p in release.rglob("*")
        if p.is_file()
        and p.name not in {"manifest.json", "tokenizer_release_manifest.json", "SHA256SUMS"}
    }
    check(
        "sha256sums_covers_every_payload_file",
        on_disk == set(listed),
        sorted(on_disk ^ set(listed)),
    )

    manifest_bytes = (release / "manifest.json").read_bytes()
    check(
        "manifest_copies_byte_identical",
        manifest_bytes == (release / "tokenizer_release_manifest.json").read_bytes(),
    )
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    tokenizer_sha = sha256_file(release / "tokenizer.json")
    check("manifest_binds_tokenizer_sha256", manifest.get("tokenizer_sha256") == tokenizer_sha)

    modes = {str(p.relative_to(release)): oct(p.stat().st_mode & 0o777) for p in release.rglob("*")}
    bad_modes = {n: m for n, m in modes.items() if m not in ("0o444", "0o555")}
    check(
        "release_is_read_only",
        not bad_modes and oct(release.stat().st_mode & 0o777) == "0o555",
        bad_modes,
    )

    # ---------------------------------------------------------------- tokenizer contract
    tok = Tokenizer.from_file(str(release / "tokenizer.json"))
    obj = json.loads((release / "tokenizer.json").read_text(encoding="utf-8"))
    check("tokenizer_reloads", True)
    check("reload_reserialises_identically", json.loads(tok.to_str()) == obj)
    check(
        "vocab_size_exactly_32000",
        int(tok.get_vocab_size(with_added_tokens=True)) == EXPECTED_VOCAB_SIZE,
        int(tok.get_vocab_size(with_added_tokens=True)),
    )
    actual_specials = {name: tok.token_to_id(name) for name in EXPECTED_SPECIAL_IDS}
    check("special_ids_exact", actual_specials == EXPECTED_SPECIAL_IDS, actual_specials)

    added_special = [e for e in obj.get("added_tokens") or [] if e.get("special") is True]
    check("exactly_seven_registered_specials", len(added_special) == 7, len(added_special))

    ids = set((obj.get("model") or {}).get("vocab", {}).values())
    ids.update(e["id"] for e in obj.get("added_tokens") or [])
    check("runtime_ids_are_0_to_31999", ids == set(range(EXPECTED_VOCAB_SIZE)), len(ids))

    check("normalizer_is_null", obj.get("normalizer") is None, obj.get("normalizer"))
    check("post_processor_is_null", obj.get("post_processor") is None, obj.get("post_processor"))
    pre = obj.get("pre_tokenizer") or {}
    check(
        "pre_tokenizer_bytelevel_no_prefix_space",
        pre.get("type") == "ByteLevel" and pre.get("add_prefix_space") is False,
        pre,
    )
    check("decoder_is_bytelevel", (obj.get("decoder") or {}).get("type") == "ByteLevel")
    check("bpe_unk_token", (obj.get("model") or {}).get("unk_token") == "[UNK]")

    # No automatic BOS/EOS: encoding plain text must not introduce any special id.
    tok.encode_special_tokens = True
    probe = "The quick brown fox.\nliteral [BOS] [EOS] <|assistant|> stay text."
    probe_ids = tok.encode(probe).ids
    leaked = [i for i in probe_ids if i in set(EXPECTED_SPECIAL_IDS.values())]
    check("no_auto_bos_eos_or_injection", not leaked, leaked)
    check("probe_roundtrip", tok.decode(probe_ids) == probe)

    # vocab.json / merges.txt must describe the same model as tokenizer.json.
    vocab_json = json.loads((release / "vocab.json").read_text(encoding="utf-8"))
    check("vocab_json_matches_model", vocab_json == (obj.get("model") or {}).get("vocab"))
    merges_lines = (release / "merges.txt").read_text(encoding="utf-8").splitlines()
    if merges_lines and merges_lines[0].startswith("#"):
        merges_lines = merges_lines[1:]
    model_merges = [
        m if isinstance(m, str) else " ".join(m) for m in (obj.get("model") or {}).get("merges", [])
    ]
    check("merges_txt_matches_model", merges_lines == model_merges, len(merges_lines))

    # ---------------------------------------------------------------- provenance binding
    corpus_sha = sha256_file(args.corpus_manifest)
    check(
        "corpus_manifest_sha_matches_expected",
        corpus_sha == args.corpus_manifest_sha256,
        corpus_sha,
    )
    corpus = json.loads(args.corpus_manifest.read_text(encoding="utf-8"))
    check("corpus_manifest_schema", corpus.get("schema_version") == CORPUS_SCHEMA)

    binding = manifest.get("corpus_binding") or {}
    check("binding_verified_before_training", binding.get("verified_before_training") is True)
    check("binding_manifest_sha", binding.get("manifest_sha256") == corpus_sha)
    check(
        "binding_run_fingerprint",
        binding.get("run_fingerprint_sha256") == corpus.get("run_fingerprint_sha256"),
        binding.get("run_fingerprint_sha256"),
    )
    check(
        "binding_selected_set_fingerprint",
        binding.get("selected_occurrence_set_sha256")
        == (corpus.get("selection") or {}).get("selected_occurrence_set_sha256"),
        binding.get("selected_occurrence_set_sha256"),
    )

    corpus_outputs = {
        b["canonical_name"]: b["output"] for b in (corpus.get("selection") or {}).get("buckets", [])
    }
    bound_files = binding.get("files") or []
    check(
        "binding_covers_every_corpus_bucket",
        {f["canonical_bucket"] for f in bound_files} == set(corpus_outputs),
        sorted({f["canonical_bucket"] for f in bound_files}),
    )
    bad_files = []
    for entry in bound_files:
        expected = corpus_outputs.get(entry["canonical_bucket"], {})
        actual = sha256_file(Path(entry["path"]))
        if (
            actual != expected.get("sha256")
            or entry["sha256"] != expected.get("sha256")
            or entry["size_bytes"] != expected.get("size_bytes")
            or entry["expected_occurrences"] != expected.get("rows")
        ):
            bad_files.append(entry["canonical_bucket"])
    check("bound_files_rehash_to_frozen_corpus", not bad_files, bad_files)

    per_file = (manifest.get("training") or {}).get("per_file") or []
    drift = [
        e["canonical_bucket"]
        for e in per_file
        if e.get("yielded_samples") != e.get("expected_occurrences")
    ]
    check("consumed_occurrences_match_manifest", not drift, drift)
    check(
        "consumed_totals_match_corpus",
        (manifest.get("training") or {}).get("consumed_occurrences")
        == (corpus.get("selection") or {}).get("total_selected_documents")
        and (manifest.get("training") or {}).get("consumed_utf8_bytes")
        == (corpus.get("selection") or {}).get("total_realized_cleaned_utf8_bytes"),
        {
            "occurrences": (manifest.get("training") or {}).get("consumed_occurrences"),
            "bytes": (manifest.get("training") or {}).get("consumed_utf8_bytes"),
        },
    )

    # ---------------------------------------------------------------- validation evidence
    validation = json.loads((release / "validation.json").read_text(encoding="utf-8"))
    check("validation_binds_tokenizer_sha", validation.get("tokenizer_sha256") == tokenizer_sha)
    check("validation_status_pass", validation.get("status") == "PASS")
    stream = validation.get("full_corpus_stream") or {}
    totals = stream.get("totals") or {}
    check("full_stream_present", bool(stream), stream.get("status"))
    check("full_stream_status_pass", stream.get("status") == "PASS")
    check(
        "full_stream_occurrences_match_corpus",
        totals.get("occurrences")
        == (corpus.get("selection") or {}).get("total_selected_documents"),
        totals.get("occurrences"),
    )
    check(
        "full_stream_bytes_match_corpus",
        totals.get("utf8_bytes")
        == (corpus.get("selection") or {}).get("total_realized_cleaned_utf8_bytes"),
        totals.get("utf8_bytes"),
    )
    check("full_stream_zero_roundtrip_failures", totals.get("roundtrip_failures") == 0)
    check("full_stream_zero_unk", totals.get("unk_occurrences") == 0)
    check("full_stream_zero_special_ids", totals.get("special_id_occurrences") == 0)
    check("fixtures_zero_failures", (validation.get("fixtures") or {}).get("failures") == 0)
    check(
        "fixtures_injection_hardening",
        (validation.get("fixtures") or {}).get("injection_hardening_ok") is True,
    )

    # ---------------------------------------------------------------- environment binding
    environment = json.loads((release / "environment.json").read_text(encoding="utf-8"))
    check("environment_records_python", bool(environment.get("python_version")))
    check("environment_records_tokenizers", bool(environment.get("tokenizers_version")))
    check(
        "environment_tokenizers_matches_runtime",
        environment.get("tokenizers_version") == getattr(tokenizers, "__version__", None),
        environment.get("tokenizers_version"),
    )
    check("environment_records_git_head", bool(environment.get("git_head")))
    check(
        "environment_trainer_sha_matches_manifest",
        environment.get("trainer_sha256")
        == (manifest.get("environment") or {}).get("trainer_sha256"),
    )
    check(
        "environment_input_order_matches_binding",
        environment.get("input_file_order") == [f["supplied_as"] for f in bound_files],
        environment.get("input_file_order"),
    )
    check(
        "manifest_marks_release_canonical",
        (manifest.get("contract") or {}).get("canonical") is True,
    )
    check("manifest_reports_no_contract_issues", not (manifest.get("contract") or {}).get("issues"))

    failures = [c for c in checks if c["status"] != "PASS"]
    report = {
        "schema_version": "petitgpt-g-independent-verification-v1",
        "release": str(release),
        "tokenizer_json_sha256": tokenizer_sha,
        "manifest_sha256": sha256_file(release / "manifest.json"),
        "sha256sums_sha256": sha256_file(release / "SHA256SUMS"),
        "corpus_manifest_sha256": corpus_sha,
        "verifier_python": platform.python_version(),
        "verifier_tokenizers": getattr(tokenizers, "__version__", None),
        "checks": checks,
        "checks_total": len(checks),
        "checks_failed": len(failures),
        "status": "PASS" if not failures else "FAIL",
        "note": (
            "The 10 GB streaming round-trip is not duplicated; its published evidence is verified "
            "for internal consistency and bound to this tokenizer and corpus."
        ),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out_json.with_suffix(args.out_json.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(args.out_json)

    for item in checks:
        marker = "ok " if item["status"] == "PASS" else "FAIL"
        print(f"[{marker}] {item['check']}")
    print(f"\nstatus={report['status']}  {len(checks) - len(failures)}/{len(checks)} checks passed")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
