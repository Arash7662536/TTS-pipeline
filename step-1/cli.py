"""
Corpus pass: read a dataset, normalize every transcript, write a VoxCPM
manifest, emit a statistics report.

    # dry run on 5k rows first -- always
    python -m persian_tts_frontend.cli normalize \
        --input  /data/fidibo/raw.jsonl --adapter jsonl \
        --output /data/fidibo/manifest.jsonl \
        --report /data/fidibo/report.json \
        --limit 5000 --selftest

    # full pass
    python -m persian_tts_frontend.cli normalize \
        --input /data/fidibo/raw.jsonl --adapter jsonl \
        --output /data/fidibo/manifest.jsonl --dataset-id 1 \
        --tier core --speaker-field narrator

    # Common Voice (diversity tier)
    python -m persian_tts_frontend.cli normalize \
        --input /data/cv-fa/validated.tsv --adapter commonvoice \
        --audio-root /data/cv-fa/clips \
        --output /data/cv-fa/manifest.jsonl --dataset-id 3 --tier diversity

Output rows carry the VoxCPM required fields plus provenance:

    {"audio": ..., "text": ..., "duration": ...,
     "dataset_id": 1, "text_raw": ..., "norm_version": "fa-fe-...",
     "speaker": "narrator_042", "tier": "core", "diacritic_density": 0.14}
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

from .diacritics import report as diacritic_report
from .pipeline import Normalizer, NormalizerConfig

# ------------------------------------------------------------------ adapters


def adapter_jsonl(path, audio_root=None, text_field="text",
                  audio_field="audio", speaker_field=None,
                  duration_field="duration"):
    """Generic JSONL. Use for Fidibo and for thomclas once you know its
    field names."""
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                yield {"_error": f"bad_json@{ln}"}
                continue
            audio = d.get(audio_field, "")
            if audio_root and not os.path.isabs(audio):
                audio = os.path.join(audio_root, audio)
            yield {
                "audio": audio,
                "text": d.get(text_field, ""),
                "duration": float(d.get(duration_field) or 0.0) or None,
                "speaker": d.get(speaker_field) if speaker_field else None,
                "_extra": {k: v for k, v in d.items()
                           if k not in {text_field, audio_field, speaker_field,
                                        duration_field}},
            }


def adapter_csv(path, audio_root=None, text_field="text", audio_field="audio",
                speaker_field=None, duration_field="duration", delimiter=","):
    with open(path, encoding="utf-8", newline="") as f:
        for d in csv.DictReader(f, delimiter=delimiter):
            audio = d.get(audio_field, "")
            if audio_root and not os.path.isabs(audio):
                audio = os.path.join(audio_root, audio)
            dur = d.get(duration_field)
            yield {
                "audio": audio,
                "text": d.get(text_field, ""),
                "duration": float(dur) if dur else None,
                "speaker": d.get(speaker_field) if speaker_field else None,
                "_extra": {},
            }


def adapter_commonvoice(path, audio_root=None, **kw):
    """Mozilla Common Voice Persian TSV.

    This is the recommended diversity-tier source: thousands of distinct
    speakers, which is the one thing 2000h of audiobook cannot give you. Audio
    quality is much worse than Fidibo -- gate it hard on DNSMOS and cap its
    share of the mix. Clips are short (3-6s), so they also cost you nothing
    against `max_batch_tokens`.
    """
    with open(path, encoding="utf-8", newline="") as f:
        for d in csv.DictReader(f, delimiter="\t"):
            fn = d.get("path", "")
            audio = os.path.join(audio_root, fn) if audio_root else fn
            yield {
                "audio": audio,
                "text": d.get("sentence", ""),
                "duration": None,
                "speaker": d.get("client_id"),
                "_extra": {"up_votes": d.get("up_votes"),
                           "down_votes": d.get("down_votes"),
                           "accent": d.get("accents") or d.get("accent")},
            }


def adapter_librispeech_style(path, audio_root=None, **kw):
    """`{utt_id} {transcript}` per line, audio alongside. Covers a lot of
    open-source Persian ASR releases."""
    root = audio_root or os.path.dirname(path)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or " " not in line:
                continue
            utt, text = line.split(" ", 1)
            yield {"audio": os.path.join(root, utt + ".wav"), "text": text,
                   "duration": None, "speaker": None, "_extra": {}}


ADAPTERS = {
    "jsonl": adapter_jsonl,
    "csv": adapter_csv,
    "tsv": lambda *a, **k: adapter_csv(*a, delimiter="\t", **k),
    "commonvoice": adapter_commonvoice,
    "trans": adapter_librispeech_style,
}


# ------------------------------------------------------------------ main pass

def run_normalize(args):
    cfg = NormalizerConfig(
        keep_harakat=not args.strip_harakat,
        decimal_style=args.decimal_style,
        latin_strategy=args.latin_strategy,
        parens_to_commas=not args.keep_parens,
    )
    nz = Normalizer(config=cfg)
    print(f"frontend version: {nz.version}", file=sys.stderr)

    adapter = ADAPTERS[args.adapter]
    kwargs = dict(audio_root=args.audio_root)
    if args.adapter in ("jsonl", "csv", "tsv"):
        kwargs.update(text_field=args.text_field, audio_field=args.audio_field,
                      speaker_field=args.speaker_field,
                      duration_field=args.duration_field)
    rows = adapter(args.input, **kwargs)

    stats = Counter()
    unexpected_chars = Counter()
    problems = Counter()
    speakers = Counter()
    speaker_seconds = defaultdict(float)
    densities = []
    samples_for_selftest = []
    dropped_examples = []

    out = open(args.output, "w", encoding="utf-8") if args.output else None
    try:
        for i, r in enumerate(rows):
            if args.limit and i >= args.limit:
                break
            if r.get("_error"):
                stats[r["_error"]] += 1
                continue
            stats["read"] += 1

            raw = (r.get("text") or "").strip()
            if not raw:
                stats["drop_empty_source"] += 1
                continue

            res = nz.normalize(raw)

            if len(samples_for_selftest) < 2000:
                samples_for_selftest.append(raw)

            for c in res.unexpected:
                unexpected_chars[c] += 1
            for p in res.problems:
                problems[p.split("@")[0]] += 1

            if not res.text.strip():
                stats["drop_empty_after_norm"] += 1
                if len(dropped_examples) < 10:
                    dropped_examples.append({"reason": "empty_after_norm",
                                             "raw": raw[:120]})
                continue
            if args.min_words and res.n_words < args.min_words:
                stats["drop_too_short"] += 1
                continue
            if args.max_chars and len(res.text) > args.max_chars:
                stats["drop_too_long"] += 1
                if len(dropped_examples) < 10:
                    dropped_examples.append({"reason": "too_long",
                                             "chars": len(res.text)})
                continue
            if args.require_audio and not os.path.exists(r["audio"]):
                stats["drop_missing_audio"] += 1
                if len(dropped_examples) < 10:
                    dropped_examples.append({"reason": "missing_audio",
                                             "audio": r["audio"]})
                continue

            stats["kept"] += 1
            densities.append(res.diacritic_density)
            spk = r.get("speaker") or args.default_speaker
            if spk:
                speakers[spk] += 1
                if r.get("duration"):
                    speaker_seconds[spk] += r["duration"]

            if out:
                row = {"audio": r["audio"], "text": res.text}
                if r.get("duration"):
                    row["duration"] = round(float(r["duration"]), 3)
                row["dataset_id"] = args.dataset_id
                if not args.no_provenance:
                    row["text_raw"] = raw
                    row["norm_version"] = nz.version
                    row["tier"] = args.tier
                    row["diacritic_density"] = round(res.diacritic_density, 4)
                    if spk:
                        row["speaker"] = spk
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if out:
            out.close()

    # ------------------------------------------------------------ the report
    rep = {
        "frontend_version": nz.version,
        "config": {k: (sorted(v) if isinstance(v, frozenset) else v)
                   for k, v in cfg.__dict__.items()},
        "counts": dict(stats),
        "keep_rate": round(stats["kept"] / max(1, stats["read"]), 4),
        "unexpected_chars": {
            f"U+{ord(c):04X} {c!r}": n
            for c, n in unexpected_chars.most_common(40)
        },
        "diacritic_problems": dict(problems),
        "diacritic_density": {
            "mean": round(sum(densities) / max(1, len(densities)), 4),
            "zero_share": round(
                sum(1 for d in densities if d == 0) / max(1, len(densities)), 4),
        },
        "speakers": {
            "n_distinct": len(speakers),
            "top_20_by_clips": speakers.most_common(20),
            "top_20_by_hours": sorted(
                ((s, round(sec / 3600, 2)) for s, sec in speaker_seconds.items()),
                key=lambda x: -x[1])[:20],
            "gini_hint": _skew(list(speakers.values())),
        },
        "latin_work_queue": nz.latin.work_queue(60),
        "dropped_examples": dropped_examples,
    }

    if args.selftest:
        rep["selftest_failures"] = {
            k: (len(v), v[:5]) for k, v in
            nz.selftest(samples_for_selftest).items()
        }

    txt = json.dumps(rep, indent=2, ensure_ascii=False)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(txt)
    print(txt)

    if rep["unexpected_chars"]:
        print("\n!! unexpected characters survived normalization -- "
              "extend chars.FOLD_MAP or chars.allowed_charset before the full "
              "pass", file=sys.stderr)
    if len(speakers) and len(speakers) < 50:
        print(f"\n!! only {len(speakers)} distinct speakers -- zero-shot "
              "cloning will generalize poorly; grow the diversity tier",
              file=sys.stderr)
    return 0


def _skew(counts):
    """Crude concentration measure: share of rows held by the top 10% of
    speakers. Above ~0.5 means you need the per-narrator cap."""
    if not counts:
        return None
    counts = sorted(counts, reverse=True)
    k = max(1, len(counts) // 10)
    return round(sum(counts[:k]) / sum(counts), 3)


# ------------------------------------------------------------------ argparse

def main(argv=None):
    p = argparse.ArgumentParser(prog="persian_tts_frontend")
    sub = p.add_subparsers(dest="cmd", required=True)

    n = sub.add_parser("normalize", help="normalize a corpus into a manifest")
    n.add_argument("--input", required=True)
    n.add_argument("--adapter", default="jsonl", choices=sorted(ADAPTERS))
    n.add_argument("--output", default=None)
    n.add_argument("--report", default=None)
    n.add_argument("--audio-root", default=None)
    n.add_argument("--text-field", default="text")
    n.add_argument("--audio-field", default="audio")
    n.add_argument("--speaker-field", default=None)
    n.add_argument("--duration-field", default="duration")
    n.add_argument("--default-speaker", default=None)
    n.add_argument("--dataset-id", type=int, default=0)
    n.add_argument("--tier", default="core",
                   choices=["gold", "core", "diversity", "register",
                            "emotion", "replay"])
    n.add_argument("--limit", type=int, default=0)
    n.add_argument("--min-words", type=int, default=2)
    n.add_argument("--max-chars", type=int, default=600)
    n.add_argument("--require-audio", action="store_true")
    n.add_argument("--strip-harakat", action="store_true")
    n.add_argument("--keep-parens", action="store_true")
    n.add_argument("--decimal-style", default="momayez",
                   choices=["momayez", "fractional"])
    n.add_argument("--latin-strategy", default="escalate",
                   choices=["escalate", "lexicon", "transliterate"])
    n.add_argument("--no-provenance", action="store_true")
    n.add_argument("--selftest", action="store_true")
    n.set_defaults(func=run_normalize)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
