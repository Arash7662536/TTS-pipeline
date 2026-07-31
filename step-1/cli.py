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

    # Arrow / HuggingFace `datasets` dataset (audio + text + whatever else).
    # Look at the columns first, then point the field flags at them:
    python -m persian_tts_frontend.cli inspect --input /data/fidibo/arrow
    python -m persian_tts_frontend.cli normalize \
        --input /data/fidibo/arrow --adapter arrow \
        --text-field text --audio-field audio --speaker-field narrator \
        --output /tmp/probe.jsonl --report /tmp/probe.json \
        --limit 5000 --selftest

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
import glob
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


# ------------------------------------------------------------------- arrow

# HuggingFace `Audio` columns are struct<bytes, path>; other exporters name the
# path member differently. Checked in order.
_AUDIO_PATH_KEYS = ("path", "filename", "file", "audio_path", "audio_filepath")


def _stringy(pa, t):
    return (pa.types.is_string(t) or pa.types.is_large_string(t)
            or str(t) == "string_view")


def _pyarrow():
    try:
        import pyarrow as pa
    except ImportError:
        raise SystemExit("the arrow adapter needs pyarrow -- "
                         "pip install 'pyarrow>=12'")
    return pa


def arrow_fragments(path, split=None):
    """Resolve `--input` into an ordered list of Arrow/Parquet files.

    Accepts, in order of how you are likely to have the data:

    * a `save_to_disk()` directory -- `data-00000-of-000NN.arrow` + `state.json`
    * a `DatasetDict` directory -- pass `--split train` to pick one
    * a `load_dataset()` cache directory under ~/.cache/huggingface/datasets
    * a single `.arrow` or `.parquet` file
    * a glob, e.g. `/data/shards/*.arrow`

    Both Arrow IPC encodings are handled (file/Feather-v2 and the stream format
    that `datasets` actually writes), so no `datasets` install is required.
    """
    if any(ch in path for ch in "*?["):
        files = sorted(glob.glob(path))
    elif os.path.isdir(path):
        root = path
        if split and os.path.isdir(os.path.join(path, split)):
            root = os.path.join(path, split)
        arrow, parquet = [], []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            for fn in sorted(filenames):
                if fn.startswith("."):
                    continue
                if fn.endswith(".arrow"):
                    arrow.append(os.path.join(dirpath, fn))
                elif fn.endswith(".parquet"):
                    parquet.append(os.path.join(dirpath, fn))
        files = arrow or parquet
        # A DatasetDict whose splits are not directories (dataset-train.arrow).
        if split and root == path:
            files = [f for f in files if split in os.path.relpath(f, path)]
    else:
        files = [path] if os.path.isfile(path) else []
    if not files:
        raise SystemExit(f"no .arrow/.parquet files found under {path!r}"
                         + (f" for split {split!r}" if split else ""))
    return files


def _arrow_batches(pa, fp):
    """Stream one file as RecordBatches. Memory-mapped: the audio bytes are
    never faulted in unless something actually reads that column."""
    if fp.endswith(".parquet"):
        import pyarrow.parquet as pq
        for b in pq.ParquetFile(fp).iter_batches(batch_size=1024):
            yield b
        return
    with pa.memory_map(fp, "rb") as src:
        try:
            reader = pa.ipc.open_file(src)
        except pa.ArrowInvalid:
            src.seek(0)
            for b in pa.ipc.open_stream(src):
                yield b
            return
        for i in range(reader.num_record_batches):
            yield reader.get_batch(i)


def _arrow_schema(pa, fp):
    if fp.endswith(".parquet"):
        import pyarrow.parquet as pq
        return pq.ParquetFile(fp).schema_arrow
    with pa.memory_map(fp, "rb") as src:
        try:
            return pa.ipc.open_file(src).schema
        except pa.ArrowInvalid:
            src.seek(0)
            return pa.ipc.open_stream(src).schema


def _arrow_rows(pa, fp):
    if fp.endswith(".parquet"):
        import pyarrow.parquet as pq
        return pq.ParquetFile(fp).metadata.num_rows
    return sum(b.num_rows for b in _arrow_batches(pa, fp))


def _field_names(pa, schema, prefix=""):
    """Flat, dotted list of columns -- struct members included, so the error
    message can suggest `--audio-field audio.path`."""
    names = []
    for f in schema:
        names.append(prefix + f.name)
        if pa.types.is_struct(f.type):
            names.extend(_field_names(pa, f.type, prefix + f.name + "."))
    return names


def _column(pa, batch, dotted):
    """Column by (possibly dotted) name, or None if it is not there."""
    if not dotted:
        return None
    head, _, rest = dotted.partition(".")
    if head not in batch.schema.names:
        return None
    arr = batch.column(batch.schema.names.index(head))
    for part in filter(None, rest.split(".")):
        if not pa.types.is_struct(arr.type):
            return None
        try:
            arr = arr.field(part)
        except (KeyError, IndexError):
            return None
    return arr


def _audio_paths(pa, arr, n):
    """Pull filesystem paths out of an audio column without touching the audio
    bytes. Returns [None]*n when the column carries only embedded bytes."""
    if arr is None:
        return [None] * n
    if pa.types.is_struct(arr.type):
        members = [f.name for f in arr.type]
        for key in _AUDIO_PATH_KEYS:
            if key in members:
                return arr.field(key).to_pylist()
        return [None] * n
    if pa.types.is_binary(arr.type) or pa.types.is_large_binary(arr.type):
        return [None] * n
    return arr.to_pylist()


def _as_float(v):
    try:
        return float(v) or None
    except (TypeError, ValueError):
        return None


def adapter_arrow(path, audio_root=None, text_field="text",
                  audio_field="audio", speaker_field=None,
                  duration_field="duration", split=None):
    """Arrow / HuggingFace `datasets` dataset.

    Only the columns named by the field flags are converted to Python, so a
    dataset with megabytes of audio per row streams at the speed of its text
    column. Rows whose audio is embedded (struct<bytes, path> with no path) get
    a `<file>#row=N` locator instead of a path -- traceable, but do not pass
    `--require-audio` for those.
    """
    pa = _pyarrow()
    files = arrow_fragments(path, split=split)
    # No upfront row count -- it would walk every shard's metadata before the
    # first row, which defeats `--limit` on a big corpus. `inspect` counts.
    print(f"arrow: {len(files)} file(s), first={files[0]}", file=sys.stderr)

    warned = set()

    def warn(key, msg):
        if key not in warned:
            warned.add(key)
            print(f"!! {msg}", file=sys.stderr)

    row_no = 0
    for fp in files:
        for batch in _arrow_batches(pa, fp):
            n = batch.num_rows
            if not n:
                continue

            texts = _column(pa, batch, text_field)
            if texts is None:
                raise SystemExit(
                    f"--text-field {text_field!r} is not a column of this "
                    f"dataset.\navailable: "
                    + ", ".join(_field_names(pa, batch.schema))
                    + "\n(run `cli inspect --input ...` to see the rows)")
            texts = texts.to_pylist()

            audio_col = _column(pa, batch, audio_field)
            if audio_col is None and audio_field:
                warn("audio", f"no {audio_field!r} column -- emitting "
                              "'<file>#row=N' locators")
            auds = _audio_paths(pa, audio_col, n)
            if audio_col is not None and not any(auds):
                warn("audio_bytes",
                     f"{audio_field!r} carries embedded audio with no path -- "
                     "emitting '<file>#row=N' locators; --require-audio will "
                     "drop every row")

            if speaker_field:
                spk_col = _column(pa, batch, speaker_field)
                if spk_col is None:
                    warn("speaker", f"no {speaker_field!r} column -- speaker "
                                    "stats will be empty")
                spks = spk_col.to_pylist() if spk_col is not None else [None] * n
            else:
                spks = [None] * n

            dur_col = _column(pa, batch, duration_field)
            if dur_col is None and duration_field:
                warn("duration", f"no {duration_field!r} column -- "
                                 "speaker-hours will be empty")
            durs = dur_col.to_pylist() if dur_col is not None else [None] * n

            for k in range(n):
                audio = auds[k] or ""
                if audio and audio_root and not os.path.isabs(audio):
                    audio = os.path.join(audio_root, audio)
                if not audio:
                    audio = f"{fp}#row={row_no + k}"
                text = texts[k]
                spk = spks[k]
                yield {
                    "audio": audio,
                    "text": text if isinstance(text, str) else
                            ("" if text is None else str(text)),
                    "duration": _as_float(durs[k]),
                    "speaker": None if spk is None else str(spk),
                    "_extra": {},
                }
            row_no += n


ADAPTERS = {
    "jsonl": adapter_jsonl,
    "csv": adapter_csv,
    "tsv": lambda *a, **k: adapter_csv(*a, delimiter="\t", **k),
    "commonvoice": adapter_commonvoice,
    "trans": adapter_librispeech_style,
    "arrow": adapter_arrow,
    "parquet": adapter_arrow,      # same reader; resolved by file suffix
}

FIELD_AWARE = {"jsonl", "csv", "tsv", "arrow", "parquet"}


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
    if args.adapter in FIELD_AWARE:
        kwargs.update(text_field=args.text_field, audio_field=args.audio_field,
                      speaker_field=args.speaker_field,
                      duration_field=args.duration_field)
    if args.adapter in ("arrow", "parquet"):
        kwargs.update(split=args.split)
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


def run_inspect(args):
    """Print the schema of an Arrow/Parquet dataset plus a few rows, so you know
    what to give `--text-field` / `--audio-field` / `--speaker-field`."""
    pa = _pyarrow()
    files = arrow_fragments(args.input, split=args.split)

    print(f"{len(files)} file(s):")
    for f in files[:8]:
        print("  " + f)
    if len(files) > 8:
        print(f"  ... +{len(files) - 8} more")

    schema = _arrow_schema(pa, files[0])
    print("\ncolumns:")
    for f in schema:
        print(f"  {f.name:<28} {f.type}")
        if pa.types.is_struct(f.type):
            for sub in f.type:
                hint = ("   (usable as --audio-field "
                        f"{f.name}.{sub.name})") if _stringy(pa, sub.type) else ""
                print(f"    .{sub.name:<25} {sub.type}{hint}")

    print(f"\nrows: {sum(_arrow_rows(pa, f) for f in files)}")

    if args.rows:
        print(f"\nfirst {args.rows} row(s):")
        shown = 0
        for batch in _arrow_batches(pa, files[0]):
            for i in range(batch.num_rows):
                if shown >= args.rows:
                    break
                row = {name: _cell(pa, batch.column(j), i)
                       for j, name in enumerate(batch.schema.names)}
                print("  " + json.dumps(row, ensure_ascii=False))
                shown += 1
            if shown >= args.rows:
                break
    return 0


def _cell(pa, arr, i):
    """One value, rendered for human eyes -- blobs and waveforms summarised
    rather than dumped."""
    t = arr.type
    if pa.types.is_struct(t):
        return {f.name: _cell(pa, arr.field(f.name), i) for f in t}
    if pa.types.is_binary(t) or pa.types.is_large_binary(t):
        v = arr[i].as_py()
        return None if v is None else f"<binary {len(v)} bytes>"
    if (pa.types.is_list(t) or pa.types.is_large_list(t)
            or pa.types.is_fixed_size_list(t)):
        s = arr[i]
        return None if not s.is_valid else f"<list len={len(s)}>"
    v = arr[i].as_py()
    if isinstance(v, str) and len(v) > 160:
        return v[:160] + f"... (+{len(v) - 160} chars)"
    return v if isinstance(v, (str, int, float, bool, type(None))) else str(v)


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
    n.add_argument("--input", required=True,
                   help="file, glob, or (arrow/parquet) dataset directory")
    n.add_argument("--adapter", default="jsonl", choices=sorted(ADAPTERS))
    n.add_argument("--split", default=None,
                   help="arrow/parquet only: pick one split of a DatasetDict")
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

    q = sub.add_parser("inspect",
                       help="show the schema + first rows of an arrow/parquet "
                            "dataset")
    q.add_argument("--input", required=True)
    q.add_argument("--split", default=None)
    q.add_argument("--rows", type=int, default=3)
    q.set_defaults(func=run_inspect)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
