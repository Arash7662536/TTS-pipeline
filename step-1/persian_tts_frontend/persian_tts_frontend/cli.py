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
import re
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


def _wav_duration(head):
    """Seconds from a WAV header prefix, or None if it is not parseable PCM.

    Reads the `fmt ` and `data` chunk descriptors rather than assuming the
    canonical 44-byte layout -- encoders interleave `LIST`/`fact` chunks, and
    guessing the offset would silently mis-time those files. The `data` chunk
    carries its own byte count, so the payload never has to be touched.
    """
    if not head or len(head) < 16 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        return None
    import struct
    pos, rate, channels, bits = 12, None, None, None
    while pos + 8 <= len(head):
        cid = head[pos:pos + 4]
        size = int.from_bytes(head[pos + 4:pos + 8], "little")
        body = pos + 8
        if cid == b"fmt " and body + 16 <= len(head):
            _, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", head, body)
        elif cid == b"data":
            if not (rate and channels and bits):
                return None
            per_sec = rate * channels * (bits // 8)
            return (size / per_sec) if per_sec else None
        pos = body + size + (size & 1)
    return None


def _audio_durations(pa, arr, n):
    """Durations for a struct<bytes, ...> audio column, read from the WAV
    headers only. `binary_slice` keeps this to the first bytes of each blob, so
    a 45 GB corpus is scanned without decoding any audio."""
    if arr is None or not pa.types.is_struct(arr.type):
        return [None] * n
    members = [f.name for f in arr.type]
    if "bytes" not in members:
        return [None] * n
    import pyarrow.compute as pc
    heads = pc.binary_slice(arr.field("bytes"), 0, 128).to_pylist()
    return [_wav_duration(h) for h in heads]


def adapter_arrow(path, audio_root=None, text_field="text",
                  audio_field="audio", speaker_field=None,
                  duration_field="duration", split=None, extra_fields=(),
                  duration_from_audio=False):
    """Arrow / HuggingFace `datasets` dataset.

    Only the columns named by the field flags are converted to Python, so a
    dataset with megabytes of audio per row streams at the speed of its text
    column. Rows whose audio is embedded (struct<bytes, path> with no path) get
    a `<file>#row=N` locator instead of a path -- traceable, but do not pass
    `--require-audio` for those.

    `extra_fields` names further columns to carry in `_extra` -- per-clip
    quality scores (DNSMOS and friends) that the caller wants to gate on or
    record in the manifest. They are fetched per batch like everything else, so
    naming them costs a column read, not a decode of the audio.
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

    for fp in files:
        # Per-file, not cumulative: the locator pairs a filename with an index,
        # so the index has to be the one you would pass to that file's reader.
        row_no = 0
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
            if dur_col is not None:
                durs = dur_col.to_pylist()
            elif duration_from_audio:
                durs = _audio_durations(pa, audio_col, n)
                if not any(d for d in durs):
                    warn("duration_audio",
                         "--duration-from-audio found no readable WAV header "
                         f"in {audio_field!r} -- durations stay empty")
            else:
                if duration_field:
                    warn("duration", f"no {duration_field!r} column -- "
                                     "speaker-hours will be empty; pass "
                                     "--duration-from-audio to read it from "
                                     "the audio headers")
                durs = [None] * n

            extras = {}
            for name in extra_fields:
                col = _column(pa, batch, name)
                if col is None:
                    warn("extra:" + name, f"no {name!r} column -- "
                                          "ignored by --keep-fields/--min-field")
                    continue
                extras[name] = col.to_pylist()

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
                    "_extra": {name: v[k] for name, v in extras.items()},
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

    bounds = _parse_bounds(args.min_field, args.max_field)
    keep_fields = [f for f in (args.keep_fields or "").split(",") if f]
    # A gate on a column implies reading it, whether or not it is also kept.
    wanted = list(dict.fromkeys(keep_fields + list(bounds)))
    drop_re = re.compile(args.drop_text_matching) if args.drop_text_matching else None

    adapter = ADAPTERS[args.adapter]
    kwargs = dict(audio_root=args.audio_root)
    if args.adapter in FIELD_AWARE:
        kwargs.update(text_field=args.text_field, audio_field=args.audio_field,
                      speaker_field=args.speaker_field,
                      duration_field=args.duration_field)
    if args.adapter in ("arrow", "parquet"):
        kwargs.update(split=args.split, extra_fields=wanted,
                      duration_from_audio=args.duration_from_audio)
    rows = adapter(args.input, **kwargs)

    stats = Counter()
    unexpected_chars = Counter()
    problems = Counter()
    speakers = Counter()
    speaker_seconds = defaultdict(float)
    densities = []
    samples_for_selftest = []
    dropped_examples = []
    durations_kept = []
    # Distribution of every gated/kept numeric column, over the rows that
    # survived -- this is what you read to choose the next threshold.
    field_values = defaultdict(list)

    out = open(args.output, "w", encoding="utf-8") if args.output else None
    arrow_out = ArrowManifestWriter(args.output_arrow) if args.output_arrow else None
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

            # Source-level gates, before the normalizer does any work: a row
            # rejected on DNSMOS or on a junk pattern should not cost a pass.
            if drop_re is not None and drop_re.search(raw):
                stats["drop_text_pattern"] += 1
                if len(dropped_examples) < 10:
                    dropped_examples.append({"reason": "text_pattern",
                                             "raw": raw[:120]})
                continue
            extra = r.get("_extra") or {}
            gated = False
            for name, (lo, hi) in bounds.items():
                v = _as_number(extra.get(name))
                if v is None:
                    stats[f"drop_{name}_missing"] += 1
                    gated = True
                    break
                if (lo is not None and v < lo) or (hi is not None and v > hi):
                    stats[f"drop_{name}_out_of_range"] += 1
                    gated = True
                    break
            if gated:
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
            dur = r.get("duration")
            if args.min_duration or args.max_duration:
                if dur is None:
                    stats["drop_duration_unknown"] += 1
                    continue
                if args.min_duration and dur < args.min_duration:
                    stats["drop_too_short_audio"] += 1
                    continue
                if args.max_duration and dur > args.max_duration:
                    stats["drop_too_long_audio"] += 1
                    if len(dropped_examples) < 10:
                        dropped_examples.append({"reason": "audio_too_long",
                                                 "seconds": round(dur, 2)})
                    continue
            if args.require_audio and not os.path.exists(r["audio"]):
                stats["drop_missing_audio"] += 1
                if len(dropped_examples) < 10:
                    dropped_examples.append({"reason": "missing_audio",
                                             "audio": r["audio"]})
                continue

            stats["kept"] += 1
            densities.append(res.diacritic_density)
            if dur:
                durations_kept.append(dur)
            for name in wanted:
                v = _as_number(extra.get(name))
                if v is not None:
                    field_values[name].append(v)
            spk = r.get("speaker") or args.default_speaker
            if spk:
                speakers[spk] += 1
                if dur:
                    speaker_seconds[spk] += dur

            if out or arrow_out is not None:
                row = {"audio": r["audio"], "text": res.text}
                if dur:
                    row["duration"] = round(float(dur), 3)
                row["dataset_id"] = args.dataset_id
                if not args.no_provenance:
                    row["text_raw"] = raw
                    row["norm_version"] = nz.version
                    row["tier"] = args.tier
                    row["diacritic_density"] = round(res.diacritic_density, 4)
                    if spk:
                        row["speaker"] = spk
                for name in keep_fields:
                    if name in extra and extra[name] is not None:
                        v = extra[name]
                        row[name] = round(v, 4) if isinstance(v, float) else v
                if out:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
            if arrow_out is not None:
                arrow_out.write(row)
    finally:
        if out:
            out.close()
        if arrow_out is not None:
            n_rows, cols = arrow_out.close()
            print(f"arrow manifest: {n_rows} rows -> {args.output_arrow}\n  columns: {', '.join(cols)}", file=sys.stderr)

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

    if field_values:
        rep["field_stats"] = {name: _quantiles(v)
                              for name, v in sorted(field_values.items())}

    if durations_kept:
        rep["audio_duration"] = dict(_quantiles(durations_kept),
                                     total_hours=round(sum(durations_kept) / 3600, 2))

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


def run_try(args):
    """Normalize sentences given on the command line or on stdin, and show what
    the frontend did to them. This is the thing to reach for before trusting a
    corpus pass -- it is the same Normalizer, same config, same version."""
    cfg = NormalizerConfig(
        keep_harakat=not args.strip_harakat,
        decimal_style=args.decimal_style,
        latin_strategy=args.latin_strategy,
        parens_to_commas=not args.keep_parens,
    )
    nz = Normalizer(config=cfg)

    if args.text:
        lines = list(args.text)
    else:
        lines = [ln.rstrip("\n") for ln in sys.stdin if ln.strip()]
    if not lines:
        raise SystemExit("nothing to normalize -- pass sentences as arguments "
                         "or pipe them on stdin")

    print(f"frontend version: {nz.version}", file=sys.stderr)
    for i, raw in enumerate(lines):
        if i:
            print()
        r = nz.normalize(raw)
        print(f"  in   {raw}")
        print(f"  out  {r.text}")
        notes = [f"{r.n_words} words",
                 f"diacritic_density={r.diacritic_density:.3f}"]
        if not r.changed:
            notes.append("unchanged")
        if nz(r.text) != r.text:
            notes.append("NOT IDEMPOTENT")
        print("       " + ", ".join(notes))
        if r.unexpected:
            print("       stripped: " + " ".join(
                f"U+{ord(c):04X} {c!r}" for c in r.unexpected))
        if r.problems:
            print("       diacritic problems: " + ", ".join(r.problems))
        esc = re.findall(r"\{[^{}]*\}", r.text)
        if esc:
            print("       unresolved Latin: " + " ".join(esc))
    if nz.latin.unknown:
        print("\nnot in the lexicon: "
              + ", ".join(t for t, _ in nz.latin.work_queue(20)),
              file=sys.stderr)
    return 0


class ArrowManifestWriter:
    """Write manifest rows as an Arrow IPC stream, readable by
    `datasets.Dataset.from_file()` and by `pyarrow.ipc.open_stream`.

    Rows are buffered because the column set is not known up front -- `duration`
    and the `--keep-fields` columns are per-row optional -- so the schema is the
    union of every key emitted. A manifest is text plus a handful of scalars, so
    this stays small next to the audio it points at.
    """

    def __init__(self, path):
        self.pa = _pyarrow()
        self.path = path
        self.rows = []

    def write(self, row):
        self.rows.append(row)

    def close(self):
        pa = self.pa
        keys = []
        for r in self.rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        table = pa.Table.from_pydict(
            {k: [r.get(k) for r in self.rows] for k in keys})
        with pa.OSFile(self.path, "wb") as sink:
            w = pa.ipc.new_stream(sink, table.schema)
            w.write_table(table)
            w.close()
        return len(self.rows), keys


def run_build(args):
    """Write a self-contained Arrow dataset: the source audio carried through
    untouched, the transcript replaced by its normalized form.

    Unlike `normalize`, which emits a manifest of pointers, this reproduces the
    corpus. The audio never becomes a Python object -- rows are selected with
    an Arrow `take`, so the payload moves buffer-to-buffer and peak memory stays
    at roughly one source batch regardless of corpus size.

    The result is a `save_to_disk()` layout: `datasets.load_from_disk()` reads
    it directly, with `audio` still typed as `Audio(sampling_rate=...)`.
    """
    pa = _pyarrow()
    cfg = NormalizerConfig(
        keep_harakat=not args.strip_harakat,
        decimal_style=args.decimal_style,
        latin_strategy=args.latin_strategy,
        parens_to_commas=not args.keep_parens,
    )
    nz = Normalizer(config=cfg)
    print(f"frontend version: {nz.version}", file=sys.stderr)

    bounds = _parse_bounds(args.min_field, args.max_field)
    keep_fields = [f for f in (args.keep_fields or "").split(",") if f]
    wanted = list(dict.fromkeys(keep_fields + list(bounds)))
    drop_re = re.compile(args.drop_text_matching) if args.drop_text_matching else None

    files = arrow_fragments(args.input, split=args.split)
    outdir = args.output_dataset
    os.makedirs(outdir, exist_ok=True)
    for stale in glob.glob(os.path.join(outdir, "data-*.arrow")):
        os.remove(stale)

    stats = Counter()
    densities = []
    durations = []
    shard_lengths = []
    schema = [None]
    writer = {"w": None, "sink": None, "rows": 0, "bytes": 0, "n": 0}
    shard_bytes = args.shard_mb * 1024 * 1024

    def open_shard():
        path = os.path.join(outdir, f"data-{writer['n']:05d}.arrow")
        writer["sink"] = pa.OSFile(path, "wb")
        writer["w"] = pa.ipc.new_stream(writer["sink"], schema[0])
        writer["rows"] = 0
        writer["bytes"] = 0

    def close_shard():
        if writer["w"] is None:
            return
        writer["w"].close()
        writer["sink"].close()
        shard_lengths.append(writer["rows"])
        writer["w"] = None
        writer["n"] += 1

    for fp in files:
        for batch in _arrow_batches(pa, fp):
            n = batch.num_rows
            if not n:
                continue

            texts = _column(pa, batch, args.text_field)
            if texts is None:
                raise SystemExit(
                    f"--text-field {args.text_field!r} is not a column.\n"
                    "available: " + ", ".join(_field_names(pa, batch.schema)))
            texts = texts.to_pylist()

            audio_col = _column(pa, batch, args.audio_field)
            if audio_col is None:
                raise SystemExit(
                    f"--audio-field {args.audio_field!r} is not a column.\n"
                    "available: " + ", ".join(_field_names(pa, batch.schema)))

            dur_col = _column(pa, batch, args.duration_field)
            if dur_col is not None:
                durs = dur_col.to_pylist()
            elif args.duration_from_audio:
                durs = _audio_durations(pa, audio_col, n)
            else:
                durs = [None] * n

            extras = {}
            for name in wanted:
                col = _column(pa, batch, name)
                if col is not None:
                    extras[name] = col.to_pylist()

            keep, norm_text, raw_text, keep_dur, keep_dens = [], [], [], [], []
            for k in range(n):
                stats["read"] += 1
                raw = (texts[k] or "").strip()
                if not raw:
                    stats["drop_empty_source"] += 1
                    continue
                if drop_re is not None and drop_re.search(raw):
                    stats["drop_text_pattern"] += 1
                    continue
                gated = False
                for name, (lo, hi) in bounds.items():
                    v = _as_number(extras.get(name, [None] * n)[k])
                    if v is None:
                        stats[f"drop_{name}_missing"] += 1
                        gated = True
                        break
                    if (lo is not None and v < lo) or (hi is not None and v > hi):
                        stats[f"drop_{name}_out_of_range"] += 1
                        gated = True
                        break
                if gated:
                    continue
                d = _as_float(durs[k])
                if args.min_duration or args.max_duration:
                    if d is None:
                        stats["drop_duration_unknown"] += 1
                        continue
                    if args.min_duration and d < args.min_duration:
                        stats["drop_too_short_audio"] += 1
                        continue
                    if args.max_duration and d > args.max_duration:
                        stats["drop_too_long_audio"] += 1
                        continue

                res = nz.normalize(raw)
                if not res.text.strip():
                    stats["drop_empty_after_norm"] += 1
                    continue
                if args.min_words and res.n_words < args.min_words:
                    stats["drop_too_short"] += 1
                    continue
                if args.max_chars and len(res.text) > args.max_chars:
                    stats["drop_too_long"] += 1
                    continue

                stats["kept"] += 1
                keep.append(k)
                norm_text.append(res.text)
                raw_text.append(raw)
                keep_dur.append(d)
                keep_dens.append(round(res.diacritic_density, 4))
                densities.append(res.diacritic_density)
                if d:
                    durations.append(d)

            if not keep:
                continue

            idx = pa.array(keep, type=pa.int32())
            cols = {
                "audio": audio_col.take(idx),          # buffers, not Python
                "text": pa.array(norm_text, type=pa.string()),
                "text_raw": pa.array(raw_text, type=pa.string()),
                "duration": pa.array(keep_dur, type=pa.float64()),
                "dataset_id": pa.array([args.dataset_id] * len(keep),
                                       type=pa.int64()),
                "tier": pa.array([args.tier] * len(keep), type=pa.string()),
                "norm_version": pa.array([nz.version] * len(keep),
                                         type=pa.string()),
                "diacritic_density": pa.array(keep_dens, type=pa.float64()),
            }
            for name in keep_fields:
                if name in extras:
                    cols[name] = pa.array([extras[name][k] for k in keep],
                                          type=pa.float64())

            if schema[0] is None:
                schema[0] = _hf_schema(pa, cols, args.sampling_rate)
            rb = pa.RecordBatch.from_arrays(list(cols.values()),
                                            schema=schema[0])
            if writer["w"] is None:
                open_shard()
            writer["w"].write_batch(rb)
            writer["rows"] += rb.num_rows
            writer["bytes"] += rb.nbytes
            if writer["bytes"] >= shard_bytes:
                close_shard()
    close_shard()

    if not shard_lengths:
        raise SystemExit("no rows survived the filters -- nothing written")

    total = sum(shard_lengths)
    _write_hf_metadata(outdir, schema[0], shard_lengths, total, args.split or "train")

    rep = {
        "frontend_version": nz.version,
        "counts": dict(stats),
        "keep_rate": round(stats["kept"] / max(1, stats["read"]), 4),
        "shards": len(shard_lengths),
        "rows": total,
        "diacritic_density": {
            "mean": round(sum(densities) / max(1, len(densities)), 4)},
    }
    if durations:
        rep["audio_duration"] = dict(
            _quantiles(durations),
            total_hours=round(sum(durations) / 3600, 2))
    txt = json.dumps(rep, indent=2, ensure_ascii=False)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(txt)
    print(txt)
    print(f"\ndataset: {total} rows in {len(shard_lengths)} shard(s) -> {outdir}",
          file=sys.stderr)
    return 0


def _hf_schema(pa, cols, sampling_rate):
    """Arrow schema carrying the `datasets` feature block, so `audio` comes back
    as an Audio column rather than a raw struct."""
    features = {}
    for name, arr in cols.items():
        if name == "audio":
            features[name] = {"sampling_rate": sampling_rate, "_type": "Audio"}
        elif pa.types.is_string(arr.type):
            features[name] = {"dtype": "string", "_type": "Value"}
        elif pa.types.is_int64(arr.type):
            features[name] = {"dtype": "int64", "_type": "Value"}
        else:
            features[name] = {"dtype": "float64", "_type": "Value"}
    schema = pa.schema([(n, a.type) for n, a in cols.items()])
    return schema.with_metadata({
        "huggingface": json.dumps({"info": {"features": features}})})


def _write_hf_metadata(outdir, schema, shard_lengths, total, split):
    features = json.loads(
        schema.metadata[b"huggingface"].decode())["info"]["features"]
    with open(os.path.join(outdir, "dataset_info.json"), "w",
              encoding="utf-8") as f:
        json.dump({"builder_name": "persian_tts_frontend", "config_name":
                   "default", "dataset_name": "persian_tts_frontend",
                   "description": "", "citation": "", "homepage": "",
                   "license": "", "features": features,
                   "splits": {split: {"name": split, "num_examples": total,
                                      "shard_lengths": shard_lengths,
                                      "dataset_name": "persian_tts_frontend"}},
                   "version": {"version_str": "1.0.0", "major": 1, "minor": 0,
                               "patch": 0}},
                  f, ensure_ascii=False, indent=2)
    with open(os.path.join(outdir, "state.json"), "w", encoding="utf-8") as f:
        json.dump({"_data_files": [{"filename": f"data-{i:05d}.arrow"}
                                   for i in range(len(shard_lengths))],
                   "_fingerprint": "ptf" + "%012x" % (total * 2654435761 % 2**48),
                   "_format_columns": None, "_format_kwargs": {},
                   "_format_type": None, "_output_all_columns": False,
                   "_split": split}, f, ensure_ascii=False, indent=2)


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


def _as_number(v):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_bounds(min_specs, max_specs):
    """`["mos_ovr=3.5"]`, `["wer=0.2"]` -> `{"mos_ovr": (3.5, None), ...}`."""
    bounds = {}
    for specs, idx in ((min_specs or [], 0), (max_specs or [], 1)):
        for spec in specs:
            name, sep, raw = spec.partition("=")
            name = name.strip()
            if not sep or not name:
                raise SystemExit(
                    f"bad field bound {spec!r} -- expected NAME=VALUE, "
                    "e.g. --min-field mos_ovr=3.5")
            v = _as_number(raw.strip())
            if v is None:
                raise SystemExit(f"bad field bound {spec!r} -- "
                                 f"{raw.strip()!r} is not a number")
            lo, hi = bounds.get(name, (None, None))
            bounds[name] = (v, hi) if idx == 0 else (lo, v)
    return bounds


def _quantiles(values):
    """Distribution summary for a gated column. Reported over kept rows, so
    after a gate it shows what you kept -- rerun without the gate to see the
    whole corpus and pick the next threshold."""
    v = sorted(values)
    n = len(v)

    def q(p):
        return round(v[min(n - 1, int(p * n))], 4)

    return {"n": n, "min": round(v[0], 4), "p10": q(0.10), "p25": q(0.25),
            "median": q(0.50), "p75": q(0.75), "p90": q(0.90),
            "max": round(v[-1], 4),
            "mean": round(sum(v) / n, 4)}


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
    n.add_argument("--output-arrow", default=None, metavar="PATH",
                   help="also write the manifest as an Arrow IPC "
                        "stream (datasets.Dataset.from_file)")
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
    n.add_argument("--duration-from-audio", action="store_true",
                   help="arrow/parquet: derive duration from the embedded "
                        "audio's WAV header when there is no duration column")
    n.add_argument("--min-duration", type=float, default=0.0, metavar="SEC")
    n.add_argument("--max-duration", type=float, default=0.0, metavar="SEC")
    n.add_argument("--strip-harakat", action="store_true")
    n.add_argument("--keep-parens", action="store_true")
    n.add_argument("--decimal-style", default="momayez",
                   choices=["momayez", "fractional"])
    n.add_argument("--latin-strategy", default="escalate",
                   choices=["escalate", "lexicon", "transliterate"])
    n.add_argument("--keep-fields", default=None, metavar="A,B",
                   help="extra source columns to copy into the manifest rows, "
                        "e.g. mos_ovr,mos_p808")
    n.add_argument("--min-field", action="append", default=[], metavar="NAME=V",
                   help="drop rows whose NAME is below V (repeatable), "
                        "e.g. --min-field mos_ovr=3.5")
    n.add_argument("--max-field", action="append", default=[], metavar="NAME=V",
                   help="drop rows whose NAME is above V (repeatable)")
    n.add_argument("--drop-text-matching", default=None, metavar="REGEX",
                   help="drop rows whose raw text matches, e.g. subtitle "
                        r"leftovers: '-->|\d{2}:\d{2}:\d{2}[,.]\d{3}'")
    n.add_argument("--no-provenance", action="store_true")
    n.add_argument("--selftest", action="store_true")
    n.set_defaults(func=run_normalize)

    b = sub.add_parser("build", help="write a full Arrow dataset: source audio "
                                     "carried through, text normalized")
    b.add_argument("--input", required=True)
    b.add_argument("--output-dataset", required=True, metavar="DIR",
                   help="save_to_disk() layout; load_from_disk() reads it")
    b.add_argument("--report", default=None)
    b.add_argument("--split", default=None)
    b.add_argument("--text-field", default="text")
    b.add_argument("--audio-field", default="audio")
    b.add_argument("--duration-field", default="duration")
    b.add_argument("--duration-from-audio", action="store_true")
    b.add_argument("--sampling-rate", type=int, default=16000,
                   help="declared on the Audio feature; must match the audio")
    b.add_argument("--shard-mb", type=int, default=500,
                   help="roll a new shard past this many MB (default 500)")
    b.add_argument("--keep-fields", default=None, metavar="A,B")
    b.add_argument("--min-field", action="append", default=[], metavar="NAME=V")
    b.add_argument("--max-field", action="append", default=[], metavar="NAME=V")
    b.add_argument("--drop-text-matching", default=None, metavar="REGEX")
    b.add_argument("--min-duration", type=float, default=0.0, metavar="SEC")
    b.add_argument("--max-duration", type=float, default=0.0, metavar="SEC")
    b.add_argument("--min-words", type=int, default=2)
    b.add_argument("--max-chars", type=int, default=600)
    b.add_argument("--dataset-id", type=int, default=0)
    b.add_argument("--tier", default="core",
                   choices=["gold", "core", "diversity", "register",
                            "emotion", "replay"])
    b.add_argument("--strip-harakat", action="store_true")
    b.add_argument("--keep-parens", action="store_true")
    b.add_argument("--decimal-style", default="momayez",
                   choices=["momayez", "fractional"])
    b.add_argument("--latin-strategy", default="escalate",
                   choices=["escalate", "lexicon", "transliterate"])
    b.set_defaults(func=run_build)

    t = sub.add_parser("try", help="normalize sentences from the command line "
                                   "or stdin and show what changed")
    t.add_argument("text", nargs="*", help="sentences; omit to read stdin")
    t.add_argument("--strip-harakat", action="store_true")
    t.add_argument("--keep-parens", action="store_true")
    t.add_argument("--decimal-style", default="momayez",
                   choices=["momayez", "fractional"])
    t.add_argument("--latin-strategy", default="escalate",
                   choices=["escalate", "lexicon", "transliterate"])
    t.set_defaults(func=run_try)

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
