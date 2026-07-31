#!/usr/bin/env python3
"""
Run the full tokenizer audit against the real VoxCPM 2 tokenizer.

    python scripts/run_audit.py \
        --model /models/VoxCPM2 \
        --manifest /data/fidibo/manifest.jsonl \
        --tokens-per-second 25 \
        --sample 5000

The --tokens-per-second value MUST be measured, not guessed. To measure it:
push one clip of known duration through the dataloader / AudioVAE encoder plus
patching, read off the resulting latent sequence length, and divide. Then:

    python scripts/run_audit.py ... --latent-len 750 --clip-duration 30

The clip-survival section is the one that matters most. `max_batch_tokens`
drops overlong clips SILENTLY. Compare `kept` against the number of rows your
dataloader actually yields -- if those two numbers disagree you have a second
problem; if they agree and `kept` is low, reconfigure batching or re-segment.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persian_tts_frontend import audit, diacritics


def load_manifest(path, sample=0, seed=0):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "text" in d:
                rows.append(d)
    if sample and len(rows) > sample:
        rows = random.Random(seed).sample(rows, sample)
    return rows


def get_tokenizer(model_path, mock=False):
    if mock:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
        from mock_voxcpm_tokenizer import MockVoxCPMTokenizer
        print("!! using MOCK tokenizer -- estimates only", file=sys.stderr)
        return MockVoxCPMTokenizer()
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--mock", action="store_true",
                   help="use the calibrated mock instead of a real tokenizer")
    p.add_argument("--manifest", required=True)
    p.add_argument("--sample", type=int, default=5000)
    p.add_argument("--tokens-per-second", type=float, default=None)
    p.add_argument("--latent-len", type=int, default=None)
    p.add_argument("--clip-duration", type=float, default=None)
    p.add_argument("--max-batch-tokens", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--default-duration", type=float, default=None,
                   help="fill in for rows lacking a duration field")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    tok = get_tokenizer(a.model, mock=a.mock or not a.model)
    rows = load_manifest(a.manifest, a.sample)
    texts = [r["text"] for r in rows]
    print(f"loaded {len(rows)} rows from {a.manifest}", file=sys.stderr)

    results = {}

    # --- 1. roundtrip -------------------------------------------------------
    if not (a.mock or not a.model):
        rt = audit.roundtrip(tok)
        results["roundtrip"] = {"all_passed": rt["all_passed"],
                                "failures": rt["failures"]}
        print("\n### roundtrip")
        for r in rt["rows"]:
            flag = "ok " if r["equal"] else "FAIL"
            print(f"  {flag} {r['input']!r:28} chars={r['n_chars']:3} "
                  f"tokens={r['n_tokens']:3} "
                  f"({r['n_tokens'] / max(1, r['n_chars']):.2f} tok/char)")
        if not rt["all_passed"]:
            print("\n!! ROUNDTRIP FAILED -- the sparse-diacritics design is "
                  "invalid until this is resolved. Stop here.", file=sys.stderr)

    # --- 2. fertility -------------------------------------------------------
    fert = audit.fertility(tok, texts)
    results["fertility"] = fert
    print("\n### fertility")
    for k, v in fert.items():
        print(f"  {k:32} {v if not isinstance(v, float) else round(v, 4)}")

    # --- 3. diacritic cost --------------------------------------------------
    dens = diacritics.report(texts)
    results["diacritic_stats"] = dens
    print("\n### diacritic density in this manifest")
    for k, v in dens.items():
        print(f"  {k:32} {v if not isinstance(v, float) else round(v, 4)}")

    marked = [t for t in texts if diacritics.has_marks(t)]
    if marked:
        curve = audit.diacritic_cost_curve(tok, marked[:2000])
        results["diacritic_cost_curve"] = curve
        print("\n### sequence-length cost vs marking density")
        print("  (run on already-marked text; compare against full marking)")
        for r in curve:
            print(f"  keep_rate={r['keep_rate']:.2f}  "
                  f"tokens={r['total_tokens']:8}  "
                  f"overhead={r.get('overhead_pct', 0):+.1f}%")
    else:
        print("\n### diacritic cost: manifest has no marks yet -- rerun after "
              "the diacritizer pass")

    # --- 4. clip survival ---------------------------------------------------
    tps = a.tokens_per_second
    if tps is None and a.latent_len and a.clip_duration:
        tps = audit.measure_tokens_per_second(a.latent_len, a.clip_duration)
        print(f"\nmeasured tokens_per_second = {tps:.2f}", file=sys.stderr)
    if tps is None:
        print("\n!! --tokens-per-second not supplied; skipping the clip-survival "
              "check.\n   This is the check most likely to save you a training "
              "run. Measure it.", file=sys.stderr)
    else:
        surv_rows = []
        missing_dur = 0
        for r in rows:
            d = r.get("duration") or a.default_duration
            if d is None:
                missing_dur += 1
                continue
            surv_rows.append({"text": r["text"], "duration": float(d)})
        if missing_dur:
            print(f"\n!! {missing_dur} rows lack a duration field and were "
                  f"skipped; pass --default-duration to include them",
                  file=sys.stderr)
        if surv_rows:
            surv = audit.survival(tok, surv_rows, tps,
                                  max_batch_tokens=a.max_batch_tokens,
                                  batch_size=a.batch_size)
            results["survival"] = surv
            print("\n### clip survival "
                  f"(max_batch_tokens={a.max_batch_tokens}, "
                  f"batch_size={a.batch_size})")
            for k, v in surv.items():
                if k != "dropped_examples":
                    print(f"  {k:32} {v}")
            if surv["dropped_examples"]:
                print("  dropped examples:")
                for e in surv["dropped_examples"]:
                    print(f"    {e}")

            grid = audit.suggest_batching(tok, surv_rows, tps)
            results["batching_grid"] = grid
            print("\n### batching options")
            print(f"  {'mbt':>7} {'bs':>4} {'ceiling':>8} {'drop%':>7} "
                  f"{'kept_h':>9}  ok")
            for g in grid:
                print(f"  {g['max_batch_tokens']:>7} {g['batch_size']:>4} "
                      f"{g['per_clip_ceiling']:>8} {g['dropped_pct']:>7} "
                      f"{g['kept_hours']:>9}  "
                      f"{'yes' if g['acceptable'] else 'no'}")

    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"\nwrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
