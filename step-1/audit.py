"""
Tokenizer audit.

Three jobs:

  1. Verify harakat and ZWNJ survive a roundtrip (they do on VoxCPM 2, but
     re-verify after any tokenizer change).
  2. Measure real fertility on your corpus, and the real cost of diacritics.
  3. Estimate clip survival under `max_batch_tokens // batch_size`.

(3) is the one that will save you a training run. That filter drops overlong
clips SILENTLY -- no warning, no error, they simply never appear in a batch.
Persian text is token-expensive (Persian-specific letters and every harakat are
byte-fallback pairs), so 30s audiobook clips are far more likely to be dropped
than the English examples in the docs suggest.
"""

from collections import Counter

from .chars import BYTE_FALLBACK_LETTERS, HARAKAT, ZWNJ
from . import diacritics

ROUNDTRIP_PROBES = [
    "کِرْم", "پژگچ", "می\u200cروم", "نیم\u200cفاصله", "واقعاً",
    "دانشگاهِ تهران", "۱۴۰۳", "مُرْد", "مَرْد", "سِپاهانِ اصفهان",
    "کُتُبْ", "بچّه", "مسئله", "آنها", "«نقل قول»", "سه ممیز یک چهار",
]


def roundtrip(tokenizer, probes=None) -> dict:
    """Every probe must roundtrip exactly. A single failure invalidates the
    whole sparse-diacritics design and must be resolved before training."""
    probes = probes or ROUNDTRIP_PROBES
    rows, failures = [], []
    for p in probes:
        ids = tokenizer.encode(p, add_special_tokens=False)
        dec = tokenizer.decode(ids)
        ok = dec == p
        rows.append({
            "input": p, "n_tokens": len(ids), "n_chars": len(p),
            "decoded": dec, "equal": ok,
            "tokens": tokenizer.convert_ids_to_tokens(ids),
        })
        if not ok:
            failures.append(p)
    return {"rows": rows, "failures": failures,
            "all_passed": not failures}


def fertility(tokenizer, texts) -> dict:
    """Tokens per character, overall and broken down by what drives it."""
    tot_tok = tot_chars = 0
    bare_tok = bare_chars = 0
    n_marks = n_bf_letters = 0
    per_text = []
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)
        bare = diacritics.strip(t)
        bids = tokenizer.encode(bare, add_special_tokens=False)
        tot_tok += len(ids); tot_chars += len(t)
        bare_tok += len(bids); bare_chars += len(bare)
        n_marks += diacritics.count_marks(t)
        n_bf_letters += sum(1 for c in t if c in BYTE_FALLBACK_LETTERS)
        per_text.append(len(ids))
    if not tot_chars:
        return {}
    per_text.sort()
    n = len(per_text)
    return {
        "n_texts": n,
        "tok_per_char": tot_tok / tot_chars,
        "tok_per_char_bare": bare_tok / max(1, bare_chars),
        "diacritic_overhead_pct": 100.0 * (tot_tok - bare_tok) / max(1, bare_tok),
        "tokens_per_mark": (tot_tok - bare_tok) / max(1, n_marks),
        "byte_fallback_letter_share": n_bf_letters / tot_chars,
        "mean_tokens": tot_tok / n,
        "median_tokens": per_text[n // 2],
        "p95_tokens": per_text[int(n * 0.95)],
        "max_tokens": per_text[-1],
    }


def diacritic_cost_curve(tokenizer, texts, densities=(0.0, 0.15, 0.3, 0.5, 1.0)):
    """Sequence-length cost as a function of marking density.

    Run this on fully-diacritized text to see exactly what sparse marking buys
    you. On VoxCPM 2 the full-marking penalty lands near +90%.
    """
    import random
    rng = random.Random(0)
    base = None
    out = []
    for d in densities:
        total = 0
        for t in texts:
            masked = diacritics.mask(t, keep_rate=d, rng=rng)
            total += len(tokenizer.encode(masked, add_special_tokens=False))
        if base is None:
            base = total if d == 0.0 else None
        out.append({"keep_rate": d, "total_tokens": total})
    b = next((r["total_tokens"] for r in out if r["keep_rate"] == 0.0), None)
    if b:
        for r in out:
            r["overhead_pct"] = round(100.0 * (r["total_tokens"] - b) / b, 1)
    return out


# --------------------------------------------------------- clip survival check

def audio_tokens(duration_s: float, tokens_per_second: float) -> int:
    """Audio-side token count. `tokens_per_second` is the effective patch rate
    of the VoxCPM 2 local encoder and MUST be measured, not guessed -- see
    `measure_tokens_per_second`."""
    return int(round(duration_s * tokens_per_second))


def measure_tokens_per_second(latent_len: int, duration_s: float) -> float:
    """Derive the patch rate from one real clip.

    Push a single clip of known duration through the dataloader / AudioVAE
    encoder plus patching, read off the resulting latent sequence length, and
    pass both here. Do this before Stage A.
    """
    return latent_len / max(1e-9, duration_s)


def survival(tokenizer, rows, tokens_per_second: float,
             max_batch_tokens: int = 8192, batch_size: int = 8,
             overhead_tokens: int = 8) -> dict:
    """How many clips survive the length filter?

    rows: iterable of dicts with "text" and "duration" keys.

    The per-clip ceiling is `max_batch_tokens // batch_size`. Anything above it
    is dropped without notice. Compare the `kept` figure here against the
    number of rows your dataloader actually yields -- if they disagree, the
    problem is elsewhere; if they agree and `kept` is low, reconfigure.
    """
    ceiling = max_batch_tokens // batch_size
    kept, dropped = 0, 0
    dropped_hours = kept_hours = 0.0
    by_bucket = Counter()
    examples = []
    for r in rows:
        n_text = len(tokenizer.encode(r["text"], add_special_tokens=False))
        n_audio = audio_tokens(r["duration"], tokens_per_second)
        total = n_text + n_audio + overhead_tokens
        bucket = int(r["duration"] // 5) * 5
        if total > ceiling:
            dropped += 1
            dropped_hours += r["duration"] / 3600.0
            by_bucket[f"{bucket}-{bucket+5}s"] += 1
            if len(examples) < 5:
                examples.append({"duration": r["duration"], "n_text": n_text,
                                 "n_audio": n_audio, "total": total})
        else:
            kept += 1
            kept_hours += r["duration"] / 3600.0
    n = kept + dropped
    return {
        "per_clip_ceiling": ceiling,
        "n_rows": n,
        "kept": kept,
        "dropped": dropped,
        "dropped_pct": round(100.0 * dropped / max(1, n), 2),
        "kept_hours": round(kept_hours, 1),
        "dropped_hours": round(dropped_hours, 1),
        "dropped_by_duration": dict(by_bucket),
        "dropped_examples": examples,
        "verdict": ("OK" if dropped / max(1, n) < 0.02 else
                    "RECONFIGURE: lower batch_size, raise max_batch_tokens, "
                    "or re-segment"),
    }


def suggest_batching(tokenizer, rows, tokens_per_second: float,
                     target_drop_pct: float = 1.0,
                     candidates=((8192, 8), (8192, 4), (8192, 2),
                                 (16384, 8), (16384, 4), (16384, 2))) -> list:
    """Grid over (max_batch_tokens, batch_size) and report drop rates so the
    tradeoff against effective batch size is explicit."""
    out = []
    rows = list(rows)
    for mbt, bs in candidates:
        s = survival(tokenizer, rows, tokens_per_second,
                     max_batch_tokens=mbt, batch_size=bs)
        out.append({"max_batch_tokens": mbt, "batch_size": bs,
                    "per_clip_ceiling": s["per_clip_ceiling"],
                    "dropped_pct": s["dropped_pct"],
                    "kept_hours": s["kept_hours"],
                    "acceptable": s["dropped_pct"] <= target_drop_pct})
    return out


def print_report(d: dict, title: str = ""):
    import json as _j
    if title:
        print("=" * 78); print(title); print("=" * 78)
    print(_j.dumps(d, indent=2, ensure_ascii=False, default=str))
