# persian_tts_frontend

Step 1 of the Persian VoxCPM 2 build plan: text normalization, numeric
expansion, Latin resolution, diacritic masking, and tokenizer auditing.

**This package is the frozen artifact.** It must be byte-identical at training
time and at serving time. Frontend/model skew is the standard way to ship a TTS
system that scores well offline and mispronounces things in production. Ship it
in the same container as the model and refuse to serve if the recorded
`norm_version` does not match the deployed frontend.

```
pip install -e .
python tests/test_frontend.py        # 17 test functions, all must pass
```

---

## Quick start

```python
from persian_tts_frontend import Normalizer

nz = Normalizer()
print(nz.version)          # 'fa-fe-9c75e8f74268' -> record in every manifest row
print(nz("در سال ۱۴۰۳ حدود ٪۲۵ رشد داشت."))
# در سال هزار و چهارصد و سه حدود بیست و پنج درصد رشد داشت.
```

`normalize()` returns a `NormResult` with the diagnostics you actually need:

```python
r = nz.normalize(raw)
r.text                 # normalized
r.unexpected           # chars that survived the charset guard -> a bug or exotic input
r.problems             # structural diacritic problems
r.diacritic_density    # fraction of words carrying marks
r.n_words
```

---

## Corpus pass

```bash
# 1. ALWAYS dry-run on a slice first and read the report
python -m persian_tts_frontend.cli normalize \
    --input /data/fidibo/raw.jsonl --adapter jsonl \
    --speaker-field narrator \
    --output /tmp/probe.jsonl --report /tmp/probe.json \
    --limit 5000 --selftest

# 2. full pass, per tier
python -m persian_tts_frontend.cli normalize \
    --input /data/fidibo/raw.jsonl --adapter jsonl --speaker-field narrator \
    --output /data/manifests/fidibo.jsonl --dataset-id 1 --tier gold \
    --require-audio --report /data/manifests/fidibo_report.json

# 3. diversity tier -- Common Voice Persian
python -m persian_tts_frontend.cli normalize \
    --input /data/cv-fa/validated.tsv --adapter commonvoice \
    --audio-root /data/cv-fa/clips \
    --output /data/manifests/cv.jsonl --dataset-id 3 --tier diversity
```

Adapters: `jsonl`, `csv`, `tsv`, `commonvoice`, `trans` (LibriSpeech-style
`{utt_id} {transcript}` lines). For thomclas, use `jsonl`/`csv`/`tsv` with
`--text-field` / `--audio-field` / `--speaker-field` pointed at its columns.

### Read these four numbers in the report before proceeding

| Field | What it means |
|---|---|
| `unexpected_chars` | **Must be empty.** Anything here escaped normalization — extend `chars.FOLD_MAP` or `chars.allowed_charset` and rerun. |
| `speakers.n_distinct` | Below ~100 and zero-shot cloning will generalize poorly. Grow the diversity tier. |
| `speakers.gini_hint` | Share of rows held by the top 10% of speakers. Above ~0.5, enforce the per-narrator cap. |
| `latin_work_queue` | Ranked unresolved Latin tokens. This is your lexicon backlog — paste the top entries into `latin.LEXICON` and rerun. |

`--selftest` additionally verifies idempotence across a 2000-row sample. Any
`selftest_failures` is a bug in this package, not in your data.

---

## Tokenizer audit — run before Stage A

```bash
python scripts/run_audit.py \
    --model /models/VoxCPM2 \
    --manifest /data/manifests/fidibo.jsonl \
    --latent-len 750 --clip-duration 30 \
    --sample 5000 --out audit.json
```

The **clip-survival** section is the one that will save you a training run.
`max_batch_tokens // batch_size` is a hard per-clip ceiling and overlong clips
are dropped **silently** — no warning, no error, they simply never appear in a
batch. Measured on realistic 30s Persian narration:

```
mbt= 8192 bs=8  ceiling= 1024  dropped=30.0%   <- every 30s clip lost
mbt= 8192 bs=4  ceiling= 2048  dropped= 0.0%
```

A 30s clip runs ~321 text tokens + ~750 audio tokens = 1079, against a 1024
ceiling. It misses by 55 tokens and disappears.

`--latent-len` / `--clip-duration` must be **measured**: push one clip of known
duration through the dataloader/AudioVAE encoder plus patching and read off the
latent sequence length. Do not guess the patch rate.

`--mock` runs a calibrated mock tokenizer instead, for offline estimates.

---

## Diacritic masking — wire into the dataloader, not the manifest

Masking is resampled per epoch, so it belongs in the dataloader. The manifest
stores the *maximally* marked text; the loader decides how much of it the model
sees on this pass.

### Stage A (language acquisition)

```python
import json
from persian_tts_frontend import MaskingSchedule

HOMOGRAPHS = set(json.load(open("data/homographs_seed.json"))["homographs"])
sched = MaskingSchedule(total_steps=8000)     # match max_steps in the YAML

text = sched.apply(row["text"], step=global_step, priority=HOMOGRAPHS)
```

Curriculum: `keep_rate` 0.9 for the first 20% of steps so the
grapheme→phoneme mapping forms against a clean signal, annealing to
`U(0,1)` thereafter so the model learns to work from bare text and to treat
marks as soft hints. Reversing this order measurably hurts.

### Stage B (polish on Gemini-labeled gold)

```python
from persian_tts_frontend import diacritics
text = diacritics.mask(row["text"],
                       keep_rate=diacritics.sample_density(),
                       priority=HOMOGRAPHS)
```

`sample_density()` draws from `STAGE_B_DENSITY_MIX`: 60% sparse (0.10–0.25),
25% medium, 15% full. Gemini gives fully-marked ground truth; sampling from it
keeps the labels correct while matching the sparse distribution your frontend
actually emits at inference. **Training Stage B at 100% marking is the mistake
this mix exists to prevent** — it leaves the model's final gradients assuming
an input distribution production cannot reproduce.

Word-final kasra (ezafe) is never dropped.

---

## Pipeline order

Order is load-bearing. Do not reorder without rerunning the tests.

1. **Protect `{phoneme}` escapes** → PUA placeholders
2. Presentation forms → base letters; NFC
3. Remove invisibles (**keep ZWNJ**); fold codepoints; drop Arabic-only marks
4. Punctuation normalization (paired straight quotes → guillemets)
5. Numeric/symbolic expansion — abbreviations, thousands, time, dates, phone,
   currency, percent, units, fractions, ordinals, ranges, decimals, integers
6. Latin resolution; re-protect newly emitted escapes
7. Cleanup (stray dashes → pause, collapse punctuation runs), ZWNJ repair, spacing
8. NFC again — expansion inserted new letters
9. **Charset guard** — report and strip anything outside the inventory
10. Restore escapes

### On combining-mark order

Arabic harakat all carry distinct nonzero canonical combining classes
(fathatan 27 … sukun 34), so **NFC already reorders them deterministically**.
That canonical order puts the vowel *before* shadda — the opposite of typing
convention. Do not hand-roll a reorder pass that fights NFC; apply NFC
consistently and assert idempotence. Only consistency matters to the tokenizer,
not which order wins. `test_nfc_reorders_marks` pins this down.

---

## Phoneme escape hatch

```python
nz("به فرودگاه {mehrɒbɒd} رسیدیم")   # -> '...{mehrɒbɒd}...' passed through intact
```

Train it into ~3% of samples and expose it in the serving API. It is how
production forces pronunciation of brand names, foreign names and acronyms
without retraining. With `latin_strategy="escalate"` (the default) any Latin
token missing from the lexicon is *automatically* escalated to an escape rather
than silently transliterated — safer for training data, and it populates
`latin.work_queue()` so the lexicon gets built from what actually occurs.

---

## Invariants

Every one of these is enforced by the test suite:

- **Idempotent**: `nz(nz(x)) == nz(x)` for all inputs
- **Deterministic**: no randomness in `Normalizer`; all randomness lives in
  `diacritics` and takes an explicit `Random`
- **Versioned**: `nz.version` hashes the rule source *and* the config, so any
  edit anywhere changes it
- **Total**: every character in the output is in `chars.allowed_charset()`
- **ZWNJ preserved**, harakat roundtrip exactly
- **No digits survive** numeric expansion
- Escapes pass through untouched

---

## Files

```
persian_tts_frontend/
  chars.py        codepoint constants, fold map, charset guard
  numbers.py      cardinals, ordinals, decimals, fractions, digit-reading
  expand.py       dates, times, currency, percent, units, ranges, abbreviations
  normalize.py    NFC, folding, punctuation, cleanup, escape protection
  latin.py        transliteration lexicon, acronym spelling, work queue
  diacritics.py   strip/mask/validate, MaskingSchedule, STAGE_B_DENSITY_MIX
  pipeline.py     Normalizer, config, version hash, selftest
  audit.py        roundtrip, fertility, cost curve, clip survival
  cli.py          dataset adapters + manifest builder + report
data/
  homographs_seed.json    ~57 entries with diacritized readings and glosses
scripts/run_audit.py
examples/mock_voxcpm_tokenizer.py
tests/test_frontend.py
```

---

## Known limits

- **Ordinals** are rule-based and correct for the common range; exotic values
  may need special-casing.
- **Rule-based transliteration** in `latin.py` is deliberately crude. It exists
  so an unknown token produces *something* pronounceable. Anything that matters
  belongs in `LEXICON`. Prefer `latin_strategy="escalate"` for training data.
- **`data/homographs_seed.json` is a seed**, not a lexicon. Extend it from
  HomoRich and from your own corpus frequency counts before Stage A — Track B
  forced resolution is only as good as this list's coverage.
- **Date disambiguation** assumes Jalali when the year falls in 1200–1500 and
  Gregorian otherwise. Fine for Persian text; check it against your corpus.
- **Decimal style** defaults to `momayez` (digit-by-digit after the point),
  which is how it is normally said aloud. Use `fractional` for mathematical
  register.
- Invisible-character removal follows Unicode semantics: ZWSP and soft hyphen
  are break opportunities *within* a word and are removed, not converted to
  spaces. If your source uses them as word separators, preprocess first.

---

## Next

1. Extend `latin.LEXICON` from the corpus work queue; rerun.
2. Extend `data/homographs_seed.json` from HomoRich; this gates Track B.
3. Measure the AudioVAE patch rate and run `scripts/run_audit.py` for real.
4. Build the eval set (Step 1.3 of the plan) and pin `nz.version` alongside it.
5. Then Step 2: the two-track diacritizer.
