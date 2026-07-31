"""Test suite. Run: python -m pytest tests/ -q  (or python tests/test_frontend.py)"""

import random
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persian_tts_frontend import Normalizer, NormalizerConfig, diacritics, numbers
from persian_tts_frontend.chars import (FATHA, HARAKAT, KASRA, SHADDA, ZWNJ)
from persian_tts_frontend import normalize as nrm

FAILURES = []


def check(cond, label, got=None, want=None):
    if cond:
        return True
    FAILURES.append(f"{label}\n    got:  {got!r}\n    want: {want!r}")
    return False


def eq(got, want, label):
    return check(got == want, label, got, want)


# --------------------------------------------------------------------- numbers

def test_cardinals():
    cases = {
        0: "صفر", 1: "یک", 7: "هفت", 10: "ده", 11: "یازده", 15: "پانزده",
        19: "نوزده", 20: "بیست", 21: "بیست و یک", 30: "سی", 42: "چهل و دو",
        99: "نود و نه", 100: "صد", 101: "صد و یک", 110: "صد و ده",
        200: "دویست", 300: "سیصد", 500: "پانصد", 900: "نهصد",
        999: "نهصد و نود و نه",
        1000: "هزار", 1001: "هزار و یک",
        1403: "هزار و چهارصد و سه",
        2000: "دو هزار",
        12345: "دوازده هزار و سیصد و چهل و پنج",
        100000: "صد هزار",
        1000000: "یک میلیون",
        1234567: "یک میلیون و دویست و سی و چهار هزار و پانصد و شصت و هفت",
        1000000000: "یک میلیارد",
        -5: "منفی پنج",
    }
    for n, want in cases.items():
        eq(numbers.cardinal(n), want, f"cardinal({n})")


def test_ordinals():
    cases = {1: "اول", 2: "دوم", 3: "سوم", 4: "چهارم", 5: "پنجم", 9: "نهم",
             10: "دهم", 11: "یازدهم", 20: "بیستم", 21: "بیست و یکم",
             23: "بیست و سوم", 30: "سی" + ZWNJ + "ام", 40: "چهلم",
             100: "صدم", 1000: "هزارم"}
    for n, want in cases.items():
        eq(numbers.ordinal(n), want, f"ordinal({n})")


def test_digits_and_decimals():
    eq(numbers.digits("09123"), "صفر نه یک دو سه", "digits")
    eq(numbers.decimal("3", "14"), "سه ممیز یک چهار", "decimal momayez")
    eq(numbers.decimal("3", "14", style="fractional"),
       "سه و چهارده صدم", "decimal fractional")
    eq(numbers.fraction(1, 2), "نیم", "fraction 1/2")
    eq(numbers.fraction(3, 4), "سه چهارم", "fraction 3/4")


# ----------------------------------------------------------- NFC mark ordering

def test_nfc_reorders_marks():
    """The claim the design rests on: NFC canonically orders Arabic harakat,
    so no hand-rolled reorder pass is needed -- and it puts the vowel BEFORE
    shadda, opposite to typing convention."""
    shadda_first = "\u0628" + SHADDA + FATHA        # ب + shadda + fatha
    vowel_first = "\u0628" + FATHA + SHADDA
    a = unicodedata.normalize("NFC", shadda_first)
    b = unicodedata.normalize("NFC", vowel_first)
    eq(a, b, "NFC collapses both mark orders to one form")
    check(a == vowel_first,
          "NFC canonical order is vowel-before-shadda",
          [hex(ord(c)) for c in a], [hex(ord(c)) for c in vowel_first])
    check(nrm.assert_nfc_idempotent(a), "NFC idempotent")


# ------------------------------------------------------------------ normalizer

def test_folding():
    nz = Normalizer()
    eq(nz("كتاب"), "کتاب", "Arabic kaf -> Persian kaf")
    eq(nz("مي‌رود"), "می" + ZWNJ + "رود", "Arabic yeh -> Persian yeh")
    eq(nz("علي"), "علی", "final Arabic yeh")
    check(ZWNJ in nz("می\u200cروم"), "ZWNJ preserved", nz("می\u200cروم"), "has ZWNJ")
    check("\u0640" not in nz("کتــاب"), "tatweel removed")
    check("\u200f" not in nz("\u200fسلام"), "RLM removed")


def test_numeric_expansion():
    nz = Normalizer()
    eq(nz("سال ۱۴۰۳"), "سال هزار و چهارصد و سه", "Persian digits year")
    eq(nz("۲۵٪"), "بیست و پنج درصد", "Persian percent sign")
    eq(nz("%25"), "بیست و پنج درصد", "ASCII percent")
    check("ساعت چهارده و سی دقیقه" in nz("۱۴:۳۰"), "time", nz("۱۴:۳۰"), "ساعت ...")
    check("مرداد" in nz("۱۴۰۳/۰۵/۱۲"), "jalali date", nz("۱۴۰۳/۰۵/۱۲"), "has مرداد")
    check("سپتامبر" in nz("2024/09/15"), "gregorian date",
          nz("2024/09/15"), "has سپتامبر")
    check("صفر نه" in nz("۰۹۱۲۳۴۵۶۷۸۹"), "phone digit-by-digit",
          nz("۰۹۱۲۳۴۵۶۷۸۹"), "starts صفر نه")
    check("دلار" in nz("$500"), "currency symbol", nz("$500"), "has دلار")
    check("تا" in nz("۱۰-۲۰"), "range", nz("۱۰-۲۰"), "has تا")
    check(not any(c.isdigit() for c in nz("در ۱۳۹۹ حدود ۲۵٪ و ۳۰۰ نفر")),
          "no ASCII digits survive expansion")


def test_punctuation():
    nz = Normalizer()
    check("\u060c" in nz("سلام, خوبی"), "ASCII comma -> Persian comma")
    check("\u061f" in nz("خوبی?"), "ASCII question -> Persian question")
    check("\u2026" in nz("خب...."), "ellipsis collapsed")
    check("!" in nz("عجب!!!") and nz("عجب!!!").count("!") == 1,
          "repeated bang collapsed", nz("عجب!!!"), "one !")
    check("\u00ab" in nz('او گفت "سلام"'), "straight quotes -> guillemets",
          nz('او گفت "سلام"'), "has «")


def test_escapes_survive():
    nz = Normalizer()
    out = nz("به فرودگاه {mehrɒbɒd} رسیدیم")
    check("{mehrɒbɒd}" in out, "phoneme escape untouched", out, "{mehrɒbɒd}")
    out2 = nz("عدد {ʃɒnzdæh} و ۱۶")
    check("{ʃɒnzdæh}" in out2 and "شانزده" in out2,
          "escape survives alongside expansion", out2, "both")


def test_charset_guard():
    nz = Normalizer()
    r = nz.normalize("سلام \u2603 دنیا")            # snowman
    check("\u2603" in r.unexpected or "\u2603" not in r.text,
          "unexpected char detected or stripped", r.unexpected, "detected")
    check("\u2603" not in r.text, "unexpected char stripped from output")


def test_idempotence():
    nz = Normalizer()
    samples = [
        "در سال ۱۴۰۳ حدود ٪۲۵ رشد داشت.",
        "او گفت: «مي‌روم به دانشگاهِ تهران»",
        "قیمت ۱۲٬۵۰۰ تومان بود، یعنی حدود $3.5",
        "ساعت ۱۴:۳۰ روز ۱۴۰۳/۰۵/۱۲ در google جلسه داریم.",
        "کِرْمِ ابریشم و بچّه‌ها",
        "شماره‌اش ۰۹۱۲۳۴۵۶۷۸۹ است و ۲۵-۳۰ درصد تخفیف دارد.",
        "متن با  فاصله‌های   زیاد   و ‌‌ZWNJ اضافی",
    ]
    for s in samples:
        check(nz.check_idempotent(s), f"idempotent: {s[:38]}",
              nz(nz(s)), nz(s))
    fails = nz.selftest(samples)
    check(not fails, "selftest clean", fails, {})


# ------------------------------------------------------------------ diacritics

def test_diacritic_utils():
    t = "کِرْمِ ابریشم"
    eq(diacritics.strip(t), "کرم ابریشم", "strip marks")
    eq(diacritics.count_marks(t), 3, "count marks")
    check(0 < diacritics.word_density(t) <= 1.0, "word density in range",
          diacritics.word_density(t), "0<d<=1")
    eq(round(diacritics.word_density(t), 3), 0.5, "1 of 2 words marked")
    check(diacritics.ends_with_ezafe("کِرْمِ"), "ezafe detected")
    check(not diacritics.ends_with_ezafe("کِرْم"), "non-ezafe not flagged")


def test_masking():
    rng = random.Random(0)
    t = "مَرْدِ بُزُرْگ کِتاب خَرید"
    bare = diacritics.strip(t)
    eq(diacritics.mask(t, keep_rate=0.0, rng=rng, always_keep_ezafe=False),
       bare, "keep_rate=0 strips everything")
    eq(diacritics.mask(t, keep_rate=1.0, rng=rng), t, "keep_rate=1 is identity")
    # ezafe preserved even at keep_rate 0
    kept = diacritics.mask("مَرْدِ بزرگ", keep_rate=0.0, rng=rng)
    check(kept.split()[0].endswith(KASRA), "ezafe survives keep_rate=0",
          kept, "ends with kasra")
    # priority words biased upward
    hits = sum(
        1 for _ in range(400)
        if diacritics.has_marks(
            diacritics.mask("مَرْد بُزُرْگ", keep_rate=0.0,
                            rng=random.Random(_), priority={"مرد"},
                            priority_boost=1.0).split()[0])
    )
    check(hits > 380, "priority words kept at boost=1.0", hits, ">380")


def test_masking_schedule():
    s = diacritics.MaskingSchedule(total_steps=1000, seed=7)
    eq(s.keep_rate(0), 0.9, "warm phase keep_rate")
    eq(s.keep_rate(100), 0.9, "still warm at 10%")
    mid = [s.keep_rate(i) for i in range(200, 500, 10)]
    check(all(0.0 <= k <= 0.9 for k in mid), "anneal phase in range")
    late = [s.keep_rate(i) for i in range(600, 1000, 10)]
    check(min(late) < 0.35 and max(late) > 0.65,
          "match phase spans the full range",
          (round(min(late), 2), round(max(late), 2)), "wide")


def test_stage_b_density_mix():
    rng = random.Random(3)
    ds = [diacritics.sample_density(rng) for _ in range(4000)]
    sparse = sum(1 for d in ds if d <= 0.25) / len(ds)
    full = sum(1 for d in ds if d >= 1.0) / len(ds)
    check(0.52 < sparse < 0.68, "≈60% sparse", round(sparse, 3), "~0.60")
    check(0.10 < full < 0.20, "≈15% full", round(full, 3), "~0.15")


def test_validate_catches_bad_marks():
    check(diacritics.validate("کِرْم") == [], "clean text validates",
          diacritics.validate("کِرْم"), [])
    check("mark_at_start" in diacritics.validate(KASRA + "کرم"),
          "leading mark flagged")
    bad = "کرم" + KASRA + KASRA
    check(any("double_mark" in p for p in diacritics.validate(bad)),
          "double vowel flagged", diacritics.validate(bad), "double_mark")


# ------------------------------------------------------------------ versioning

def test_version_stability():
    a, b = Normalizer(), Normalizer()
    eq(a.version, b.version, "version deterministic")
    c = Normalizer(config=NormalizerConfig(decimal_style="fractional"))
    check(c.version != a.version, "config change alters version",
          c.version, f"!= {a.version}")
    check(a.version.startswith("fa-fe-"), "version prefix", a.version, "fa-fe-*")


def test_latin_resolution():
    nz = Normalizer(config=NormalizerConfig(latin_strategy="escalate"))
    out = nz("در google جستجو کردم")
    eq(out, "در گوگل جستجو کردم", "lexicon hit")
    out2 = nz("نرم‌افزار Zqxwv را نصب کن")
    check("{zqxwv}" in out2, "unknown Latin escalated to escape hatch",
          out2, "{zqxwv}")
    check(nz.latin.unknown["zqxwv"] >= 1, "unknown recorded in work queue")
    nz3 = Normalizer(config=NormalizerConfig(latin_strategy="transliterate"))
    check(not any(c.isascii() and c.isalpha() for c in nz3("برند Sharp")),
          "transliterate leaves no Latin", nz3("برند Sharp"), "no ascii")


# -------------------------------------------------------------- arrow adapter

def test_arrow_adapter():
    """Skipped when pyarrow is absent -- the rest of the package is stdlib."""
    try:
        import pyarrow as pa
    except ImportError:
        return
    import tempfile
    from persian_tts_frontend.cli import adapter_arrow, arrow_fragments

    audio_t = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    rows = ["در سال ۱۴۰۳ رشد داشت", "قیمت ۱۲٫۵ میلیون تومان"]

    def tbl(with_path):
        return pa.table({
            "audio": pa.array(
                [{"bytes": b"RIFF", "path": f"c/{i}.wav" if with_path else None}
                 for i in range(len(rows))], type=audio_t),
            "text": pa.array(rows, type=pa.large_string()),
            "narrator": pa.array(["spk_a", "spk_b"]),
            "duration": pa.array([1.5, 2.5], type=pa.float64()),
        })

    with tempfile.TemporaryDirectory() as d:
        # `datasets.save_to_disk` writes the IPC *stream* encoding ...
        stream = Path(d) / "ds"
        stream.mkdir()
        with pa.OSFile(str(stream / "data-00000-of-00001.arrow"), "wb") as sink:
            w = pa.ipc.new_stream(sink, tbl(True).schema)
            w.write_table(tbl(True))
            w.close()
        # ... while `Table.to_file`/Feather writes the file encoding.
        with pa.OSFile(str(Path(d) / "one.arrow"), "wb") as sink:
            w = pa.ipc.new_file(sink, tbl(False).schema)
            w.write_table(tbl(False))
            w.close()

        eq(len(arrow_fragments(str(stream))), 1, "arrow: dir -> 1 fragment")

        got = list(adapter_arrow(str(stream), speaker_field="narrator"))
        eq(len(got), 2, "arrow stream: row count")
        eq(got[0]["text"], rows[0], "arrow stream: text")
        eq(got[0]["audio"], "c/0.wav", "arrow stream: audio path")
        eq(got[1]["speaker"], "spk_b", "arrow stream: speaker")
        eq(got[0]["duration"], 1.5, "arrow stream: duration")

        rooted = list(adapter_arrow(str(stream), audio_root="/mnt/wav"))
        eq(rooted[0]["audio"], "/mnt/wav/c/0.wav", "arrow: audio_root joined")

        # file encoding + audio with bytes but no path -> row locator
        bytesonly = list(adapter_arrow(str(Path(d) / "one.arrow")))
        eq(len(bytesonly), 2, "arrow file encoding: row count")
        check(bytesonly[1]["audio"].endswith("one.arrow#row=1"),
              "arrow: embedded audio gets a row locator",
              bytesonly[1]["audio"], "*one.arrow#row=1")

        # nested field access, and a bad field name must not pass silently
        nested = list(adapter_arrow(str(stream), audio_field="audio.path"))
        eq(nested[0]["audio"], "c/0.wav", "arrow: dotted --audio-field")
        try:
            list(adapter_arrow(str(stream), text_field="nope"))
            check(False, "arrow: unknown --text-field raises")
        except SystemExit as e:
            check("nope" in str(e) and "narrator" in str(e),
                  "arrow: unknown --text-field lists the real columns", str(e),
                  "message naming the available columns")


# ------------------------------------------------------------------ runner

def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        try:
            t()
        except Exception as e:
            FAILURES.append(f"{t.__name__} RAISED {type(e).__name__}: {e}")
    n = len(tests)
    if FAILURES:
        print(f"\n{len(FAILURES)} failure(s) across {n} test functions:\n")
        for f in FAILURES:
            print("  FAIL " + f)
        return 1
    print(f"all checks passed ({n} test functions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
