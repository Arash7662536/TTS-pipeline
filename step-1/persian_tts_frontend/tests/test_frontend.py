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


def test_signed_numbers():
    """A sign glued to a number is read. Both halves used to be wrong: `+` was
    dropped by the charset guard, `-` was rewritten to a comma, so a negative
    temperature was voiced as a positive one."""
    nz = Normalizer()
    eq(nz("دمای هوا -۵ درجه بود"), "دمای هوا منفی پنج درجه بود", "negative int")
    eq(nz("به +11 درجه رسید"), "به مثبت یازده درجه رسید", "positive int")
    check("منفی دوازده ممیز پنج" in nz("اختلاف -۱۲٫۵ درجه"),
          "signed decimal", nz("اختلاف -۱۲٫۵ درجه"), "منفی دوازده ممیز پنج")
    check(not nz.normalize("قاره +16 و +11 درجه").unexpected,
          "no stray sign reaches the charset guard")
    # ... but a dash that is not a sign must keep its old meaning
    check("تا" in nz("بین ۱۲-۱۵ نفر"), "digit-dash-digit stays a range",
          nz("بین ۱۲-۱۵ نفر"), "has تا")
    check("منفی" not in nz("- ۵ نکته مهم"), "spaced dash is a list marker",
          nz("- ۵ نکته مهم"), "no منفی")
    check("ژانویه" in nz("در تاریخ 2024-01-15 منتشر شد"),
          "dashed date still a date", nz("در تاریخ 2024-01-15"), "has ژانویه")


def test_thousands_separator():
    """The punctuation pass turns "," into "،" before expansion sees it, so the
    grouping rule has to recognise the Persian comma too -- otherwise "1,250"
    reads as "one, two hundred fifty"."""
    nz = Normalizer()
    eq(nz("بدهی 1,250 دلار"), "بدهی هزار و دویست و پنجاه دلار", "ASCII grouping")
    eq(nz("قیمت ۱٬۲۵۰ تومان"), "قیمت هزار و دویست و پنجاه تومان",
       "Persian grouping mark")
    check("یک میلیون و دویست و پنجاه هزار" in nz("جمعیت 1,250,000 نفر"),
          "multi-group", nz("جمعیت 1,250,000 نفر"), "یک میلیون و ...")
    # A spaced comma is an enumeration, not a grouping mark.
    check("دویست و پنجاه" in nz("بندهای 1, 250 را ببین")
          and "هزار" not in nz("بندهای 1, 250 را ببین"),
          "spaced comma stays an enumeration", nz("بندهای 1, 250 را ببین"),
          "یک، دویست و پنجاه")


def test_abbreviation_guard_covers_zwnj():
    """ZWNJ is U+200C, outside the Arabic block, so it has to be named in the
    abbreviation boundary guard explicitly. Colloquial clitics attach with one,
    and at the end of a sentence "خونه‌م." was being read as "خونه میلادی"."""
    nz = Normalizer()
    for s in ["خونه‌م.", "بچه‌م.", "خونه‌ش.", "مسئول‌ش."]:
        eq(nz(s), s, f"ZWNJ clitic untouched: {s}")
    check("میلادی" not in nz("این حرفِ منه، خونه‌م. بعد رفتیم"),
          "clitic mid-sentence", nz("این حرفِ منه، خونه‌م. بعد رفتیم"),
          "no میلادی")
    # the guard must not have been loosened into uselessness
    eq(nz("ص. ۱۲ را ببین"), "صفحه دوازده را ببین", "real abbreviation still expands")
    eq(nz("ج. ۳ منتشر شد"), "جلد سه منتشر شد", "real abbreviation still expands")
    check("قبل از میلاد" in nz("در ۳۳۰ ق.م"), "multi-part abbreviation",
          nz("در ۳۳۰ ق.م"), "قبل از میلاد")
    eq(nz("گرفتیم."), "گرفتیم.", "plain letter boundary still guarded")


def test_idempotence_regressions():
    """Each of these broke `nz(nz(x)) == nz(x)` on the thomclas corpus."""
    nz = Normalizer()
    cases = {
        "srt": "استفاده کرده 130 00:13:30,686 --> 00:13:30,666 میشه اینطور",
        "colon_zwnj": "یک:‌ توانایی یادگیری‌مون داره می‌آد پایین",
        "double_dot": "ارتشی و نظامی هم که مسئول‌ش .. این‌که من در واقع",
    }
    for label, s in cases.items():
        once = nz(s)
        eq(nz(once), once, f"idempotent: {label}")
    check("توانایی" in nz(cases["colon_zwnj"])
          and "یک: توانایی" in nz(cases["colon_zwnj"]),
          "ZWNJ after a colon does not glue the next word",
          nz(cases["colon_zwnj"]), "یک: توانایی ...")


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
    # pipeline.py holds the rule order and the charset-guard policy, so it has
    # to be inside the hash or two frontends that disagree on the output text
    # can advertise the same version.
    import inspect as _i
    from persian_tts_frontend import pipeline as _p
    src = _i.getsource(_p)
    check("sys.modules[__name__]" in src, "version hash covers pipeline.py")


def test_zwnj_suffix_does_not_chain():
    """A stuttered "شبکه ها ها" joined one "ها" per pass, so the text was a
    different string every time it went through -- found on the full corpus."""
    nz = Normalizer()
    once = nz("روی سایر شبکه ها ها نیست")
    eq(nz(once), once, "doubled suffix is idempotent")
    eq(once, "روی سایر شبکه‌ها ها نیست", "second ها stays detached")
    # the ordinary repairs must be untouched, including stems holding a ZWNJ
    eq(nz("کتاب ها را بردم"), "کتاب‌ها را بردم", "plain plural still joins")
    eq(nz("بزرگ ترین شهر"), "بزرگ‌ترین شهر", "superlative still joins")
    eq(nz("خانه ام را فروختم"), "خانه‌ام را فروختم", "possessive still joins")
    check(ZWNJ in nz("نرم‌افزار ها را نصب کن"),
          "stem containing a ZWNJ still takes its suffix",
          nz("نرم‌افزار ها را نصب کن"), "joined")


def test_unknown_chars_do_not_weld_words():
    """The charset guard replaces with a space, not "" -- deleting outright
    turned "۲×۳" into the fabricated word "دوسه"."""
    nz = Normalizer()
    eq(nz("و/یا این"), "و یا این", "slash does not weld")
    check(" " in nz("۲×۳ برابر شش"), "multiplication sign does not weld",
          nz("۲×۳ برابر شش"), "دو سه ...")
    # a stripped char already beside whitespace must not add a second space
    eq(nz("آهنگ ♫ پخش شد"), "آهنگ پخش شد", "no doubled space")
    eq(nz("کتاب دنیل بل* را"), "کتاب دنیل بل را", "trailing marker")


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


def test_field_bounds_and_quantiles():
    from persian_tts_frontend.cli import _parse_bounds, _quantiles, _as_number

    eq(_parse_bounds(["mos_ovr=3.5"], []), {"mos_ovr": (3.5, None)}, "min only")
    eq(_parse_bounds([], ["wer=0.2"]), {"wer": (None, 0.2)}, "max only")
    eq(_parse_bounds(["a=1"], ["a=9"]), {"a": (1.0, 9.0)}, "both ends of one field")
    for bad in ["mos_ovr", "mos_ovr=high", "=3"]:
        try:
            _parse_bounds([bad], [])
            check(False, f"bad bound rejected: {bad}")
        except SystemExit:
            pass
    # a numeric string is a number; a bool is not a score
    eq(_as_number("3.5"), 3.5, "numeric string")
    eq(_as_number(True), None, "bool is not a score")
    eq(_as_number(None), None, "null is not a score")

    q = _quantiles([1.0, 2.0, 3.0, 4.0, 5.0])
    eq(q["n"], 5, "quantile count")
    eq(q["min"], 1.0, "quantile min")
    eq(q["max"], 5.0, "quantile max")
    eq(q["median"], 3.0, "quantile median")
    eq(_quantiles([2.0])["median"], 2.0, "single value does not index past end")


def test_script_boundary_separation():
    """Transcripts drop the space between scripts constantly. Every rule
    substitutes in place, so "60تا" was expanding into the welded "شصتتا"."""
    nz = Normalizer(config=NormalizerConfig(latin_strategy="transliterate"))
    eq(nz("60تا تاجر"), "شصت تا تاجر", "digit welded to Persian")
    eq(nz("حدود 33ساله شده"), "حدود سی و سه ساله شده", "digit before a suffix")
    eq(nz("ما Great Depressionایم"), "ما گریت دیپرشن ایم", "Latin welded to Persian")
    # a digit expanded mid-token creates a fresh boundary: GTA6 -> GTAشش
    eq(nz("تریلر GTA6منتشر شد"), "تریلر جی‌تی‌ای شش منتشر شد",
       "boundary created by expansion")
    # Persian punctuation lives in the same Unicode block as the letters and
    # must NOT count as script, or a thousands separator gets split apart
    eq(nz("جمعیت 1,250,000 نفر"), "جمعیت یک میلیون و دویست و پنجاه هزار نفر",
       "thousands separator survives")
    eq(nz("کتاب ۵ام را خواند"), "کتاب پنجم را خواند", "ordinal still joins")
    eq(nz("سرعت ۸۰ کیلومتر بود"), "سرعت هشتاد کیلومتر بود", "units unaffected")


def test_latin_lexicon_and_single_letters():
    """The corpus head must resolve to real Persian, not to the crude
    transliterator -- "the" transliterates to the single letter "ت"."""
    nz = Normalizer(config=NormalizerConfig(latin_strategy="transliterate"))
    eq(nz("کتاب the great depression را خواند"),
       "کتاب د گریت دیپرشن را خواند", "frequent English words come from lexicon")
    # a lone Latin letter is a letter name, never a transliterated consonant
    eq(nz("ویتامین d بخور"), "ویتامین دی بخور", "single letter reads as a name")
    eq(nz("نقطه a تا b"), "نقطه ای تا بی", "single letters in sequence")
    # acronyms still spell out, and the long tail still transliterates
    check("دی‌ان‌ای" in nz("آزمایش DNA داد"), "acronym spelled out",
          nz("آزمایش DNA داد"), "دی‌ان‌ای")
    out = nz("شرکت zqxwv")
    check("{" not in out and any("؀" <= c <= "ۿ" for c in out),
          "unknown word transliterates rather than escaping", out, "Persian")
    # ... but under the default strategy it must still escalate, not invent
    nz2 = Normalizer()
    check("{zqxwv}" in nz2("شرکت zqxwv"), "escalate is still the safe default",
          nz2("شرکت zqxwv"), "{zqxwv}")


def test_arrow_manifest_writer():
    try:
        import pyarrow as pa
    except ImportError:
        return
    import tempfile
    from persian_tts_frontend.cli import ArrowManifestWriter

    with tempfile.TemporaryDirectory() as d:
        p = str(Path(d) / "m.arrow")
        w = ArrowManifestWriter(p)
        w.write({"audio": "a.wav", "text": "یک", "duration": 1.5})
        w.write({"audio": "b.wav", "text": "دو", "mos_bak": 4.0})  # ragged keys
        n, cols = w.close()
        eq(n, 2, "arrow writer row count")
        check("mos_bak" in cols and "duration" in cols,
              "schema is the union of all row keys", cols, "both columns")
        with pa.memory_map(p, "rb") as src:
            back = pa.ipc.open_stream(src).read_all().to_pylist()
        eq(len(back), 2, "round-trips")
        eq(back[0]["text"], "یک", "round-trips text")
        eq(back[1]["duration"], None, "missing key becomes null")


def test_build_dataset_preserves_audio():
    """`build` reproduces the corpus rather than pointing at it: the audio must
    come out byte-identical while the transcript is replaced."""
    try:
        import pyarrow as pa
    except ImportError:
        return
    import argparse
    import hashlib
    import io
    import json
    import tempfile
    import wave
    from persian_tts_frontend.cli import run_build

    def wav(sec, fill):
        b = io.BytesIO()
        with wave.open(b, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(fill * int(16000 * sec))
        return b.getvalue()

    rows = [("در سال ۱۴۰۳ رشد داشت", 5.0, b"\x11\x22", 4.2),
            ("این کیفیت پایین است", 4.0, b"\x33\x44", 2.1)]
    with tempfile.TemporaryDirectory() as d:
        src, out = str(Path(d) / "src"), str(Path(d) / "out")
        Path(src).mkdir()
        tbl = pa.table({
            "sentence": pa.array([r[0] for r in rows]),
            "audio": pa.array([{"bytes": wav(r[1], r[2]), "path": None}
                               for r in rows],
                              type=pa.struct([("bytes", pa.binary()),
                                              ("path", pa.string())])),
            "mos_bak": pa.array([r[3] for r in rows], type=pa.float64()),
        })
        with pa.OSFile(str(Path(src) / "data-00000-of-00001.arrow"), "wb") as s:
            w = pa.ipc.new_stream(s, tbl.schema)
            w.write_table(tbl)
            w.close()

        args = argparse.Namespace(
            input=src, output_dataset=out, report=None, split=None,
            text_field="sentence", audio_field="audio",
            duration_field="duration", duration_from_audio=True,
            sampling_rate=16000, shard_mb=500, keep_fields="mos_bak",
            min_field=["mos_bak=3.0"], max_field=[], drop_text_matching=None,
            min_duration=0.0, max_duration=0.0, min_words=2, max_chars=600,
            dataset_id=1, tier="core", strip_harakat=False, keep_parens=False,
            decimal_style="momayez", latin_strategy="escalate")
        run_build(args)

        with pa.memory_map(str(Path(out) / "data-00000.arrow"), "rb") as s:
            t = pa.ipc.open_stream(s).read_all()
        eq(t.num_rows, 1, "gate applied during build")
        got = t.column("audio").to_pylist()[0]["bytes"]
        eq(hashlib.md5(got).hexdigest(),
           hashlib.md5(wav(5.0, b"\x11\x22")).hexdigest(),
           "audio bytes survive byte-identical")
        eq(t.column("text").to_pylist()[0], "در سال هزار و چهارصد و سه رشد داشت",
           "text is the normalized form")
        eq(t.column("duration").to_pylist()[0], 5.0, "duration derived")
        feats = json.loads(t.schema.metadata[b"huggingface"])["info"]["features"]
        eq(feats["audio"]["_type"], "Audio", "audio stays an Audio feature")
        eq(feats["audio"]["sampling_rate"], 16000, "sampling rate declared")
        # the save_to_disk sidecars datasets.load_from_disk needs
        info = json.load(open(str(Path(out) / "dataset_info.json")))
        state = json.load(open(str(Path(out) / "state.json")))
        eq(info["splits"]["train"]["num_examples"], 1, "dataset_info row count")
        eq(len(state["_data_files"]), 1, "state lists the shard")


def test_wav_duration_from_header():
    """Datasets with embedded audio and no duration column: read it from the
    WAV header, without touching the payload."""
    import io
    import wave
    from persian_tts_frontend.cli import _wav_duration

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000 * 7)
    canonical = buf.getvalue()
    eq(_wav_duration(canonical[:128]), 7.0, "canonical 16k mono header")

    # an interleaved LIST chunk moves `data`; a fixed 44-byte guess mis-times it
    patched = (canonical[:36] + b"LIST" + (10).to_bytes(4, "little")
               + b"INFOxxxxxx" + canonical[36:])
    patched = b"RIFF" + (len(patched) - 8).to_bytes(4, "little") + patched[8:]
    eq(_wav_duration(patched[:128]), 7.0, "header with a LIST chunk")

    for junk in [b"", b"ID3\x04junkjunkjunk", b"RIFF" + b"\x00" * 8]:
        eq(_wav_duration(junk), None, "non-WAV yields no duration")


def test_arrow_gating():
    """The thomclas shape: text + embedded audio + per-clip DNSMOS columns."""
    try:
        import pyarrow as pa
    except ImportError:
        return
    import tempfile
    from persian_tts_frontend.cli import adapter_arrow

    with tempfile.TemporaryDirectory() as d:
        tbl = pa.table({
            "sentence": pa.array(["جمله اول است", "جمله دوم است"]),
            "audio": pa.array([{"bytes": b"RIFF", "path": None}] * 2,
                              type=pa.struct([("bytes", pa.binary()),
                                              ("path", pa.string())])),
            "mos_ovr": pa.array([4.25, 1.75], type=pa.float64()),
        })
        with pa.OSFile(str(Path(d) / "data-00000-of-00001.arrow"), "wb") as sink:
            w = pa.ipc.new_stream(sink, tbl.schema)
            w.write_table(tbl)
            w.close()

        rows = list(adapter_arrow(d, text_field="sentence",
                                  extra_fields=["mos_ovr"]))
        eq(len(rows), 2, "gated adapter: row count")
        eq(rows[0]["_extra"]["mos_ovr"], 4.25, "extra column reaches _extra")
        eq(rows[1]["_extra"]["mos_ovr"], 1.75, "extra column per row")
        # a column that is not there must not crash the pass
        rows2 = list(adapter_arrow(d, text_field="sentence",
                                   extra_fields=["nope"]))
        eq(rows2[0]["_extra"], {}, "absent extra column yields no key")

        # Row locators must restart per shard -- they name a file and an index
        # into *that* file, so a cumulative counter points at the wrong row.
        second = Path(d) / "data-00001-of-00002.arrow"
        with pa.OSFile(str(second), "wb") as sink:
            w = pa.ipc.new_stream(sink, tbl.schema)
            w.write_table(tbl)
            w.close()
        locators = [r["audio"] for r in adapter_arrow(d, text_field="sentence")]
        eq(len(locators), 4, "two shards read")
        check(locators[2].endswith("#row=0") and locators[3].endswith("#row=1"),
              "locator index restarts on the second shard",
              locators[2:], "...#row=0, ...#row=1")


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
