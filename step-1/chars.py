"""
Character constants and inventories.

Single source of truth for every codepoint the frontend cares about.
Nothing in this package should hardcode a Persian character inline.
"""

# ---------------------------------------------------------------- invisibles

ZWNJ = "\u200c"  # ZERO WIDTH NON-JOINER -- a real Persian morpheme boundary. KEEP.
ZWJ = "\u200d"
ZWSP = "\u200b"
LRM = "\u200e"
RLM = "\u200f"
LRE, RLE, PDF, LRO, RLO = "\u202a", "\u202b", "\u202c", "\u202d", "\u202e"
BOM = "\ufeff"
SOFT_HYPHEN = "\u00ad"
TATWEEL = "\u0640"  # ARABIC TATWEEL (kashida) -- purely decorative
NBSP = "\u00a0"
NARROW_NBSP = "\u202f"

# Removed outright. ZWNJ is deliberately absent from this set.
INVISIBLE_REMOVE = frozenset(
    [ZWJ, ZWSP, LRM, RLM, LRE, RLE, PDF, LRO, RLO, BOM, SOFT_HYPHEN, TATWEEL]
)

# Collapsed to a plain space.
SPACE_LIKE = frozenset([NBSP, NARROW_NBSP, "\u2000", "\u2001", "\u2002", "\u2003",
                        "\u2004", "\u2005", "\u2006", "\u2007", "\u2008", "\u2009",
                        "\u200a", "\u205f", "\u3000", "\t", "\v", "\f"])

# ---------------------------------------------------------------- diacritics

FATHATAN = "\u064b"  # an  (Arabic loans: واقعاً)
DAMMATAN = "\u064c"
KASRATAN = "\u064d"
FATHA = "\u064e"  # a
DAMMA = "\u064f"  # o
KASRA = "\u0650"  # e  (also ezafe when word-final)
SHADDA = "\u0651"  # gemination
SUKUN = "\u0652"  # no vowel
SUPERSCRIPT_ALEF = "\u0670"
MADDA_ABOVE = "\u0653"
HAMZA_ABOVE = "\u0654"
HAMZA_BELOW = "\u0655"

# The set we keep and train on.
HARAKAT = frozenset([FATHATAN, FATHA, DAMMA, KASRA, SHADDA, SUKUN])

# Arabic-only marks with no Persian phonological function: dropped.
HARAKAT_DROP = frozenset([DAMMATAN, KASRATAN, SUPERSCRIPT_ALEF, MADDA_ABOVE,
                          HAMZA_ABOVE, HAMZA_BELOW])

ALL_COMBINING = HARAKAT | HARAKAT_DROP

# ---------------------------------------------------------------- letters

PERSIAN_LETTERS = frozenset(
    "ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیآأإئؤءۀ"
)

# Letters that the VoxCPM tokenizer encodes via byte fallback (2 tokens each).
# Measured, not assumed -- see audit.py.
BYTE_FALLBACK_LETTERS = frozenset("پچژگ")

# ---------------------------------------------------------------- folding map

# Arabic / legacy / presentation forms -> canonical Persian.
FOLD_MAP = {
    "\u0643": "\u06a9",  # ARABIC KAF        ك -> ک
    "\u06aa": "\u06a9",  # SWASH KAF         ڪ -> ک
    "\u0649": "\u06cc",  # ALEF MAKSURA      ى -> ی
    "\u064a": "\u06cc",  # ARABIC YEH        ي -> ی
    "\u06d2": "\u06cc",  # YEH BARREE        ے -> ی
    "\u06d3": "\u06cc",  # YEH BARREE HAMZA  ۓ -> ی
    "\u0629": "\u0647",  # TEH MARBUTA       ة -> ه
    "\u06c0": "\u0647\u200c\u0627\u06cc",  # HEH WITH YEH ABOVE ۀ -> ه‌ای? no: keep simple
    "\u0671": "\u0627",  # ALEF WASLA        ٱ -> ا
    "\u0675": "\u0627",  # HIGH HAMZA ALEF
    "\u0676": "\u0648",
    "\u0677": "\u0648",
    "\u06a4": "\u0641",  # VEH ڤ -> ف
    "\u06a5": "\u0641",
    "\u06a7": "\u0642",
    "\u06a8": "\u0642",
    "\u0660": "0", "\u0661": "1", "\u0662": "2", "\u0663": "3", "\u0664": "4",
    "\u0665": "5", "\u0666": "6", "\u0667": "7", "\u0668": "8", "\u0669": "9",
    "\u06f0": "0", "\u06f1": "1", "\u06f2": "2", "\u06f3": "3", "\u06f4": "4",
    "\u06f5": "5", "\u06f6": "6", "\u06f7": "7", "\u06f8": "8", "\u06f9": "9",
    "\ufdfa": "صلی الله علیه و آله و سلم",
    "\ufdfb": "جل جلاله",
    "\ufdf2": "الله",
    "\ufdfd": "بسم الله الرحمن الرحیم",
}

# ۀ is genuinely ambiguous; treat as heh + hamza-ish and let the diacritizer
# handle it. Folding it to a phrase was wrong -- override.
FOLD_MAP["\u06c0"] = "\u0647"

# Arabic presentation forms (U+FB50-U+FEFF) are handled generically by NFKC-lite
# in normalize.py rather than enumerated here.

# ---------------------------------------------------------------- punctuation

PERSIAN_COMMA = "\u060c"  # ،
PERSIAN_SEMICOLON = "\u061b"  # ؛
PERSIAN_QUESTION = "\u061f"  # ؟
PERSIAN_PERCENT = "\u066a"  # ٪
PERSIAN_DECIMAL = "\u066b"  # ٫
PERSIAN_THOUSANDS = "\u066c"  # ٬
ARABIC_FULL_STOP = "\u06d4"  # ۔
GUILLEMET_OPEN = "\u00ab"  # «
GUILLEMET_CLOSE = "\u00bb"  # »
ELLIPSIS = "\u2026"  # …

PUNCT_MAP = {
    ",": PERSIAN_COMMA,
    "\uff0c": PERSIAN_COMMA,
    "\u3001": PERSIAN_COMMA,
    ";": PERSIAN_SEMICOLON,
    "\uff1b": PERSIAN_SEMICOLON,
    "?": PERSIAN_QUESTION,
    "\uff1f": PERSIAN_QUESTION,
    ARABIC_FULL_STOP: ".",
    "\u3002": ".",
    "\uff0e": ".",
    "\uff01": "!",
    "\uff1a": ":",
    "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-",
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": GUILLEMET_OPEN, "\u201d": GUILLEMET_CLOSE,
    "\u201e": GUILLEMET_OPEN, "\u201f": GUILLEMET_CLOSE,
    "\u2039": GUILLEMET_OPEN, "\u203a": GUILLEMET_CLOSE,
    "`": "'", "\u00b4": "'",
    "\u2022": "", "\u25cf": "", "\u25aa": "",  # bullets
    "\u066d": "*",
}

# Punctuation the model is allowed to see. Everything else is stripped or
# rewritten. These are prosody signals -- do not extend casually.
KEPT_PUNCT = frozenset([".", "!", ":", "-", "'",
                        PERSIAN_COMMA, PERSIAN_SEMICOLON, PERSIAN_QUESTION,
                        GUILLEMET_OPEN, GUILLEMET_CLOSE, ELLIPSIS])

# ---------------------------------------------------------------- charset guard

def allowed_charset(keep_harakat: bool = True) -> frozenset:
    """The complete inventory of characters permitted in normalized output.

    Anything outside this set surviving normalization is a bug or an exotic
    input, and `Normalizer` reports it. This guard catches more real problems
    than any other single check in the pipeline.
    """
    s = set(PERSIAN_LETTERS) | set(KEPT_PUNCT) | {" ", ZWNJ}
    if keep_harakat:
        s |= set(HARAKAT)
    return frozenset(s)
