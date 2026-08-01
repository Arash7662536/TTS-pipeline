"""
Unicode-level normalization.

A note on combining-mark order, because it is easy to get wrong:

    Arabic harakat all carry distinct nonzero canonical combining classes
    (fathatan 27 ... sukun 34), so NFC *already* reorders them into a
    deterministic canonical sequence. That canonical order puts the vowel
    BEFORE shadda -- the opposite of typing convention.

    Do not hand-roll a reorder pass. Just apply NFC everywhere, consistently,
    and assert idempotence. Consistency is what matters to the tokenizer; the
    particular order does not.
"""

import re
import unicodedata

from .chars import (ALL_COMBINING, ELLIPSIS, FOLD_MAP, GUILLEMET_CLOSE,
                    GUILLEMET_OPEN, HARAKAT, HARAKAT_DROP, INVISIBLE_REMOVE,
                    KEPT_PUNCT, PERSIAN_COMMA, PUNCT_MAP, SPACE_LIKE, ZWNJ)

# --------------------------------------------------------- phoneme escape guard

ESCAPE_RE = re.compile(r"\{([^{}]{1,64})\}")

# Placeholders live in the Private Use Area, one codepoint per escape.
#
# An earlier version encoded the index as ASCII digits between control
# characters -- and the integer expander duly read "0" and wrote "صفر" into the
# middle of the placeholder. Digits, Latin letters and Persian letters are all
# reachable by some rule in this pipeline; PUA codepoints are reachable by none.
SENTINEL_BASE = 0xE000
SENTINEL_MAX = 0xF8FF
SENTINEL_RANGE = "\ue000-\uf8ff"
SENTINEL_RE = re.compile(f"[{SENTINEL_RANGE}]")


def protect_escapes(text: str, payloads=None):
    """Pull `{phoneme}` spans out of the text before any rule touches them.

    Must run before any other rule and be undone last, otherwise IPA characters
    get folded, transliterated, or stripped by the charset guard.

    Re-entrant: pass an existing `payloads` list to protect escapes created
    mid-pipeline (the Latin resolver emits them) without disturbing earlier
    placeholders.
    """
    payloads = [] if payloads is None else payloads

    def sub(m):
        idx = SENTINEL_BASE + len(payloads)
        if idx > SENTINEL_MAX:          # absurd input; leave it alone
            return m.group(0)
        payloads.append(m.group(1))
        return chr(idx)

    return ESCAPE_RE.sub(sub, text), payloads


def restore_escapes(text: str, payloads) -> str:
    def sub(m):
        i = ord(m.group(0)) - SENTINEL_BASE
        return "{" + payloads[i] + "}" if 0 <= i < len(payloads) else ""
    return SENTINEL_RE.sub(sub, text)


# --------------------------------------------------------------- folding passes

_FOLD_TABLE = str.maketrans(FOLD_MAP)
_PUNCT_TABLE = str.maketrans(PUNCT_MAP)
_INVISIBLE_TABLE = str.maketrans({c: None for c in INVISIBLE_REMOVE})
_SPACE_TABLE = str.maketrans({c: " " for c in SPACE_LIKE})
_DROP_MARKS_TABLE = str.maketrans({c: None for c in HARAKAT_DROP})


def decompose_presentation_forms(text: str) -> str:
    """Collapse Arabic presentation forms (U+FB50-U+FEFF isolated/initial/
    medial/final glyph variants) back to base letters.

    NFKC does this correctly for Arabic script, but NFKC also mangles things we
    care about elsewhere, so it is applied only to the presentation-form
    ranges rather than to the whole string.
    """
    out = []
    for ch in text:
        if "\ufb50" <= ch <= "\ufeff" or "\ufe70" <= ch <= "\ufefc":
            k = unicodedata.normalize("NFKC", ch)
            out.append(k if k else ch)
        else:
            out.append(ch)
    return "".join(out)


def fold_codepoints(text: str) -> str:
    """Arabic / legacy / variant codepoints -> canonical Persian, and all
    Arabic-Indic and Extended Arabic-Indic digits -> ASCII for downstream
    numeric processing."""
    return text.translate(_FOLD_TABLE)


def remove_invisibles(text: str) -> str:
    """Drop bidi controls, ZWJ, ZWSP, tatweel, soft hyphen, BOM.
    ZWNJ (U+200C) is deliberately preserved -- it is a morpheme boundary and
    the VoxCPM tokenizer encodes it as a single clean token."""
    text = text.translate(_INVISIBLE_TABLE)
    return text.translate(_SPACE_TABLE)


def drop_non_persian_marks(text: str) -> str:
    """Remove Arabic-only diacritics with no Persian phonological function
    (dammatan, kasratan, superscript alef, madda/hamza combining forms)."""
    return text.translate(_DROP_MARKS_TABLE)


STRAIGHT_QUOTE_PAIR_RE = re.compile(r'"([^"\n]{1,300})"')


def normalize_punctuation(text: str, parens_to_commas: bool = True) -> str:
    # Paired straight quotes -> guillemets. Done before the char table so the
    # pairing information is still available; unpaired leftovers are dropped by
    # the charset guard rather than guessed at.
    text = STRAIGHT_QUOTE_PAIR_RE.sub(
        lambda m: GUILLEMET_OPEN + m.group(1) + GUILLEMET_CLOSE, text)
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\.{3,}", ELLIPSIS, text)
    text = re.sub(rf"{ELLIPSIS}{{2,}}", ELLIPSIS, text)
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "\u061f", text)
    text = re.sub(r"([!\u061f])\1*", r"\1", text)
    if parens_to_commas:
        # Parenthetical asides are prosodically commas in speech.
        text = re.sub(r"\s*[\(\[\{]\s*", PERSIAN_COMMA + " ", text)
        text = re.sub(r"\s*[\)\]\}]\s*", PERSIAN_COMMA + " ", text)
    text = re.sub(rf"\s*{GUILLEMET_OPEN}\s*", " " + GUILLEMET_OPEN, text)
    text = re.sub(rf"\s*{GUILLEMET_CLOSE}\s*", GUILLEMET_CLOSE + " ", text)
    return text


def fix_spacing(text: str) -> str:
    """Whitespace and punctuation adjacency.

    Also repairs the single most common real-world Persian typing error: a
    space where a ZWNJ belongs, in the `می`/`نمی` verbal prefixes and the
    `ها`/`تر`/`ترین` suffixes.
    """
    text = re.sub(r"\s+", " ", text)
    # Nothing adjacent to ZWNJ but letters. This must precede the punctuation
    # spacing below: to the `(?=\S)` rule a ZWNJ is a non-space, so in
    # "یک:‌ توانایی" the space it inserts lands in front of a ZWNJ that the next
    # rule deletes -- leaving "یک:توانایی" glued until a second pass fixes it.
    text = re.sub(rf"\s+{ZWNJ}", ZWNJ, text)
    text = re.sub(rf"{ZWNJ}\s+", ZWNJ, text)
    text = re.sub(rf"{ZWNJ}{{2,}}", ZWNJ, text)
    text = re.sub(rf"{ZWNJ}(?![؀-ۿ])", "", text)
    text = re.sub(rf"(?<![؀-ۿ]){ZWNJ}", "", text)
    # No space before closing punctuation. Hyphen and apostrophe are excluded:
    # they are intra-word characters here, and eating the space before a stray
    # dash produces "است- یعنی" instead of a clean pause.
    _closing = KEPT_PUNCT - {GUILLEMET_OPEN, "-", "'"}
    text = re.sub(rf"\s+([{re.escape(''.join(_closing))}])", r"\1", text)
    # exactly one space after punctuation
    text = re.sub(rf"([{re.escape('.!:' + chr(0x60c) + chr(0x61b) + chr(0x61f))}])"
                  r"(?=\S)", r"\1 ", text)
    return text.strip()


_TERM = "." + "!" + "\u061f" + ELLIPSIS
_PAUSE = "\u060c\u061b"


def cleanup(text: str) -> str:
    """Post-expansion tidy-up.

    Runs after numeric expansion, so it can safely rewrite dashes that the
    range expander needed to see intact. Fixes the artifacts that expansion
    itself creates -- a comma left dangling before a full stop when a
    parenthetical ended a sentence, doubled pause marks from adjacent rules,
    a stray dash left where an em dash used to be.
    """
    # A space-surrounded dash is a prosodic break, not a hyphen. By this point
    # all numeric ranges have already been expanded to "تا".
    text = re.sub(r"\s*[-\u2013\u2014]\s+", "\u060c ", text)
    text = re.sub(r"\s+[-\u2013\u2014]\s*", "\u060c ", text)
    # collapse runs of pause marks
    text = re.sub(rf"([{_PAUSE}])[\s{_PAUSE}]*([{_PAUSE}])", r"\2", text)
    # pause mark immediately before a terminal loses to the terminal
    text = re.sub(rf"[{_PAUSE}]\s*(?=[{re.escape(_TERM)}])", "", text)
    text = re.sub(rf"([{re.escape(_TERM)}])\s*[{_PAUSE}]", r"\1", text)
    # collapse runs of terminals (keep the first)
    text = re.sub(rf"([{re.escape(_TERM)}])[\s{re.escape(_TERM)}]*"
                  rf"([{re.escape(_TERM)}])", r"\2", text)
    # nothing but punctuation at the very start
    text = re.sub(rf"^[{_PAUSE}{re.escape(_TERM)}\s]+", "", text)
    return text


ZWNJ_PREFIX_RE = re.compile(r"\b(ن?می) ([\u0600-\u06ff]{2,})")
ZWNJ_SUFFIX_RE = re.compile(r"([\u0600-\u06ff]{2,}) (ها|های|هایی|تر|ترین|ام|ات|اش)\b")


def repair_zwnj(text: str) -> str:
    text = ZWNJ_PREFIX_RE.sub(rf"\1{ZWNJ}\2", text)
    text = ZWNJ_SUFFIX_RE.sub(rf"\1{ZWNJ}\2", text)
    return text


# ------------------------------------------------------------------ NFC + guard

def to_nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def assert_nfc_idempotent(text: str) -> bool:
    once = unicodedata.normalize("NFC", text)
    return once == unicodedata.normalize("NFC", once)


def mark_order_signature(text: str) -> str:
    """Debug helper: the sequence of combining classes attached to each base
    letter. Two strings with the same signature will tokenize identically with
    respect to diacritics."""
    sig = []
    for ch in text:
        if ch in ALL_COMBINING:
            sig.append(str(unicodedata.combining(ch)))
        elif sig and sig[-1] != "|":
            sig.append("|")
    return "".join(sig)
