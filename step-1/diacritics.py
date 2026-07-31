"""
Harakat utilities.

The masking machinery here is what makes an imperfect runtime diacritizer
survivable: the model is trained across the whole spectrum from bare text to
fully marked text, so it learns to treat marks as soft hints rather than hard
constraints. A wrong mark then degrades output gracefully instead of dictating
a wrong pronunciation.

`MaskingSchedule` implements the Stage A curriculum and `sample_density` the
Stage B density mix. Both are meant to be called from the dataloader, per
sample, per epoch -- not applied once and baked into the manifest.
"""

import random
import re
import unicodedata

from .chars import HARAKAT, KASRA, SHADDA, ZWNJ

_HARAKAT_STR = "".join(sorted(HARAKAT))
HARAKAT_RE = re.compile(f"[{_HARAKAT_STR}]")
_STRIP_TABLE = str.maketrans({c: None for c in HARAKAT})

WORD_RE = re.compile(rf"[\u0600-\u06ff{ZWNJ}]+")


def strip(text: str) -> str:
    """Remove every harakat, leaving bare orthography."""
    return text.translate(_STRIP_TABLE)


def has_marks(word: str) -> bool:
    return bool(HARAKAT_RE.search(word))


def count_marks(text: str) -> int:
    return len(HARAKAT_RE.findall(text))


def word_density(text: str) -> float:
    """Fraction of Persian words carrying at least one mark.

    This is the number to report per-corpus and to target in the masking
    schedule -- character-level mark ratio is a much less useful statistic.
    """
    words = WORD_RE.findall(text)
    if not words:
        return 0.0
    return sum(1 for w in words if has_marks(w)) / len(words)


def ends_with_ezafe(word: str) -> bool:
    """A word-final kasra is (almost always) the ezafe construction rather
    than a lexical short vowel. Worth treating as a distinct, high-priority
    signal -- ezafe is the most audible prosodic error in Persian TTS."""
    return word.endswith(KASRA)


# ------------------------------------------------------------------ masking

def mask(text: str,
         keep_rate: float,
         rng: random.Random = None,
         priority: set = None,
         priority_boost: float = 0.45,
         always_keep_ezafe: bool = True) -> str:
    """Drop marks from a random subset of words.

    keep_rate       base probability that a marked word keeps its marks
    priority        bare (unmarked) word forms to preserve preferentially --
                    pass your homograph list here
    priority_boost  added to keep_rate for priority words, clamped to 1.0
    always_keep_ezafe
                    word-final kasra is never dropped

    Masking is at WORD granularity, not character. Partially-marked words are
    an unnatural input the frontend will never produce, so training on them
    teaches nothing useful.
    """
    rng = rng or random
    priority = priority or set()

    def sub(m):
        w = m.group(0)
        if not has_marks(w):
            return w
        bare = strip(w)
        p = keep_rate
        if bare in priority:
            p = min(1.0, keep_rate + priority_boost)
        if rng.random() < p:
            return w
        if always_keep_ezafe and ends_with_ezafe(w):
            return bare + KASRA
        return bare

    return WORD_RE.sub(sub, text)


class MaskingSchedule:
    """Stage A diacritic curriculum.

    Phase 1 (warm, first `warm_frac` of training): keep_rate ~0.9 so the
      grapheme->phoneme mapping forms against a clean signal.
    Phase 2 (anneal): keep_rate lower bound walks down toward 0.
    Phase 3 (match): keep_rate ~ U(0, 1), matching the range of densities the
      production frontend can produce.

    Reversing this order measurably hurts: the model spends early training
    guessing and builds a worse mapping.
    """

    def __init__(self, total_steps: int, warm_frac: float = 0.20,
                 anneal_frac: float = 0.30, warm_keep: float = 0.9,
                 seed: int = 1234):
        self.total = max(1, total_steps)
        self.warm_end = warm_frac
        self.anneal_end = warm_frac + anneal_frac
        self.warm_keep = warm_keep
        self.rng = random.Random(seed)

    def keep_rate(self, step: int) -> float:
        f = min(1.0, step / self.total)
        if f < self.warm_end:
            return self.warm_keep
        if f < self.anneal_end:
            t = (f - self.warm_end) / max(1e-9, self.anneal_end - self.warm_end)
            lo = self.warm_keep * (1 - t) + 0.1 * t
            return self.rng.uniform(lo, self.warm_keep)
        return self.rng.uniform(0.0, 1.0)

    def apply(self, text: str, step: int, priority: set = None) -> str:
        return mask(text, self.keep_rate(step), rng=self.rng, priority=priority)


#: Stage B density mix. Gemini gives fully-marked ground truth; sampling from
#: it keeps the labels correct while matching the sparse distribution the
#: frontend actually emits at inference. Training Stage B at 100% marking is
#: the mistake this table exists to prevent.
STAGE_B_DENSITY_MIX = (
    (0.60, (0.10, 0.25)),   # sparse  -- matches production
    (0.25, (0.40, 0.60)),   # medium
    (0.15, (1.00, 1.00)),   # full    -- so full marking remains usable
)


def sample_density(rng: random.Random = None) -> float:
    rng = rng or random
    r = rng.random()
    acc = 0.0
    for share, (lo, hi) in STAGE_B_DENSITY_MIX:
        acc += share
        if r <= acc:
            return rng.uniform(lo, hi)
    return 1.0


# ------------------------------------------------------------------ validation

def validate(text: str):
    """Structural checks on diacritic placement. Returns a list of problem
    codes; empty means clean.

    These catch real corpus bugs -- marks orphaned by a bad segmentation,
    double vowels from merging two diacritizer outputs, marks landing on
    punctuation after a botched regex.
    """
    problems = []
    prev = ""
    for i, ch in enumerate(text):
        if ch in HARAKAT:
            if not prev:
                problems.append("mark_at_start")
            elif not ("\u0600" <= prev <= "\u06ff") or prev in HARAKAT and ch != SHADDA:
                if prev in HARAKAT:
                    if not (prev == SHADDA or ch == SHADDA):
                        problems.append(f"double_mark@{i}")
                else:
                    problems.append(f"mark_on_nonletter@{i}")
            if prev == ZWNJ:
                problems.append(f"mark_after_zwnj@{i}")
        prev = ch
    if text != unicodedata.normalize("NFC", text):
        problems.append("not_nfc")
    return problems


def report(texts):
    """Corpus-level diacritic statistics."""
    n = len(texts)
    if not n:
        return {}
    densities = [word_density(t) for t in texts]
    marked = sum(1 for d in densities if d > 0)
    problems = {}
    for t in texts:
        for p in validate(t):
            key = p.split("@")[0]
            problems[key] = problems.get(key, 0) + 1
    densities_sorted = sorted(densities)
    return {
        "n": n,
        "samples_with_any_mark": marked,
        "mean_word_density": sum(densities) / n,
        "median_word_density": densities_sorted[n // 2],
        "p90_word_density": densities_sorted[int(n * 0.9)],
        "total_marks": sum(count_marks(t) for t in texts),
        "problems": problems,
    }
