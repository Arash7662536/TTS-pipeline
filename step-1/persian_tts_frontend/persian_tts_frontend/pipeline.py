"""
The Normalizer.

This is the frozen artifact. Its `version` must be recorded in every manifest
row and must be identical at training and at serving time. Frontend/model skew
is the standard way to ship a TTS system that scores well offline and
mispronounces things in production.
"""

import hashlib
import inspect
import json
import re
from dataclasses import dataclass, field, asdict

from . import chars, diacritics, expand, latin, normalize, numbers


@dataclass(frozen=True)
class NormalizerConfig:
    keep_harakat: bool = True
    drop_non_persian_marks: bool = True
    parens_to_commas: bool = True
    repair_zwnj: bool = True
    expand_numerics: bool = True
    decimal_style: str = "momayez"          # or "fractional"
    skip_expansions: frozenset = frozenset()
    latin_strategy: str = "escalate"        # escalate | lexicon | transliterate
    strip_unknown_chars: bool = True
    max_len: int = 0                        # 0 = no limit
    lowercase_latin: bool = False


@dataclass
class NormResult:
    text: str
    raw: str
    changed: bool
    unexpected: tuple = ()
    diacritic_density: float = 0.0
    n_words: int = 0
    problems: tuple = ()

    def ok(self) -> bool:
        return not self.unexpected and not self.problems


class Normalizer:
    """Deterministic, idempotent Persian text normalization for TTS.

        >>> nz = Normalizer()
        >>> nz("در سال ۱۴۰۳ حدود ٪۲۵ رشد داشت.")
        'در سال هزار و چهارصد و سه حدود بیست و پنج درصد رشد داشت.'
    """

    def __init__(self, config: NormalizerConfig = None, lexicon=None):
        self.cfg = config or NormalizerConfig()
        self.latin = latin.LatinResolver(lexicon=lexicon,
                                         strategy=self.cfg.latin_strategy)
        self._allowed = chars.allowed_charset(self.cfg.keep_harakat)
        # Sentinels and digits are permitted here only because the guard runs
        # before escapes are restored and before (in `skip_expansions` setups)
        # numerals may legitimately survive.
        self._allowed_re = re.compile(
            "[^" + re.escape("".join(sorted(self._allowed)))
            + normalize.SENTINEL_RANGE + "0-9]"
        )
        self._version = None

    # ------------------------------------------------------------- versioning

    @property
    def version(self) -> str:
        """Content hash over the rule source plus the active config.

        Any edit to any rule module changes this string. Write it into the
        manifest; refuse to serve a model whose recorded version does not match
        the deployed frontend.
        """
        if self._version is None:
            h = hashlib.sha256()
            for mod in (chars, normalize, numbers, expand, latin, diacritics):
                h.update(inspect.getsource(mod).encode("utf-8"))
            h.update(json.dumps(
                {k: sorted(v) if isinstance(v, frozenset) else v
                 for k, v in asdict(self.cfg).items()},
                sort_keys=True, ensure_ascii=False).encode("utf-8"))
            self._version = "fa-fe-" + h.hexdigest()[:12]
        return self._version

    # --------------------------------------------------------------- the pass

    def normalize(self, text: str) -> NormResult:
        raw = text
        if not isinstance(text, str):
            return NormResult("", str(text), False, ("non_str_input",))

        # 1. phoneme escapes out of harm's way -- must be first
        text, payloads = normalize.protect_escapes(text)

        # 2. unicode canonicalisation
        text = normalize.decompose_presentation_forms(text)
        text = normalize.to_nfc(text)

        # 3. invisibles and codepoint folding
        text = normalize.remove_invisibles(text)
        text = normalize.fold_codepoints(text)
        if self.cfg.drop_non_persian_marks:
            text = normalize.drop_non_persian_marks(text)

        # 4. punctuation
        text = normalize.normalize_punctuation(
            text, parens_to_commas=self.cfg.parens_to_commas)

        # 5. numeric / symbolic expansion (needs ASCII digits, hence after fold)
        if self.cfg.expand_numerics:
            text = expand.expand_all(text,
                                     decimal_style=self.cfg.decimal_style,
                                     skip=self.cfg.skip_expansions)

        # 6. latin resolution (after numerics so "iPhone 15" reads correctly).
        #    The resolver's "escalate" strategy emits new {escapes}; those must
        #    be protected too or the charset guard strips the Latin inside them
        #    and leaves an empty "{}".
        text = self.latin(text)
        text, payloads = normalize.protect_escapes(text, payloads)

        # 7. cleanup, ZWNJ repair, spacing
        text = normalize.cleanup(text)
        if self.cfg.repair_zwnj:
            text = normalize.repair_zwnj(text)
        text = normalize.fix_spacing(text)

        # 8. NFC again -- expansion inserted new letters
        text = normalize.to_nfc(text)

        # 9. charset guard
        unexpected = tuple(sorted(set(self._allowed_re.findall(text))))
        if unexpected and self.cfg.strip_unknown_chars:
            text = self._allowed_re.sub("", text)
            text = normalize.fix_spacing(text)

        # 10. escapes back
        text = normalize.restore_escapes(text, payloads)

        problems = tuple(diacritics.validate(text)) if self.cfg.keep_harakat else ()
        words = diacritics.WORD_RE.findall(text)

        return NormResult(
            text=text,
            raw=raw,
            changed=(text != raw),
            unexpected=unexpected,
            diacritic_density=diacritics.word_density(text),
            n_words=len(words),
            problems=problems,
        )

    def __call__(self, text: str) -> str:
        return self.normalize(text).text

    # ------------------------------------------------------------- invariants

    def check_idempotent(self, text: str) -> bool:
        once = self(text)
        return once == self(once)

    def selftest(self, samples) -> dict:
        """Run the invariants that must hold for every corpus. Call this on a
        few thousand rows before committing to a full pass."""
        fails = {"idempotent": [], "unexpected": [], "problems": [], "empty": []}
        for s in samples:
            r = self.normalize(s)
            if not r.text.strip():
                fails["empty"].append(s)
                continue
            if self(r.text) != r.text:
                fails["idempotent"].append(s)
            if r.unexpected:
                fails["unexpected"].append((s, r.unexpected))
            if r.problems:
                fails["problems"].append((s, r.problems))
        return {k: v for k, v in fails.items() if v}
