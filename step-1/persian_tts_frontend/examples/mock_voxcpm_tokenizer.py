"""
A mock of the VoxCPM 2 (MiniCPM-4) tokenizer, calibrated against the measured
roundtrip output.

Observed behaviour it reproduces:
  * shared Arabic-script letters (ک ی ر م ن ه ا ...) -> 1 token
  * پ چ ژ گ                                          -> 2 tokens (byte fallback)
  * every harakat                                    -> 2 tokens (byte fallback)
  * Persian digits ۰-۹                               -> 2 tokens (byte fallback)
  * ZWNJ U+200C                                      -> 1 token
  * a small number of Persian bigram merges (ان, می, ...)
  * a leading SentencePiece space marker

Use it to validate audit code and to sanity-check fertility estimates offline.
Re-run the real audit against the actual tokenizer before Stage A -- this mock
is calibrated, not authoritative.
"""

SINGLE_TOKEN_LETTERS = set("ابتثجحخدذرزسشصضطظعغفقکلمنوهیآأإئؤء")
BYTE_FALLBACK = set("پچژگ") | set("\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652") \
    | set("۰۱۲۳۴۵۶۷۸۹")
MERGES = ["ان", "می", "ها", "ست", "ند", "را", "در", "که", "این", "آن"]


class MockVoxCPMTokenizer:
    def __init__(self, apply_merges: bool = True):
        self.apply_merges = apply_merges

    def _pieces(self, text: str):
        pieces = []
        i = 0
        if text and not text.startswith(" "):
            pieces.append("\u2581")          # SentencePiece leading space
        while i < len(text):
            if self.apply_merges:
                for m in MERGES:
                    if text.startswith(m, i):
                        pieces.append(m)
                        i += len(m)
                        break
                else:
                    pieces.extend(self._one(text[i]))
                    i += 1
                continue
            pieces.extend(self._one(text[i]))
            i += 1
        return pieces

    @staticmethod
    def _one(ch: str):
        if ch in BYTE_FALLBACK:
            return [f"<0x{b:02X}>" for b in ch.encode("utf-8")]
        if ch in SINGLE_TOKEN_LETTERS or ch in " \u200c.,!?:«»\u060c\u061b\u061f\u2026":
            return [ch]
        # anything else: byte fallback, matching real SentencePiece behaviour
        return [f"<0x{b:02X}>" for b in ch.encode("utf-8")]

    def encode(self, text: str, add_special_tokens: bool = False):
        return [hash(p) % 100000 for p in self._pieces(text)]

    def decode(self, ids):
        raise NotImplementedError("mock cannot decode; use for counting only")

    def convert_ids_to_tokens(self, ids):
        return [str(i) for i in ids]

    def count(self, text: str) -> int:
        return len(self._pieces(text))
