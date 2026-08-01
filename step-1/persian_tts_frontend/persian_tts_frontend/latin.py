"""
Latin-script handling.

Policy (locked in the build plan): do not try to learn code-switching from a
corpus that barely contains any. Instead resolve Latin at the frontend --
lexicon first, rule-based transliteration second, and escalate whatever is left
to the phoneme escape hatch.

The `unknown` collector is the point of this module: running it over the full
corpus produces a ranked work queue of Latin tokens to add to the lexicon, so
the lexicon is built from what actually occurs rather than guessed at.
"""

import re
from collections import Counter

LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\.\-]*")
ACRONYM_RE = re.compile(r"^[A-Z]{2,6}$")

# ------------------------------------------------------------------ lexicon

#: Seed lexicon. Extend from the ranked `unknown` counter after the first
#: corpus pass -- do not try to write this by hand up front.
LEXICON = {
    # platforms / brands
    "google": "گوگل", "youtube": "یوتیوب", "instagram": "اینستاگرام",
    "telegram": "تلگرام", "whatsapp": "واتساپ", "twitter": "توییتر",
    "facebook": "فیسبوک", "microsoft": "مایکروسافت", "apple": "اپل",
    "amazon": "آمازون", "netflix": "نتفلیکس", "spotify": "اسپاتیفای",
    "linkedin": "لینکدین", "tiktok": "تیک‌تاک", "wikipedia": "ویکی‌پدیا",
    "android": "اندروید", "windows": "ویندوز", "linux": "لینوکس",
    "iphone": "آیفون", "ipad": "آی‌پد", "macbook": "مک‌بوک",
    "samsung": "سامسونگ", "huawei": "هوآوی", "xiaomi": "شیائومی",
    "chatgpt": "چت‌جی‌پی‌تی", "openai": "اوپن‌ای‌آی",
    # tech vocabulary
    "email": "ایمیل", "internet": "اینترنت", "online": "آنلاین",
    "offline": "آفلاین", "download": "دانلود", "upload": "آپلود",
    "website": "وبسایت", "web": "وب", "app": "اپلیکیشن",
    "software": "نرم‌افزار", "hardware": "سخت‌افزار",
    "password": "پسورد", "username": "نام کاربری", "login": "لاگین",
    "wifi": "وای‌فای", "bluetooth": "بلوتوث", "usb": "یو‌اس‌بی",
    "podcast": "پادکست", "blog": "بلاگ", "server": "سرور",
    "laptop": "لپ‌تاپ", "tablet": "تبلت", "camera": "دوربین",
    # general loanwords common in audiobooks
    "ok": "اوکی", "okay": "اوکی", "hotel": "هتل", "taxi": "تاکسی",
    "radio": "رادیو", "television": "تلویزیون", "cinema": "سینما",
    "doctor": "دکتر", "professor": "پروفسور", "restaurant": "رستوران",
    "piano": "پیانو", "guitar": "گیتار", "orchestra": "ارکستر",
    # acronyms spoken letter-by-letter in Persian
    "cd": "سی‌دی", "dvd": "دی‌وی‌دی", "tv": "تی‌وی", "pc": "پی‌سی",
    "ai": "هوش مصنوعی", "gps": "جی‌پی‌اس", "sms": "اس‌ام‌اس",
    "id": "آی‌دی", "url": "یو‌آر‌ال", "pdf": "پی‌دی‌اف",
}

#: Corpus-derived additions, ranked from the thomclas `unknown` counter (2317
#: distinct tokens / 7710 occurrences over 109k rows). These are the head of
#: that distribution -- the ~90 entries below cover roughly a quarter of all
#: unresolved word occurrences, and they are the ones the crude transliterator
#: mangles worst ("the" -> "ت"). A native speaker should review the readings.
LEXICON.update({
    "the": "د", "of": "آو", "and": "اند", "in": "این", "to": "تو",
    "on": "آن", "is": "ایز", "it": "ایت", "if": "ایف", "we": "وی",
    "out": "اوت", "off": "آف", "per": "پر", "see": "سی", "what": "وات",
    "deal": "دیل", "new": "نیو", "value": "ولیو", "war": "وار",
    "great": "گریت", "work": "ورک", "state": "استیت", "junk": "جانک",
    "people": "پیپل", "good": "گود", "deep": "دیپ", "world": "ورلد",
    "d-day": "دی‌دی", "uptalk": "آپ‌تاک", "time": "تایم", "game": "گیم",
    "man": "من", "interaction": "اینتراکشن", "blinkist": "بلینکیست",
    "company": "کامپانی", "papal": "پیپال", "dollar": "دلار",
    "street": "استریت", "german": "جرمن", "enlightenment": "انلایتنمنت",
    "public": "پابلیک", "choice": "چویس", "society": "سوسایتی",
    "passion": "پشن", "chain": "چین", "oil": "اویل", "british": "بریتیش",
    "origin": "اوریجین", "history": "هیستوری", "universal": "یونیورسال",
    "cost": "کاست", "serfdom": "سرفدام", "india": "ایندیا",
    "sovereignty": "ساورنتی", "transcontinental": "ترنس‌کانتیننتال",
    "sleep": "اسلیپ", "oversight": "اوورسایت", "effect": "افکت",
    "stream": "استریم", "social": "سوشال", "power": "پاور",
    "city-state": "سیتی‌استیت", "holy": "هولی", "theory": "تئوری",
    "museum": "موزیوم", "trade": "ترید", "balance": "بالانس",
    "big": "بیگ", "first": "فرست", "zoom": "زوم", "anxiety": "انگزایتی",
    "supply": "ساپلای", "efficiency": "افیشنسی", "compromise": "کامپرومایز",
    "james": "جیمز", "depression": "دیپرشن", "recovery": "ریکاوری",
    "poor": "پور", "resurrection": "رزرکشن", "einstein": "اینشتین",
    "art": "آرت", "thinking": "تینکینگ", "productivity": "پرودکتیویتی",
    "lost": "لاست", "connections": "کانکشنز", "main": "مین",
    "puritan": "پیوریتن", "freakonomics": "فریکونومیکس",
})

#: Latin letter names, for spelling out unknown acronyms.
LETTER_NAMES = {
    "a": "ای", "b": "بی", "c": "سی", "d": "دی", "e": "ای", "f": "اف",
    "g": "جی", "h": "اچ", "i": "آی", "j": "جی", "k": "کی", "l": "ال",
    "m": "ام", "n": "ان", "o": "او", "p": "پی", "q": "کیو", "r": "آر",
    "s": "اس", "t": "تی", "u": "یو", "v": "وی", "w": "دبلیو",
    "x": "ایکس", "y": "وای", "z": "زد",
}

# ------------------------------------------------------- rule-based fallback

#: Deliberately crude. It exists so an unknown token produces *something*
#: pronounceable rather than raw Latin the model has never seen in a Persian
#: context. Anything that matters should end up in LEXICON instead.
_DIGRAPHS = [
    ("sch", "ش"), ("tch", "چ"), ("ch", "چ"), ("sh", "ش"), ("th", "ت"),
    ("ph", "ف"), ("gh", "گ"), ("kh", "خ"), ("ck", "ک"), ("qu", "کو"),
    ("oo", "و"), ("ee", "ی"), ("ea", "ی"), ("ou", "و"), ("au", "و"),
    ("ai", "ای"), ("ay", "ی"), ("ey", "ی"), ("oy", "وی"),
]
_SINGLES = {
    "a": "ا", "b": "ب", "c": "ک", "d": "د", "e": "", "f": "ف", "g": "گ",
    "h": "ه", "i": "ی", "j": "ج", "k": "ک", "l": "ل", "m": "م", "n": "ن",
    "o": "و", "p": "پ", "q": "ک", "r": "ر", "s": "س", "t": "ت", "u": "و",
    "v": "و", "w": "و", "x": "کس", "y": "ی", "z": "ز",
}


def transliterate(token: str) -> str:
    t = token.lower()
    for src, dst in _DIGRAPHS:
        t = t.replace(src, dst)
    out = []
    for ch in t:
        if ch in _SINGLES:
            out.append(_SINGLES[ch])
        elif "\u0600" <= ch <= "\u06ff":
            out.append(ch)
    s = "".join(out)
    if s and s[0] in "اوی":
        s = "آ" + s[1:] if s[0] == "ا" else s
    return s or token


def spell_acronym(token: str) -> str:
    return "\u200c".join(LETTER_NAMES.get(c.lower(), c) for c in token if c.isalpha())


# ------------------------------------------------------------------ main pass

class LatinResolver:
    """Resolves Latin tokens and records what it could not resolve.

    strategy:
      "lexicon"     -- lexicon hit only; misses are escaped as {token}
      "escalate"    -- lexicon, then acronym spelling, then escape (default;
                       safest for training data because it never invents a
                       pronunciation silently)
      "transliterate" -- lexicon, acronym, then rule-based transliteration
    """

    def __init__(self, lexicon=None, strategy: str = "escalate"):
        self.lexicon = dict(LEXICON)
        if lexicon:
            self.lexicon.update({k.lower(): v for k, v in lexicon.items()})
        self.strategy = strategy
        self.unknown = Counter()
        self.resolved = Counter()

    def resolve_token(self, token: str) -> str:
        key = token.lower().strip(".-'")
        if key in self.lexicon:
            self.resolved[key] += 1
            return self.lexicon[key]
        # A lone Latin letter is read as its name, never transliterated:
        # "ویتامین d" is "ویتامین دی", and the transliterator would give "د".
        if len(key) == 1 and key.isascii() and key.isalpha():
            self.resolved[key] += 1
            return LETTER_NAMES[key]
        if ACRONYM_RE.match(token):
            self.unknown[token] += 1
            return spell_acronym(token)
        self.unknown[key] += 1
        if self.strategy == "transliterate":
            return transliterate(key)
        if self.strategy == "lexicon":
            return "{" + key + "}"
        return "{" + key + "}"          # escalate -> phoneme escape hatch

    def __call__(self, text: str) -> str:
        return LATIN_RE.sub(lambda m: self.resolve_token(m.group(0)), text)

    def work_queue(self, top: int = 200):
        """Ranked list of unresolved Latin tokens -- feed this back into
        LEXICON after the first corpus pass."""
        return self.unknown.most_common(top)
