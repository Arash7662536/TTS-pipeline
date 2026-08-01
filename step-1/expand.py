"""
Expansion of numeric and symbolic expressions into spoken Persian.

Order of application is significant and is fixed in `expand_all`: the most
specific patterns must run first, or a generic integer rule will consume the
digits a date rule needed.
"""

import re

from . import numbers as N
from .chars import (PERSIAN_COMMA, PERSIAN_DECIMAL, PERSIAN_PERCENT,
                    PERSIAN_THOUSANDS, ZWNJ)

# ------------------------------------------------------------------ calendars

JALALI_MONTHS = ["فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور",
                 "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند"]

GREGORIAN_MONTHS = ["ژانویه", "فوریه", "مارس", "آوریل", "مه", "ژوئن",
                    "ژوئیه", "اوت", "سپتامبر", "اکتبر", "نوامبر", "دسامبر"]

# ------------------------------------------------------------------ lexicons

CURRENCY = {
    "$": "دلار", "USD": "دلار", "usd": "دلار",
    "€": "یورو", "EUR": "یورو",
    "£": "پوند", "GBP": "پوند",
    "\ufdfc": "ریال", "IRR": "ریال",
    "\u00a5": "ین",
}

CURRENCY_WORDS = ("تومان", "ریال", "دلار", "یورو", "پوند", "درهم", "دینار", "ین")

UNITS = {
    "km": "کیلومتر", "m": "متر", "cm": "سانتی" + ZWNJ + "متر",
    "mm": "میلی" + ZWNJ + "متر", "kg": "کیلوگرم", "g": "گرم",
    "mg": "میلی" + ZWNJ + "گرم", "l": "لیتر", "ml": "میلی" + ZWNJ + "لیتر",
    "km/h": "کیلومتر بر ساعت", "m/s": "متر بر ثانیه",
    "GB": "گیگابایت", "MB": "مگابایت", "KB": "کیلوبایت", "TB": "ترابایت",
    "GHz": "گیگاهرتز", "MHz": "مگاهرتز", "Hz": "هرتز",
    "\u00b0C": "درجه سانتی" + ZWNJ + "گراد", "\u00b0F": "درجه فارنهایت",
    "\u00b0": "درجه",
}

ABBREVIATIONS = {
    "ه.ق": "هجری قمری", "ه\u200cق": "هجری قمری", "هـ.ق": "هجری قمری",
    "ه.ش": "هجری شمسی", "ه\u200cش": "هجری شمسی", "هـ.ش": "هجری شمسی",
    "ق.م": "قبل از میلاد", "ب.م": "بعد از میلاد",
    "ص.": "صفحه", "ج.": "جلد", "ش.": "شماره",
    "و غیره": "و غیره", "الخ": "و غیره",
    "ن.ک": "نگاه کنید", "ر.ک": "رجوع کنید",
    "م.": "میلادی",
}

# ------------------------------------------------------------------ patterns

_D = r"[0-9]"
RE_TIME = re.compile(rf"\b({_D}{{1,2}}):({_D}{{2}})(?::({_D}{{2}}))?\b")
RE_DATE_YMD = re.compile(rf"\b({_D}{{4}})[/\-\.]({_D}{{1,2}})[/\-\.]({_D}{{1,2}})\b")
RE_DATE_DMY = re.compile(rf"\b({_D}{{1,2}})[/\-\.]({_D}{{1,2}})[/\-\.]({_D}{{4}})\b")
RE_PHONE = re.compile(rf"(?<!{_D})0{_D}{{9,10}}(?!{_D})")
RE_LONG_ID = re.compile(rf"(?<!{_D})({_D}{{11,}})(?!{_D})")
RE_PERCENT = re.compile(
    rf"({_D}+(?:[{PERSIAN_DECIMAL}\.]{_D}+)?)\s*[{PERSIAN_PERCENT}%]"
)
RE_PERCENT_PRE = re.compile(rf"[{PERSIAN_PERCENT}%]\s*({_D}+)")
RE_CURRENCY_SYM = re.compile(
    r"([$\u20ac\u00a3\u00a5\ufdfc])\s*(" + _D + r"+(?:[,\u066c]" + _D + r"{3})*"
    r"(?:[\u066b\.]" + _D + r"+)?)"
)
RE_FRACTION = re.compile(rf"\b({_D}+)\s*/\s*({_D}+)\b")
RE_RANGE = re.compile(rf"({_D}+)\s*[-\u2013\u2014]\s*({_D}+)")
RE_ORDINAL = re.compile(rf"\b({_D}+)\s*(?:{ZWNJ}?ام|{ZWNJ}?مین)\b")
RE_DECIMAL = re.compile(rf"({_D}+)[{PERSIAN_DECIMAL}\.]({_D}+)")
# PERSIAN_COMMA belongs here even though it is not a thousands separator: the
# punctuation pass runs first and has already turned every ASCII "," into "،",
# so "1,250" reaches this rule as "1،250". A real enumeration keeps its space
# ("1، 250") and the no-space lookahead below leaves it alone.
RE_THOUSANDS = re.compile(
    rf"(?<={_D})[,{PERSIAN_THOUSANDS}{PERSIAN_COMMA}](?={_D}{{3}}\b)")
# A sign only counts when it is glued to the digits AND starts a token:
#   "-5"  -> negative           "12-15"  -> range, the dash is preceded by a digit
#   "+16" -> positive           "- 5"    -> dash list marker / prosodic break
RE_SIGNED = re.compile(rf"(?:(?<=^)|(?<=\s))([+-])({_D})")
RE_INT = re.compile(rf"{_D}+")
RE_UNIT = re.compile(
    r"(?<=[0-9\s])(" + "|".join(
        sorted((re.escape(u) for u in UNITS), key=len, reverse=True)
    ) + r")\b"
)


# ------------------------------------------------------------------ expanders

def _month_name(month: int, jalali: bool) -> str:
    table = JALALI_MONTHS if jalali else GREGORIAN_MONTHS
    return table[month - 1] if 1 <= month <= 12 else N.cardinal(month)


def expand_time(text: str) -> str:
    def sub(m):
        h, mi, s = int(m.group(1)), int(m.group(2)), m.group(3)
        if h > 23 or mi > 59:
            return m.group(0)
        # Don't say "ساعت" twice when the source already has it.
        prefix = "" if re.search(r"ساعت\s*$", text[:m.start()]) else "ساعت "
        out = f"{prefix}{N.cardinal(h)}"
        if mi:
            out += f" و {N.cardinal(mi)} دقیقه"
        if s and int(s):
            out += f" و {N.cardinal(int(s))} ثانیه"
        return out
    return RE_TIME.sub(sub, text)


def expand_dates(text: str) -> str:
    def ymd(m):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not (1 <= mo <= 12 and 1 <= d <= 31):
            return m.group(0)
        jalali = 1200 <= y <= 1500
        return f"{N.ordinal(d)} {_month_name(mo, jalali)} {N.year(y)}"

    def dmy(m):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not (1 <= mo <= 12 and 1 <= d <= 31):
            return m.group(0)
        jalali = 1200 <= y <= 1500
        return f"{N.ordinal(d)} {_month_name(mo, jalali)} {N.year(y)}"

    text = RE_DATE_YMD.sub(ymd, text)
    text = RE_DATE_DMY.sub(dmy, text)
    return text


def expand_phone(text: str) -> str:
    text = RE_PHONE.sub(lambda m: N.digits(m.group(0)), text)
    return RE_LONG_ID.sub(lambda m: N.digits(m.group(1)), text)


def expand_percent(text: str) -> str:
    def sub(m):
        raw = m.group(1).replace(PERSIAN_DECIMAL, ".")
        if "." in raw:
            i, f = raw.split(".", 1)
            return f"{N.decimal(i, f)} درصد"
        return f"{N.cardinal(int(raw))} درصد"
    text = RE_PERCENT.sub(sub, text)
    return RE_PERCENT_PRE.sub(
        lambda m: f"{N.cardinal(int(m.group(1)))} درصد", text)


def expand_currency(text: str) -> str:
    def sub(m):
        sym, amount = m.group(1), m.group(2)
        amount = amount.replace(",", "").replace(PERSIAN_THOUSANDS, "")
        unit = CURRENCY.get(sym, "")
        if PERSIAN_DECIMAL in amount or "." in amount:
            i, f = re.split(rf"[{PERSIAN_DECIMAL}\.]", amount, 1)
            return f"{N.decimal(i, f)} {unit}".strip()
        return f"{N.cardinal(int(amount))} {unit}".strip()
    return RE_CURRENCY_SYM.sub(sub, text)


def expand_units(text: str) -> str:
    return RE_UNIT.sub(lambda m: " " + UNITS[m.group(1)], text)


def expand_abbreviations(text: str) -> str:
    """Boundary-aware abbreviation expansion.

    A naive str.replace here is a real hazard: "م." matches the tail of
    "گرفتیم." and silently rewrites "we took a photo" as "we took AD". Every
    abbreviation must be anchored against adjacent letters on both sides.
    """
    for abbr in sorted(ABBREVIATIONS, key=len, reverse=True):
        pat = (r"(?<![\u0600-\u06ffA-Za-z])" + re.escape(abbr)
               + r"(?![\u0600-\u06ffA-Za-z])")
        text = re.sub(pat, " " + ABBREVIATIONS[abbr] + " ", text)
    return text


def expand_fractions(text: str) -> str:
    def sub(m):
        num, den = int(m.group(1)), int(m.group(2))
        if den == 0 or den > 1000 or num > 1000:
            return m.group(0)
        return N.fraction(num, den)
    return RE_FRACTION.sub(sub, text)


def expand_signs(text: str) -> str:
    """`-5` -> `منفی 5`, `+16` -> `مثبت 16`.

    Only the sign is rewritten; the digits are left for the later numeric rules,
    so `-12.5 کیلومتر` still goes through decimals and units. Without this the
    `+` was dropped by the charset guard (visible in the report) and the `-` was
    silently rewritten to a comma by the punctuation pass -- which read a
    negative temperature as a positive one.
    """
    return RE_SIGNED.sub(
        lambda m: (N.NEGATIVE if m.group(1) == "-" else N.POSITIVE)
        + " " + m.group(2), text)


def expand_ranges(text: str) -> str:
    return RE_RANGE.sub(
        lambda m: f"{N.cardinal(int(m.group(1)))} تا {N.cardinal(int(m.group(2)))}",
        text)


def expand_ordinals(text: str) -> str:
    def sub(m):
        n = int(m.group(1))
        suffix_min = "مین" in m.group(0)
        o = N.ordinal(n, first_as_avval=not suffix_min)
        return o + ("" if not suffix_min else "")
    return RE_ORDINAL.sub(sub, text)


def expand_decimals(text: str, style: str = "momayez") -> str:
    return RE_DECIMAL.sub(
        lambda m: N.decimal(m.group(1), m.group(2), style=style), text)


def expand_integers(text: str) -> str:
    return RE_INT.sub(lambda m: N.cardinal(int(m.group(0))), text)


def strip_thousands(text: str) -> str:
    return RE_THOUSANDS.sub("", text)


# ------------------------------------------------------------------ orchestration

#: Fixed order. Do not reorder without re-running tests -- earlier rules
#: depend on later ones not having eaten their digits.
EXPANSION_ORDER = (
    "abbreviations",
    "thousands",
    "signs",       # before anything that eats digits, or the sign is orphaned
    "time",
    "dates",
    "phone",
    "currency_symbol",
    "percent",
    "units",
    "fractions",
    "ordinals",
    "ranges",
    "decimals",
    "integers",
)


def expand_all(text: str, decimal_style: str = "momayez",
               skip: frozenset = frozenset()) -> str:
    steps = {
        "abbreviations": expand_abbreviations,
        "thousands": strip_thousands,
        "signs": expand_signs,
        "time": expand_time,
        "dates": expand_dates,
        "phone": expand_phone,
        "currency_symbol": expand_currency,
        "percent": expand_percent,
        "units": expand_units,
        "fractions": expand_fractions,
        "ordinals": expand_ordinals,
        "ranges": expand_ranges,
        "decimals": lambda t: expand_decimals(t, style=decimal_style),
        "integers": expand_integers,
    }
    for name in EXPANSION_ORDER:
        if name in skip:
            continue
        text = steps[name](text)
    return text
