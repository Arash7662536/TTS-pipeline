"""
Persian number -> words.

Correctness here matters more than anywhere else in the frontend: a bad number
reading is immediately audible and there is no way for the acoustic model to
recover from it. Every branch below is covered in tests/test_numbers.py.
"""

from .chars import ZWNJ

ZERO = "صفر"
NEGATIVE = "منفی"
POSITIVE = "مثبت"
AND = " و "

ONES = ["", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه"]
TEENS = ["ده", "یازده", "دوازده", "سیزده", "چهارده",
         "پانزده", "شانزده", "هفده", "هجده", "نوزده"]
TENS = ["", "", "بیست", "سی", "چهل", "پنجاه", "شصت", "هفتاد", "هشتاد", "نود"]
HUNDREDS = ["", "صد", "دویست", "سیصد", "چهارصد",
            "پانصد", "ششصد", "هفتصد", "هشتصد", "نهصد"]
SCALES = ["", "هزار", "میلیون", "میلیارد", "بیلیون", "بیلیارد",
          "تریلیون", "تریلیارد"]

DIGIT_NAMES = ["صفر", "یک", "دو", "سه", "چهار",
               "پنج", "شش", "هفت", "هشت", "نه"]

# Decimal place names, for the "fractional" reading style.
DECIMAL_PLACES = ["", "دهم", "صدم", "هزارم", "ده‌هزارم", "صد‌هزارم", "میلیونم"]


def _three(n: int) -> str:
    """0 < n < 1000 -> words."""
    parts = []
    h, r = divmod(n, 100)
    if h:
        parts.append(HUNDREDS[h])
    if r:
        if r < 10:
            parts.append(ONES[r])
        elif r < 20:
            parts.append(TEENS[r - 10])
        else:
            t, o = divmod(r, 10)
            parts.append(TENS[t])
            if o:
                parts.append(ONES[o])
    return AND.join(parts)


def cardinal(n: int) -> str:
    """Integer -> Persian words. Handles negatives and arbitrary magnitude
    up to the SCALES table (10^24)."""
    if not isinstance(n, int):
        raise TypeError(f"cardinal() needs an int, got {type(n).__name__}")
    if n == 0:
        return ZERO
    if n < 0:
        return f"{NEGATIVE} {cardinal(-n)}"

    groups = []
    while n:
        n, g = divmod(n, 1000)
        groups.append(g)

    if len(groups) > len(SCALES):
        # Beyond the named scales, fall back to digit-by-digit rather than
        # emitting something wrong.
        return digits(str(sum(g * 1000 ** i for i, g in enumerate(groups))))

    parts = []
    for i, g in enumerate(groups):
        if g == 0:
            continue
        if i == 0:
            parts.append(_three(g))
        elif i == 1 and g == 1:
            parts.append(SCALES[1])          # "هزار", never "یک هزار"
        else:
            parts.append(f"{_three(g)} {SCALES[i]}")
    return AND.join(reversed(parts))


def ordinal(n: int, first_as_avval: bool = True) -> str:
    """Integer -> Persian ordinal.

    first_as_avval: render 1 as "اول" rather than "یکم". Compounds such as 21
    always use the "-یکم" form regardless.
    """
    if n == 1 and first_as_avval:
        return "اول"
    w = cardinal(n)
    if w.endswith("سه"):
        return w[:-2] + "سوم"
    if w.endswith("یک"):
        return w[:-2] + "یکم"
    if w.endswith("ی"):                      # سی -> سی‌ام
        return w + ZWNJ + "ام"
    return w + "م"


def digits(s: str, group: bool = False) -> str:
    """Read a digit string one digit at a time: phone numbers, ID numbers,
    anything where place value is not meant to be spoken.

    group: insert no separators (kept for API symmetry with future grouping
    strategies such as pairs for phone numbers).
    """
    out = [DIGIT_NAMES[int(c)] for c in s if c.isdigit()]
    return " ".join(out)


def decimal(int_part: str, frac_part: str, style: str = "momayez") -> str:
    """Read a decimal number.

    style="momayez"    -> "سه ممیز یک چهار"   (digit-by-digit after the point;
                          how it is almost always said aloud)
    style="fractional" -> "سه و چهارده صدم"   (mathematical register)
    """
    ip = cardinal(int(int_part)) if int_part else ZERO
    frac_part = frac_part.rstrip("0") or "0"
    if style == "fractional" and len(frac_part) < len(DECIMAL_PLACES):
        place = DECIMAL_PLACES[len(frac_part)]
        return f"{ip}{AND}{cardinal(int(frac_part))} {place}"
    return f"{ip} ممیز {digits(frac_part)}"


def fraction(num: int, den: int) -> str:
    """1/2 -> "یک دوم", 3/4 -> "سه چهارم". Common special cases spoken
    differently are handled explicitly."""
    special = {(1, 2): "نیم", (1, 3): "یک سوم", (1, 4): "یک چهارم"}
    if (num, den) in special:
        return special[(num, den)]
    return f"{cardinal(num)} {ordinal(den, first_as_avval=False)}"


def year(n: int) -> str:
    """Persian reads years as plain cardinals -- 1403 is
    "هزار و چهارصد و سه", not the English "fourteen oh three" pattern.
    Kept as a named function so the policy is explicit and changeable in
    one place."""
    return cardinal(n)
