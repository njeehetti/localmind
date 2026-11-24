"""Stage 5b: find every amount, date and time on a page, by rules only.

The LLM never writes a number (approach C of the design): rules find the
values, each gets a tag written into the page just before it
(``[M4K] 84,20 €``, ``[D7B] 15/03/2026``, ``[H2Q] 14h30``), and the model only
answers with tags. A value it could not have seen cannot reach a fact.

Tags use one numbering for all candidates, in text order, turned into a short
non-sequential code: with ``M1, D1, M2, D2…`` the model was measured counting
instead of copying (spec 7.1), pairing an amount with the wrong line's date.

French formats only for now: day first, ``1 234,56 €``, ``15 mars 2026``.
Month-only dates ("mars 2027") are not candidates: no day, nothing reliable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

CURRENCY = "EUR"

_MONTHS = {
    "janvier": 1, "fevrier": 2, "février": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
    "juillet": 7, "aout": 8, "août": 8, "septembre": 9, "octobre": 10, "novembre": 11,
    "decembre": 12, "décembre": 12,
}
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))
_SEP = "[   ]"  # thousands: space, no-break space, narrow no-break space
_NUM = rf"(?:\d{{1,3}}(?:(?:{_SEP}|\.(?=\d{{3}}(?!\d)))\d{{3}})+|\d+)(?:[,.]\d{{1,2}})?"
_CUR = r"(?:€|EUR\b|euros?\b)"
_AMOUNT = re.compile(
    rf"(?<![\d.,])(?:([-+−]){_SEP}?)?({_NUM}){_SEP}?{_CUR}|{_CUR}{_SEP}?([-+−]?)({_NUM})(?!\d)",
    re.IGNORECASE,
)
_D_NUM = re.compile(r"(?<![\d/.-])(\d{1,2})[/.-](\d{1,2})[/.-](\d{4}|\d{2})(?!\d)")
_D_ISO = re.compile(r"(?<![\d-])(\d{4})-(\d{2})-(\d{2})(?!\d)")
_D_TEXT = re.compile(rf"(?<!\d)(1er|\d{{1,2}})\s+({_MONTH_RE})\s+(\d{{4}})(?!\d)", re.IGNORECASE)
_D_RECURRING = re.compile(
    r"\ble\s+(1er|\d{1,2})\s+(?:de\s+chaque\s+mois|du\s+mois)", re.IGNORECASE
)
_TIME = re.compile(
    r"(?<![\d,.:/])([01]?\d|2[0-3])\s?(?:h(?:\s?([0-5]\d))?(?![a-zà-ÿ\d])|:([0-5]\d))(?!\d)",
    re.IGNORECASE,
)
_ALPHA = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # no I, O: never read as 1, 0
_BLOCK = 10 * len(_ALPHA)  # digit + letter codes per block


@dataclass(frozen=True)
class Amount:
    id: str
    cents: int  # signed
    currency: str
    start: int
    end: int


@dataclass(frozen=True)
class DateCandidate:
    id: str
    iso: str | None  # "2026-03-15", or None for a day of the month
    day_of_month: int | None  # "le 5 de chaque mois" -> 5
    start: int
    end: int


@dataclass(frozen=True)
class TimeCandidate:
    id: str
    hhmm: str  # "14:30"
    start: int
    end: int


Candidate = Amount | DateCandidate | TimeCandidate


@dataclass(frozen=True)
class PageCandidates:
    amounts: tuple[Amount, ...]
    dates: tuple[DateCandidate, ...]
    times: tuple[TimeCandidate, ...]

    def get(self, tag: str) -> Candidate | None:
        return next((c for c in self.all() if c.id == tag), None)

    def all(self) -> tuple[Candidate, ...]:
        return tuple(sorted((*self.amounts, *self.dates, *self.times), key=lambda c: c.start))


def _code(i: int) -> str:
    """Unique per i >= 1, deterministic, not a counter: 1 -> "2A", 2 -> "3P"…"""
    block, j = divmod(i - 1, _BLOCK)
    n = (j * 37 + 48) % _BLOCK  # 37 is prime to 240: a permutation of the block
    code = f"{n // len(_ALPHA)}{_ALPHA[n % len(_ALPHA)]}"
    while block:
        block, rest = divmod(block - 1, len(_ALPHA))
        code += _ALPHA[rest]
    return code


def _iso(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _year(text: str) -> int:
    y = int(text)
    if len(text) == 4:
        return y
    return 2000 + y if y < 70 else 1900 + y


def _day(text: str) -> int:
    return 1 if text.lower() == "1er" else int(text)


def _cents(sign: str, number: str) -> int:
    raw = re.sub(_SEP, "", number)
    decimals = re.search(r"[,.](\d{1,2})$", raw)
    if decimals:
        whole, frac = raw[: decimals.start()], (decimals[1] + "0")[:2]
    else:
        whole, frac = raw, "00"
    cents = int(whole.replace(".", "")) * 100 + int(frac)
    return -cents if sign in ("-", "−") else cents


class _Finder:
    def __init__(self, text: str) -> None:
        self.text = text
        self.taken: list[tuple[int, int]] = []
        self.found: list[tuple[str, int, int, dict]] = []  # kind, start, end, values

    def _free(self, start: int, end: int) -> bool:
        return all(end <= s or start >= e for s, e in self.taken)

    def add(self, kind: str, match: re.Match[str], **values) -> None:
        if self._free(match.start(), match.end()):
            self.taken.append((match.start(), match.end()))
            self.found.append((kind, match.start(), match.end(), values))

    def run(self) -> PageCandidates:
        text = self.text
        for m in _D_RECURRING.finditer(text):  # most specific first: dates claim their digits
            if 1 <= _day(m[1]) <= 31:
                self.add("D", m, iso=None, day_of_month=_day(m[1]))
        for m in _D_TEXT.finditer(text):
            iso = _iso(int(m[3]), _MONTHS[m[2].lower()], _day(m[1]))
            if iso:
                self.add("D", m, iso=iso, day_of_month=None)
        for pattern, ymd in ((_D_ISO, (1, 2, 3)), (_D_NUM, (3, 2, 1))):
            for m in pattern.finditer(text):
                y, mo, d = (m[i] for i in ymd)
                iso = _iso(_year(y), int(mo), int(d))
                if iso:
                    self.add("D", m, iso=iso, day_of_month=None)
        for m in _AMOUNT.finditer(text):
            sign, number = (m[1] or "", m[2]) if m[2] else (m[3], m[4])
            self.add("M", m, cents=_cents(sign, number), currency=CURRENCY)
        for m in _TIME.finditer(text):
            self.add("H", m, hhmm=f"{int(m[1]):02d}:{m[2] or m[3] or '00'}")
        return self._numbered()

    def _numbered(self) -> PageCandidates:
        amounts, dates, times = [], [], []
        for i, (kind, start, end, values) in enumerate(sorted(self.found, key=lambda f: f[1]), 1):
            tag = kind + _code(i)
            if kind == "M":
                amounts.append(Amount(tag, start=start, end=end, **values))
            elif kind == "D":
                dates.append(DateCandidate(tag, start=start, end=end, **values))
            else:
                times.append(TimeCandidate(tag, start=start, end=end, **values))
        return PageCandidates(tuple(amounts), tuple(dates), tuple(times))


def find_candidates(text: str) -> PageCandidates:
    """Every amount, date and time of ``text``, tagged in text order."""
    return _Finder(text).run()


def annotate(text: str, found: PageCandidates) -> str:
    """``text`` with ``"[TAG] "`` written right before each candidate."""
    for c in sorted(found.all(), key=lambda c: c.start, reverse=True):
        text = f"{text[: c.start]}[{c.id}] {text[c.start:]}"
    return text
