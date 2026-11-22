"""Detect sensitive data (IBAN, payment card, French social security number, passwords).

The pattern used everywhere here: **a regex finds candidates, a checksum confirms them.**

A regex like "16 digits" matches every order number and tracking code. But IBANs,
card numbers and French social security numbers (NIR) carry a *check value* computed
from the other digits. A random number passes the check only by chance (1 in 10 for
Luhn, about 1 in 97 for mod-97), which removes most false positives for free.

Nothing here stores or prints the values: callers only get positions (spans).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SensitiveSpan:
    kind: str  # "iban" | "card" | "fr_nir" | "password"
    start: int
    end: int


# --- checksums -------------------------------------------------------------


def iban_is_valid(candidate: str) -> bool:
    """ISO 13616 / ISO 7064 mod-97: move the first 4 chars to the end,
    turn letters into numbers (A=10 ... Z=35), the result mod 97 must be 1."""
    iban = candidate.replace(" ", "").upper()
    if not (15 <= len(iban) <= 34 and iban.isalnum()):
        return False
    rearranged = iban[4:] + iban[:4]
    return int("".join(str(int(ch, 36)) for ch in rearranged)) % 97 == 1


def luhn_is_valid(digits: str) -> bool:
    """Luhn: from the right, double every second digit (minus 9 if > 9);
    the total must be a multiple of 10. Catches any single-digit typo."""
    if not digits.isdigit():
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


def nir_is_valid(digits: str) -> bool:
    """French NIR: 13 digits + 2-digit key, key = 97 - (number mod 97).
    (Corsican numbers with 2A/2B are not handled.)"""
    if len(digits) != 15 or not digits.isdigit() or digits[0] not in "12":
        return False
    return 97 - int(digits[:13]) % 97 == int(digits[13:])


# --- candidate finders -----------------------------------------------------

# Two letters, two check digits, then groups of letters/digits separated by single spaces.
_IBAN = re.compile(r"\b[A-Za-z]{2}\d{2}[A-Za-z0-9]*(?: [A-Za-z0-9]+)*")
_NIR = re.compile(r"(?<!\d)[12](?: ?\d){14}(?!\d)")
# Real cards are written contiguous, 4-4-4-4(-3) or 4-6-5 (Amex), with one
# consistent separator. Accepting a space anywhere would turn every pair of
# dates on a bank statement ("2024-03-15 2024-03-17") into a candidate.
_CARD = re.compile(
    r"(?<![\d-])(?:\d{13,19}"
    r"|\d{4}([ -])\d{4}\1\d{4}\1\d{1,7}"
    r"|\d{4}([ -])\d{6}\2\d{5})(?![\d-])"
)
_PASSWORD = re.compile(
    r"^[^\S\n]*(?:password|passwd|pwd|mot de passe|mdp)[^\S\n]*[:=][^\S\n]*(\S+)",
    re.IGNORECASE | re.MULTILINE,
)


def _find_ibans(text: str) -> list[SensitiveSpan]:
    spans = []
    for m in _IBAN.finditer(text):
        # The regex is greedy ("FR76 ... 189 fin" swallows "fin"), so drop
        # trailing space-separated groups until the checksum passes.
        groups = m.group().split(" ")
        while groups:
            candidate = " ".join(groups)
            if iban_is_valid(candidate):
                spans.append(SensitiveSpan("iban", m.start(), m.start() + len(candidate)))
                break
            groups.pop()
    return spans


def _find_checked(pattern: re.Pattern[str], kind: str, check, text: str) -> list[SensitiveSpan]:
    # No trimming here: trying shorter prefixes of a long number would let
    # random digits pass Luhn by chance.
    return [
        SensitiveSpan(kind, m.start(), m.end())
        for m in pattern.finditer(text)
        if check(re.sub(r"[ -]", "", m.group()))
    ]


def _find_passwords(text: str) -> list[SensitiveSpan]:
    return [SensitiveSpan("password", m.start(1), m.end(1)) for m in _PASSWORD.finditer(text)]


def _overlaps(span: SensitiveSpan, taken: list[SensitiveSpan]) -> bool:
    return any(span.start < t.end and t.start < span.end for t in taken)


def find_sensitive(text: str) -> tuple[SensitiveSpan, ...]:
    """All sensitive spans, sorted by position. Most specific detectors run first,
    so a number already claimed as an IBAN is not reported again as a card."""
    found: list[SensitiveSpan] = []
    for spans in (
        _find_ibans(text),
        _find_checked(_NIR, "fr_nir", nir_is_valid, text),
        _find_checked(_CARD, "card", luhn_is_valid, text),
        _find_passwords(text),
    ):
        found.extend(s for s in spans if not _overlaps(s, found))
    return tuple(sorted(found, key=lambda s: s.start))
