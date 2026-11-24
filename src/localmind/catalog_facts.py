"""Catalog storage for stage 5b: facts read from documents (schema v7).

    fact_runs (one per document)        fact_pages                  facts
    ---------------------------------   -------------------------   ------------------------------
    document_id --> documents (cascade) document_id, page_no        document_id --> documents (cascade)
    model, prompt_version               status: done / failed /     page_no, kind, amount_cents,
    status, capped, facts, rejected     no_candidates               date, day_of_month, time,
                                                                    counterparty(_key), method,
                                                                    place, with_whom, label,
                                                                    amount/date spans in the page

Facts are derived data: one source of truth stays the document. An edited or
pruned file takes its document, and so its facts, with it (ON DELETE CASCADE).
Every amount and date carries its span in the masked page text: provenance.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from localmind.folders import fold

FACTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS fact_runs (
    document_id     INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    model           TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('done', 'incomplete', 'failed', 'no_candidates')),
    capped          INTEGER NOT NULL DEFAULT 0,
    facts           INTEGER NOT NULL,
    rejected        INTEGER NOT NULL,
    at              TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fact_pages (
    document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_no      INTEGER NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('done', 'failed', 'no_candidates')),
    PRIMARY KEY (document_id, page_no)
);
CREATE TABLE IF NOT EXISTS facts (
    id                INTEGER PRIMARY KEY,
    document_id       INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_no           INTEGER NOT NULL,
    kind              TEXT NOT NULL CHECK (kind IN ('payment', 'due', 'appointment', 'bank_movement')),
    amount_cents      INTEGER,
    currency          TEXT,
    date              TEXT,
    day_of_month      INTEGER,
    time              TEXT,
    counterparty      TEXT,
    counterparty_key  TEXT NOT NULL DEFAULT '',
    method            TEXT,
    place             TEXT,
    with_whom         TEXT,
    label             TEXT,
    amount_start      INTEGER,
    amount_end        INTEGER,
    date_start        INTEGER,
    date_end          INTEGER
);
CREATE INDEX IF NOT EXISTS idx_facts_document ON facts(document_id);
CREATE INDEX IF NOT EXISTS idx_facts_kind_date ON facts(kind, date);
CREATE INDEX IF NOT EXISTS idx_facts_counterparty ON facts(counterparty_key);
"""

FACT_TABLES = ("facts", "fact_pages", "fact_runs")

_LEGAL_FORMS = {"sa", "sas", "sarl", "sasu", "eurl", "se", "sci", "snc"}

PageStatus = Literal["done", "failed", "no_candidates"]


def counterparty_key(name: str | None) -> str:
    """Matching form of a counterparty: "EDF SA" -> "edf"."""
    words = re.sub(r"[^\w]+", " ", fold(name or "")).split()
    while len(words) > 1 and words[-1] in _LEGAL_FORMS:
        words.pop()
    return " ".join(words)


@dataclass(frozen=True)
class NewFact:
    page_no: int
    kind: str
    amount_cents: int | None
    currency: str | None
    date: str | None  # ISO
    day_of_month: int | None
    time: str | None  # "14:30"
    counterparty: str | None
    method: str | None
    place: str | None
    with_whom: str | None
    label: str | None
    amount_span: tuple[int, int] | None
    date_span: tuple[int, int] | None


@dataclass(frozen=True)
class PageResult:
    page_no: int
    status: PageStatus
    facts: tuple[NewFact, ...]


@dataclass(frozen=True)
class FactRun:
    model: str
    prompt_version: str
    status: str
    capped: bool


@dataclass(frozen=True)
class FactRow:
    id: int
    document_id: int
    page_no: int
    kind: str  # as stored (the query decides the effective kind)
    amount_cents: int | None
    currency: str | None
    date: str | None
    day_of_month: int | None
    time: str | None
    counterparty: str | None
    counterparty_key: str
    method: str | None
    place: str | None
    with_whom: str | None
    label: str | None


_FACT_COLUMNS = (
    "id, document_id, page_no, kind, amount_cents, currency, date, day_of_month, time, "
    "counterparty, counterparty_key, method, place, with_whom, label"
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _run_status(statuses: list[str], capped: bool) -> str:
    if capped:
        return "incomplete"
    if statuses and all(s == "failed" for s in statuses):
        return "failed"
    if "failed" in statuses:
        return "incomplete"
    if all(s == "no_candidates" for s in statuses):
        return "no_candidates"
    return "done"


class FactStore:
    """Mixin for ``Catalog``: needs ``self._conn``."""

    def documents_to_extract(self, model: str, prompt_version: str, redo: bool = False) -> list[int]:
        """Documents with usable text and no run for ``model`` + ``prompt_version``,
        or whose current run has failed pages and did not hit the cap
        (all of them if ``redo``)."""
        stale = "1" if redo else """(
            NOT EXISTS (SELECT 1 FROM fact_runs r WHERE r.document_id = d.id
                        AND r.model = :model AND r.prompt_version = :version)
            OR EXISTS (SELECT 1 FROM fact_runs r WHERE r.document_id = d.id AND r.capped = 0
                       AND r.status IN ('incomplete', 'failed')
                       AND EXISTS (SELECT 1 FROM fact_pages p WHERE p.document_id = d.id
                                   AND p.status = 'failed')))"""
        rows = self._conn.execute(  # type: ignore[attr-defined]
            f"""SELECT d.id FROM documents d
                WHERE EXISTS (SELECT 1 FROM files f WHERE f.document_id = d.id AND f.status = 'indexed')
                  AND EXISTS (SELECT 1 FROM pages p WHERE p.document_id = d.id AND p.status = 'ok')
                  AND {stale}
                ORDER BY d.id""",
            {"model": model, "version": prompt_version},
        )
        return [r[0] for r in rows]

    def fact_run(self, document_id: int) -> FactRun | None:
        row = self._conn.execute(  # type: ignore[attr-defined]
            "SELECT model, prompt_version, status, capped FROM fact_runs WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        return FactRun(row[0], row[1], row[2], bool(row[3])) if row else None

    def failed_pages(self, document_id: int) -> list[int]:
        rows = self._conn.execute(  # type: ignore[attr-defined]
            "SELECT page_no FROM fact_pages WHERE document_id = ? AND status = 'failed' ORDER BY page_no",
            (document_id,),
        )
        return [r[0] for r in rows]

    def fact_rows(
        self,
        document_ids: Collection[int] | None = None,
        counterparty: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        dated_only: bool = False,
    ) -> list[FactRow]:
        """Stored facts, filtered in SQL. ``counterparty`` is a substring of the
        matching key; dates are ISO strings, both ends included."""
        where, args = ["1"], []
        if document_ids is not None:
            ids = sorted(document_ids)
            where.append(f"document_id IN ({', '.join('?' * len(ids))})" if ids else "0")
            args.extend(ids)
        if counterparty:
            where.append("instr(counterparty_key, ?) > 0")
            args.append(counterparty)
        if dated_only:
            where.append("date IS NOT NULL")
        if date_from:
            where.append("date >= ?")
            args.append(date_from)
        if date_to:
            where.append("date <= ?")
            args.append(date_to)
        rows = self._conn.execute(  # type: ignore[attr-defined]
            f"SELECT {_FACT_COLUMNS} FROM facts WHERE {' AND '.join(where)} ORDER BY id", args
        )
        return [FactRow(*r) for r in rows]

    def coverage_counts(self, prompt_version: str, document_ids: Collection[int] | None = None) -> tuple[int, int, int]:
        """``(missing, incomplete, failed)`` among documents with usable text:
        missing = no run for ``prompt_version`` (any model)."""
        rows = self._conn.execute(  # type: ignore[attr-defined]
            """SELECT d.id, r.prompt_version, r.status FROM documents d
               LEFT JOIN fact_runs r ON r.document_id = d.id
               WHERE EXISTS (SELECT 1 FROM files f WHERE f.document_id = d.id AND f.status = 'indexed')
                 AND EXISTS (SELECT 1 FROM pages p WHERE p.document_id = d.id AND p.status = 'ok')"""
        )
        missing = incomplete = failed = 0
        for doc_id, version, status in rows:
            if document_ids is not None and doc_id not in document_ids:
                continue
            if version != prompt_version:
                missing += 1
            elif status == "incomplete":
                incomplete += 1
            elif status == "failed":
                failed += 1
        return missing, incomplete, failed

    def fact_page_numbers(self, document_id: int) -> list[int]:
        """Page number of each stored fact of the document (one entry per fact)."""
        rows = self._conn.execute(  # type: ignore[attr-defined]
            "SELECT page_no FROM facts WHERE document_id = ?", (document_id,)
        )
        return [r[0] for r in rows]

    def store_extraction(
        self,
        document_id: int,
        model: str,
        prompt_version: str,
        pages: Sequence[PageResult],
        rejected: int,
        capped: bool,
    ) -> None:
        """Store one document's extraction in one transaction. Only the given
        pages are replaced; a run for another model or prompt version is
        started afresh (its pages and facts are dropped)."""
        conn = self._conn  # type: ignore[attr-defined]
        with conn:
            current = self.fact_run(document_id)
            same_run = current is not None and (current.model, current.prompt_version) == (model, prompt_version)
            if not same_run:
                conn.execute("DELETE FROM facts WHERE document_id = ?", (document_id,))
                conn.execute("DELETE FROM fact_pages WHERE document_id = ?", (document_id,))
            else:
                rejected += conn.execute(
                    "SELECT rejected FROM fact_runs WHERE document_id = ?", (document_id,)
                ).fetchone()[0]
                capped = capped or current.capped
            for page in pages:
                conn.execute(
                    "DELETE FROM facts WHERE document_id = ? AND page_no = ?", (document_id, page.page_no)
                )
                conn.execute(
                    "INSERT OR REPLACE INTO fact_pages (document_id, page_no, status) VALUES (?, ?, ?)",
                    (document_id, page.page_no, page.status),
                )
                for f in page.facts:
                    self._insert_fact(document_id, f)
            statuses = [r[0] for r in conn.execute(
                "SELECT status FROM fact_pages WHERE document_id = ?", (document_id,)
            )]
            total = conn.execute(
                "SELECT COUNT(*) FROM facts WHERE document_id = ?", (document_id,)
            ).fetchone()[0]
            conn.execute(
                """INSERT OR REPLACE INTO fact_runs
                   (document_id, model, prompt_version, status, capped, facts, rejected, at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (document_id, model, prompt_version, _run_status(statuses, capped),
                 int(capped), total, rejected, _now()),
            )

    def _insert_fact(self, document_id: int, f: NewFact) -> None:
        amount = f.amount_span or (None, None)
        when = f.date_span or (None, None)
        self._conn.execute(  # type: ignore[attr-defined]
            """INSERT INTO facts (document_id, page_no, kind, amount_cents, currency, date,
                   day_of_month, time, counterparty, counterparty_key, method, place, with_whom,
                   label, amount_start, amount_end, date_start, date_end)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (document_id, f.page_no, f.kind, f.amount_cents, f.currency, f.date, f.day_of_month,
             f.time, f.counterparty, counterparty_key(f.counterparty), f.method, f.place,
             f.with_whom, f.label, amount[0], amount[1], when[0], when[1]),
        )
