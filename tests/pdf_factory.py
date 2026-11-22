"""Build tiny, valid PDF files by hand for tests.

Writing a PDF manually is the best way to understand why PDF text
extraction is hard: a PDF does not contain "text", it contains drawing
instructions ("move to x,y; draw these glyphs with font F1"). Extractors
have to rebuild lines, words and reading order from those instructions.
"""

from __future__ import annotations


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _content_stream(text: str) -> bytes:
    ops = ["BT", "/F1 12 Tf", "14 TL", "72 720 Td"]
    ops += [f"({_escape(line)}) Tj T*" for line in text.split("\n") if line]
    ops.append("ET")
    return "\n".join(ops).encode("latin-1")


def make_pdf(pages: list[str]) -> bytes:
    """Return the bytes of a PDF with one page per string (empty string = no text)."""
    page_ids = [4 + 2 * i for i in range(len(pages))]
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)

    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode(),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for pid, text in zip(page_ids, pages):
        cid = pid + 1
        objects[pid] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {cid} 0 R >>"
        ).encode()
        stream = _content_stream(text)
        objects[cid] = (
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for num in sorted(objects):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode() + objects[num] + b"\nendobj\n"

    xref_at = len(out)
    size = max(objects) + 1
    out += f"xref\n0 {size}\n".encode() + b"0000000000 65535 f \n"
    for num in range(1, size):
        out += f"{offsets[num]:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return bytes(out)
