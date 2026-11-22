"""What the user does with folders: confirm a proposal, answer a hesitation,
move a document, rename or merge folders.

Every answer is a **user decision**: it is recorded per file (hash + path) in
``folder_decisions``, so it survives re-ingest, follows exact copies and files
edited in place, and is never overridden by ``classify``. The user's own
documents also count double as examples for the next ones.
"""

from __future__ import annotations

from pathlib import Path

from localmind.catalog import Catalog, Classification, FileStatus
from localmind.folders import FolderName, parse_folder


def _decide(catalog: Catalog, document_id: int, folder_id: int) -> None:
    """Record the user's choice for every file of the document, and place it."""
    name = catalog.folder(folder_id).name
    doc = catalog.document_to_classify(document_id)
    for file_hash, path in zip(doc.file_hashes, doc.paths):
        catalog.record_folder_decision(file_hash, path, name)
    previous = catalog.classification(document_id)
    score = previous.score if previous is not None else None
    catalog.set_classification(
        Classification(document_id, folder_id, "placed", "user", score, None, None), "user"
    )


def _existing(catalog: Catalog, folder: str):
    found = catalog.folder_by_name(parse_folder(folder))
    if found is None:
        raise LookupError(f"no folder named {folder}")
    return found


def confirm(catalog: Catalog, folder: str) -> int:
    """Confirm a proposed folder: all documents waiting in it are placed there."""
    target = _existing(catalog, folder)
    catalog.ensure_folder(target.name, target.created_by, True)
    waiting = [
        c.document_id
        for c in catalog.pending_reviews()
        if c.folder_id == target.id
    ]
    for document_id in waiting:
        _decide(catalog, document_id, target.id)
    catalog.delete_empty_llm_folders()
    return len(waiting)


def choose_folder(catalog: Catalog, document_id: int, folder: str) -> None:
    """The user's answer for one document (a hesitation, or "another folder")."""
    folder_id = catalog.ensure_folder(parse_folder(folder), "user", True)
    _decide(catalog, document_id, folder_id)
    catalog.delete_empty_llm_folders()


def move(catalog: Catalog, path: Path, folder: str) -> None:
    """Put the document of an indexed file into ``folder`` (created if needed)."""
    name = parse_folder(folder)  # check the name before anything else
    record = catalog.file_by_path(path.resolve())
    if record is None or record.status is not FileStatus.INDEXED or record.document_id is None:
        raise LookupError(f"not an indexed file: {path}")
    choose_folder(catalog, record.document_id, str(name))


def rename_folder(catalog: Catalog, old: str, new: str) -> None:
    """Rename a folder, or merge it into an existing one. Its documents become
    the user's choice (the user named their folder)."""
    source = _existing(catalog, old)
    new_name: FolderName = parse_folder(new)
    target = catalog.folder_by_name(new_name)
    if target is None or target.id == source.id:
        catalog.rename_folder_row(source.id, new_name)
        final_id, final_name = source.id, new_name
    else:
        catalog.merge_folders(source.id, target.id)
        final_id, final_name = target.id, target.name
    catalog.rename_folder_decisions(source.name, final_name)
    for doc in catalog.documents_in(final_id):
        c = catalog.classification(doc.id)
        # Open hesitations stay open: the user answered about the folder's
        # name, not about these documents.
        if c is not None and (c.status == "placed" or c.alternative_folder_id is None):
            _decide(catalog, doc.id, final_id)
    catalog.delete_empty_llm_folders()
