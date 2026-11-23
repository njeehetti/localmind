"""Stage 4 CLI: index after ingest, ``index``, ``search``, ``prune``, stats."""

import builtins
import re
import sqlite3
from pathlib import Path

import pytest

import localmind
from localmind import cli, cli_folders, cli_search
from localmind.catalog import Catalog
from localmind.cli import main
from tests.fake_classifier import FakeClassifier, answer
from tests.stage4_data import IBAN


@pytest.fixture
def env(tmp_path, monkeypatch):
    state = {
        "tty": False,
        "answers": [],
        "classifier": FakeClassifier(lambda payload: answer("Divers/Autres", new_folder=True)),
        "embedder": FakeClassifier(),
    }

    def scripted_input(prompt=""):
        if not state["answers"]:
            raise AssertionError(f"unexpected prompt: {prompt!r}")
        return state["answers"].pop(0)

    monkeypatch.setattr(cli_folders, "make_classifier", lambda: state["classifier"])
    monkeypatch.setattr(cli_search, "make_embedder", lambda: state["embedder"])
    monkeypatch.setattr(cli, "_is_interactive", lambda: state["tty"])
    monkeypatch.setattr(builtins, "input", scripted_input)
    state["db"] = str(tmp_path / "c.db")
    return state


@pytest.fixture
def inbox(tmp_path):
    folder = tmp_path / "inbox"
    folder.mkdir()
    (folder / "abonnement_tel.txt").write_text(
        "Abonnement telephone mobile : prélèvement le 5 de chaque mois.", encoding="utf-8"
    )
    (folder / "bail.txt").write_text("Le bail de location est consenti pour trois ans.", encoding="utf-8")
    return folder


def run(env, *args):
    return main(["--db", env["db"], *args])


def test_ingest_classifies_then_indexes(env, inbox, capsys):
    assert run(env, "ingest", str(inbox)) == 0
    out = capsys.readouterr().out
    assert "document(s) classified:" in out
    assert out.rstrip().endswith("Indexed 2 document(s): 2 chunk(s)")
    assert out.index("classified:") < out.index("Indexed 2")


def test_ingest_no_index_skips_indexing(env, inbox, capsys):
    run(env, "ingest", str(inbox), "--no-classify", "--no-index")
    assert "Indexed" not in capsys.readouterr().out
    assert env["embedder"].embedded == []


def test_ingest_with_ollama_down_prints_one_hint(env, inbox, capsys):
    env["classifier"] = FakeClassifier(unavailable=True)
    env["embedder"] = FakeClassifier(unavailable=True)
    assert run(env, "ingest", str(inbox)) == 0
    out = capsys.readouterr().out
    assert out.count("Ollama not reachable") == 1
    assert (
        "Ollama not reachable at http://fake: start it (ollama serve), "
        "then: localmind classify; localmind index"
    ) in out


def test_ingest_no_classify_still_indexes_and_hints_alone(env, inbox, capsys):
    env["embedder"] = FakeClassifier(unavailable=True)
    run(env, "ingest", str(inbox), "--no-classify")
    out = capsys.readouterr().out
    assert out.count("Ollama not reachable") == 1 and out.rstrip().endswith("then: localmind index")


def test_index_command_and_all_flag(env, inbox, capsys):
    run(env, "ingest", str(inbox), "--no-classify", "--no-index")
    capsys.readouterr()
    assert run(env, "index") == 0
    assert capsys.readouterr().out == "Indexed 2 document(s): 2 chunk(s)\n"
    run(env, "index")
    assert capsys.readouterr().out == "Indexed 0 document(s): 0 chunk(s)\n"
    run(env, "index", "--all")
    assert capsys.readouterr().out == "Indexed 2 document(s): 2 chunk(s)\n"


def test_search_prints_score_file_page_excerpt(env, inbox, capsys):
    run(env, "ingest", str(inbox), "--no-classify")
    capsys.readouterr()
    assert run(env, "search", "jour du prélèvement telephone", "-k", "1") == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == ['1.00  abonnement_tel.txt p.1  "Abonnement telephone mobile : prélèvement le 5 de chaque mois."']


def test_long_excerpt_is_cut_at_100_characters(env, inbox, capsys):
    (inbox / "long.txt").write_text("Contrat telephone " + "mot " * 60, encoding="utf-8")
    run(env, "ingest", str(inbox), "--no-classify")
    capsys.readouterr()
    run(env, "search", "telephone", "-k", "3")
    line = next(l for l in capsys.readouterr().out.splitlines() if "long.txt" in l)
    excerpt = re.search(r'"(.*)"$', line).group(1)
    assert len(excerpt) == 101 and excerpt.endswith("…")


def test_search_output_masks_sensitive_values(env, inbox, capsys):
    (inbox / "rib.txt").write_text(f"Coordonnées bail : IBAN {IBAN} pour le loyer.", encoding="utf-8")
    run(env, "ingest", str(inbox), "--no-classify")
    capsys.readouterr()
    run(env, "search", "bail", "-k", "5")
    out = capsys.readouterr().out
    assert "[IBAN]" in out and "3000" not in out and "7890" not in out


def test_search_with_nothing_indexed(env, capsys):
    assert run(env, "search", "bail") == 0
    assert capsys.readouterr().out == "Nothing indexed yet: localmind index\n"


def test_ollama_down_prints_warning_once_and_keyword_hits(env, inbox, capsys):
    # stage 5 replaces stage 4's "exit 2": keywords still answer
    run(env, "ingest", str(inbox), "--no-classify")
    env["embedder"] = FakeClassifier(unavailable=True)
    capsys.readouterr()
    assert run(env, "search", "bail de location") == 0
    out, err = capsys.readouterr()
    assert err == (
        "meaning search unavailable (Ollama not reachable at http://fake: start it "
        "(ollama serve)): keyword results only\n"
    )
    assert out == '  --  bail.txt p.1  "Le bail de location est consenti pour trois ans."\n'


def _classified(env, inbox, capsys):
    """Ingest with the scripted LLM: every document goes to Divers/Autres, to confirm."""
    (inbox / "lettre.txt").write_text("Une lettre de location sans rapport.", encoding="utf-8")
    run(env, "ingest", str(inbox))
    capsys.readouterr()


def test_search_prints_folder_and_mark(env, inbox, capsys):
    _classified(env, inbox, capsys)
    assert run(env, "search", "bail de location", "-k", "1") == 0
    assert capsys.readouterr().out == (
        '1.00  bail.txt p.1  Divers/Autres (to confirm)  "Le bail de location est consenti pour trois ans."\n'
    )


def test_unclassified_hit_has_no_folder_column(env, inbox, capsys):
    run(env, "ingest", str(inbox), "--no-classify")
    capsys.readouterr()
    run(env, "search", "bail", "-k", "1")
    assert capsys.readouterr().out == '1.00  bail.txt p.1  "Le bail de location est consenti pour trois ans."\n'


def test_folder_option_limits_results(env, inbox, capsys):
    run(env, "ingest", str(inbox), "--no-classify")
    with Catalog(Path(env["db"])) as catalog:
        from localmind.catalog import Classification
        from localmind.folders import FolderName

        doc = catalog.document_for_path((inbox / "abonnement_tel.txt").resolve())
        folder_id = catalog.ensure_folder(FolderName("Maison", "Abonnements"), "user", True)
        catalog.set_classification(Classification(doc.id, folder_id, "placed", "user", 1.0, None, None), "m")
    capsys.readouterr()
    assert run(env, "search", "bail", "--folder", "maison / abonnements") == 0
    out = capsys.readouterr().out
    assert out.splitlines() == [
        '0.00  abonnement_tel.txt p.1  Maison/Abonnements  '
        '"Abonnement telephone mobile : prélèvement le 5 de chaque mois."'
    ]


def test_unknown_folder_exits_2(env, inbox, capsys):
    _classified(env, inbox, capsys)
    assert run(env, "search", "bail", "--folder", "maison/factur") == 2
    out, err = capsys.readouterr()
    assert out == "" and err == "error: no folder named 'Maison/Factur'\n"


def test_invalid_folder_exits_2(env, inbox, capsys):
    _classified(env, inbox, capsys)
    assert run(env, "search", "bail", "--folder", "a/b/c") == 2
    assert capsys.readouterr().err.startswith("error: invalid folder name")


def test_search_busy_catalog_exits_2(env, inbox, capsys, monkeypatch):
    run(env, "ingest", str(inbox), "--no-classify")
    capsys.readouterr()

    def locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(cli_search.VectorIndex, "search", locked)
    assert run(env, "search", "bail") == 2
    assert capsys.readouterr().err == "error: the catalog is busy, try again\n"


def test_prune_asks_then_forgets(env, inbox, capsys):
    run(env, "ingest", str(inbox), "--no-classify")
    (inbox / "bail.txt").unlink()
    capsys.readouterr()
    env["tty"], env["answers"] = True, ["y"]
    assert run(env, "prune") == 0
    out = capsys.readouterr().out
    assert out == (
        "1 file(s) no longer exist:\n"
        f"  {(inbox / 'bail.txt').resolve()}\n"
        "Forgot 1 file(s) and 1 document(s).\n"
    )
    run(env, "prune")
    assert capsys.readouterr().out == "Nothing to prune.\n"


def test_prune_later_keeps_everything(env, inbox, capsys):
    run(env, "ingest", str(inbox), "--no-classify")
    (inbox / "bail.txt").unlink()
    env["tty"], env["answers"] = True, [""]
    run(env, "prune")
    with Catalog(Path(env["db"])) as catalog:
        assert catalog.stats().documents == 2


def test_prune_non_interactive_needs_yes(env, inbox, capsys):
    run(env, "ingest", str(inbox), "--no-classify")
    (inbox / "bail.txt").unlink()
    capsys.readouterr()
    run(env, "prune")
    assert capsys.readouterr().out == "1 file(s) no longer exist: localmind prune --yes\n"
    with Catalog(Path(env["db"])) as catalog:
        assert catalog.stats().documents == 2
    run(env, "prune", "--yes")
    assert capsys.readouterr().out.endswith("Forgot 1 file(s) and 1 document(s).\n")


def test_stats_shows_chunks_and_indexed_docs(env, inbox, capsys):
    run(env, "ingest", str(inbox), "--no-classify")
    capsys.readouterr()
    run(env, "stats")
    out = capsys.readouterr().out
    assert "chunks:          2\n" in out and "indexed docs:    2\n" in out


def test_public_api_exports_stage4_and_stage5_names():
    for name in ("index", "IndexReport", "search", "Hit", "VectorIndex", "CatalogBusy",
                 "prune", "missing_files", "ChunkConfig", "DocFolder", "FolderFilter", "UnknownFolder"):
        assert name in localmind.__all__ and hasattr(localmind, name)
    assert localmind.__version__ == "0.4.0"
