"""Stage 3 CLI: classification summary after ingest, review, folder commands."""

import builtins
from pathlib import Path

import pytest

from localmind import cli, cli_folders
from localmind.catalog import Catalog
from localmind.cli import main
from tests.fake_classifier import FakeClassifier, answer

PROPOSAL_CHOICES = "  [y] yes  [r] rename  [o] other existing folder  [l] later "


def invoice(i):
    return f"Facture EDF numero {i}, consommation {400 + i} kWh, montant total {80 + i} euros TTC."


def payslip(i):
    return f"Bulletin de paie numero {i}, salaire de base 2100 euros, net a payer {1600 + i} euros."


def by_keyword(payload):
    excerpt = payload["excerpt"].casefold()
    if "facture" in excerpt:
        return answer("Maison/Factures", new_folder=True)
    if "paie" in excerpt:
        return answer("Travail/Bulletins de paie", new_folder=True)
    return answer("Divers/Autres", new_folder=True)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Not interactive, a keyword-driven fake LLM; override per test."""
    state = {"tty": False, "answers": [], "prompts": [], "fake": FakeClassifier(by_keyword)}

    def scripted_input(prompt=""):
        state["prompts"].append(prompt)
        if not state["answers"]:
            raise AssertionError(f"unexpected prompt: {prompt!r}")
        reply = state["answers"].pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    monkeypatch.setattr(cli_folders, "make_classifier", lambda: state["fake"])
    monkeypatch.setattr(cli, "_is_interactive", lambda: state["tty"])
    monkeypatch.setattr(builtins, "input", scripted_input)
    state["db"] = str(tmp_path / "c.db")
    return state


@pytest.fixture
def inbox(tmp_path):
    folder = tmp_path / "inbox"
    folder.mkdir()
    return folder


def run(env, *args):
    return main(["--db", env["db"], *args])


def write(inbox, name, text):
    (inbox / name).write_text(text, encoding="utf-8")
    return inbox / name


def pending(env):
    with Catalog(Path(env["db"])) as catalog:
        return len(catalog.pending_reviews())


def test_ingest_prints_folder_per_document_and_summary(env, inbox, capsys):
    for i in range(3):
        write(inbox, f"seed_{i}.txt", invoice(i))
    run(env, "ingest", str(inbox), "--no-classify")
    for i in range(3):
        assert run(env, "move", str(inbox / f"seed_{i}.txt"), "Maison/Factures") == 0
    write(inbox, "d.txt", invoice(9))
    write(inbox, "e.txt", "Contrat de location du garage, signe en mars.")
    env["fake"] = FakeClassifier(lambda p: answer("maison/facture", new_folder=True))
    capsys.readouterr()
    assert run(env, "ingest", str(inbox)) == 0
    out = capsys.readouterr().out
    assert "  d.txt  → Maison/Factures\n" in out
    assert "  e.txt  → Maison/Factures\n" in out
    assert "  (e.txt: proposed Maison/Facture, used existing Maison/Factures)" in out
    assert "2 document(s) classified:\n  2 placed  Maison/Factures (2)\n" in out
    assert "to confirm" not in out


def test_ingest_asks_to_confirm_and_review_confirms_a_proposal(env, inbox, capsys):
    write(inbox, "facture_0.txt", invoice(0))
    write(inbox, "facture_1.txt", invoice(1))
    env["tty"], env["answers"] = True, ["y", "y"]
    assert run(env, "ingest", str(inbox)) == 0
    out = capsys.readouterr().out
    assert "  facture_0.txt  → Maison/Factures (proposed)" in out
    assert "  2 to confirm  new folder proposed: Maison/Factures (2 document(s))" in out
    assert "New folder Maison/Factures for 2 document(s): facture_0.txt, facture_1.txt" in out
    assert env["prompts"] == ["Confirm the 2 now? [y/N] ", PROPOSAL_CHOICES]
    assert pending(env) == 0
    run(env, "folders")
    assert capsys.readouterr().out == "Maison\n  Factures (2)\n"


def test_review_rename_and_other(env, inbox, capsys):
    write(inbox, "facture_0.txt", invoice(0))
    write(inbox, "facture_1.txt", invoice(1))
    write(inbox, "paie.txt", payslip(1))
    run(env, "ingest", str(inbox))
    env["tty"], env["answers"] = True, ["r", "Maison/Énergie", "o", "Santé/Contrats"]
    assert run(env, "folders", "review") == 0
    assert env["prompts"] == [PROPOSAL_CHOICES, "New name: ", PROPOSAL_CHOICES, "Folder: "]
    capsys.readouterr()
    run(env, "folders")
    assert capsys.readouterr().out == "Maison\n  Énergie (2)\nSanté\n  Contrats (1)\n"


def test_hesitation_answer_two_picks_the_second(env, inbox, capsys):
    write(inbox, "seed.txt", invoice(0))
    write(inbox, "paie.txt", payslip(0))
    run(env, "ingest", str(inbox), "--no-classify")
    run(env, "move", str(inbox / "seed.txt"), "Maison/Factures")
    run(env, "move", str(inbox / "paie.txt"), "Travail/Bulletins de paie")
    write(inbox, "new.txt", invoice(5))
    env["fake"] = FakeClassifier([answer("Travail/Bulletins de paie")])
    capsys.readouterr()
    run(env, "ingest", str(inbox))
    out = capsys.readouterr().out
    assert "  1 hesitation  new.txt → Travail/Bulletins de paie or Maison/Factures?" in out
    assert "1 classification(s) to confirm: localmind folders review" in out
    env["tty"], env["answers"] = True, ["2"]
    run(env, "folders", "review")
    assert "new.txt\n" in capsys.readouterr().out
    assert env["prompts"] == ["  [1] Travail/Bulletins de paie  [2] Maison/Factures  [o] other  [l] later "]
    run(env, "folders", "show", "Maison/Factures")
    out = capsys.readouterr().out
    assert out.startswith("Maison/Factures (2)\n") and "new.txt" in out


def test_later_and_ctrl_c_keep_the_queue(env, inbox, capsys):
    write(inbox, "facture_0.txt", invoice(0))
    write(inbox, "facture_1.txt", invoice(1))
    env["tty"], env["answers"] = True, [EOFError()]
    assert run(env, "ingest", str(inbox)) == 0
    env["answers"] = ["l"]
    assert run(env, "folders", "review") == 0
    env["answers"] = [KeyboardInterrupt()]
    assert run(env, "folders", "review") == 0
    assert pending(env) == 2 and env["answers"] == []


def test_non_interactive_prints_one_line_and_never_asks(env, inbox, capsys):
    write(inbox, "facture_0.txt", invoice(0))
    write(inbox, "facture_1.txt", invoice(1))
    assert run(env, "ingest", str(inbox)) == 0
    assert "2 classification(s) to confirm: localmind folders review" in capsys.readouterr().out
    assert run(env, "folders", "review") == 0
    out = capsys.readouterr().out
    assert "New folder Maison/Factures for 2 document(s): facture_0.txt, facture_1.txt" in out
    assert env["prompts"] == []


def test_no_classify_skips_the_step(env, inbox, capsys):
    write(inbox, "a.txt", invoice(0))
    assert run(env, "ingest", str(inbox), "--no-classify") == 0
    assert "classified" not in capsys.readouterr().out
    assert env["fake"].embedded == []
    run(env, "folders")
    assert capsys.readouterr().out == "Unsorted (1)\n"


def test_ollama_absent_ingest_still_succeeds(env, inbox, capsys):
    write(inbox, "a.txt", invoice(0))
    write(inbox, "b.txt", payslip(0))
    env["fake"] = FakeClassifier(unavailable=True)
    assert run(env, "ingest", str(inbox)) == 0
    out = capsys.readouterr().out
    assert "  2 unsorted  " in out
    assert "Ollama not reachable at http://fake: start it (ollama serve), then: localmind classify" in out
    run(env, "folders")
    assert capsys.readouterr().out == "Unsorted (2)\n"


def test_folders_tree_counts_and_marks_proposed(env, inbox, capsys):
    write(inbox, "a.txt", invoice(0))
    run(env, "ingest", str(inbox), "--no-classify")
    run(env, "move", str(inbox / "a.txt"), "Maison/Factures")
    write(inbox, "b.txt", payslip(0))
    run(env, "ingest", str(inbox))
    write(inbox, "c.txt", "Notes de voyage en Bretagne, juillet.")
    run(env, "ingest", str(inbox), "--no-classify")
    capsys.readouterr()
    assert run(env, "folders") == 0
    assert capsys.readouterr().out == (
        "Maison\n  Factures (1)\nTravail\n  Bulletins de paie (1) (proposed)\nUnsorted (1)\n"
    )


def test_move_and_rename_commands(env, inbox, capsys):
    write(inbox, "a.txt", invoice(0))
    run(env, "ingest", str(inbox), "--no-classify")
    capsys.readouterr()
    assert run(env, "move", str(inbox / "a.txt"), "Maison/Factures/EDF") == 2
    assert "error:" in capsys.readouterr().err
    assert run(env, "move", str(inbox / "nope.txt"), "Maison/Factures") == 2
    assert "error:" in capsys.readouterr().err
    assert run(env, "move", str(inbox / "a.txt"), "maison/factures") == 0
    assert capsys.readouterr().out == "  a.txt  → Maison/Factures (your choice)\n"
    assert run(env, "folders", "rename", "Maison/Inconnu", "Maison/X") == 2
    assert run(env, "folders", "rename", "Maison/Factures", "Maison/Énergie") == 0
    capsys.readouterr()
    run(env, "folders")
    assert capsys.readouterr().out == "Maison\n  Énergie (1)\n"
    assert run(env, "folders", "show", "Maison/Inconnu") == 2


def test_classify_command_all_keeps_user_choices(env, inbox, capsys):
    write(inbox, "a.txt", invoice(0))
    write(inbox, "b.txt", invoice(1))
    run(env, "ingest", str(inbox), "--no-classify")
    run(env, "move", str(inbox / "a.txt"), "Maison/Factures")
    env["fake"] = FakeClassifier(lambda p: answer("Divers/Autres", new_folder=True))
    assert run(env, "classify") == 0
    env["fake"] = FakeClassifier(lambda p: answer("Culture/Livres", new_folder=True))
    capsys.readouterr()
    assert run(env, "classify", "--all") == 0
    out = capsys.readouterr().out
    assert "  b.txt  → Culture/Livres (proposed)" in out
    assert "1 document(s) classified:" in out and "a.txt" not in out
    run(env, "folders", "show", "Maison/Factures")
    assert "a.txt" in capsys.readouterr().out


def test_redirected_output_in_a_legacy_encoding_does_not_crash(tmp_path, inbox):
    import os
    import subprocess
    import sys

    write(inbox, "a.txt", invoice(0))
    db = tmp_path / "x.db"
    base = [sys.executable, "-m", "localmind.cli", "--db", str(db)]
    subprocess.run([*base, "ingest", str(inbox), "--no-classify"], capture_output=True, check=True)
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    out = subprocess.run(
        [*base, "move", str(inbox / "a.txt"), "Maison/Factures"], capture_output=True, env=env
    )
    assert out.returncode == 0, out.stderr.decode(errors="replace")
    assert "→ Maison/Factures (your choice)" in out.stdout.decode("utf-8")
