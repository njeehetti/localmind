import pytest

from localmind.folders import FolderName, InvalidFolderName, fold, parse_folder


def test_parse_folder_normalises():
    assert str(parse_folder("  maison /  factures  edf ")) == "Maison/Factures edf"
    assert parse_folder("Santé/Remboursements") == FolderName("Santé", "Remboursements")


def test_parse_folder_keeps_allowed_punctuation():
    assert str(parse_folder("Loisirs & sport/Billets d'avion - train")) == "Loisirs & sport/Billets d'avion - train"


@pytest.mark.parametrize(
    "bad",
    ["Maison", "A/B/C", "/Factures", "Maison/", "Maison/F" + "x" * 40, "Maison/Fac<tures>", ""],
)
def test_parse_folder_rejects(bad):
    with pytest.raises(InvalidFolderName):
        parse_folder(bad)


def test_invalid_folder_name_is_a_value_error():
    assert issubclass(InvalidFolderName, ValueError)


def test_fold_ignores_case_and_accents():
    assert fold("Santé") == fold("SANTE") == "sante"
    assert fold("  Électricité ") == "electricite"
