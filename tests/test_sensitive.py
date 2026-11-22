from localmind.sensitive import find_sensitive, iban_is_valid, luhn_is_valid, nir_is_valid


def kinds(text):
    return [s.kind for s in find_sensitive(text)]


def test_iban_checksum():
    assert iban_is_valid("FR7630006000011234567890189")
    assert not iban_is_valid("FR7630006000011234567890188")


def test_iban_with_spaces_and_lowercase_is_found():
    assert kinds("IBAN : fr76 3000 6000 0112 3456 7890 189") == ["iban"]


def test_luhn():
    assert luhn_is_valid("4539578763621486")
    assert not luhn_is_valid("4539578763621487")


def test_card_found_with_spaces_or_dashes():
    assert kinds("Carte 4539 5787 6362 1486") == ["card"]
    assert kinds("Carte 4539-5787-6362-1486") == ["card"]


def test_order_number_failing_luhn_is_not_flagged():
    assert kinds("Commande n° 4539578763621487") == []


def test_date_pairs_on_bank_statements_are_not_cards():  # final review Important #3
    import datetime

    day = datetime.date(2024, 1, 1)
    flagged = []
    for i in range(400):
        a = day + datetime.timedelta(days=i)
        b = a + datetime.timedelta(days=2)
        line = f"{a.isoformat()} {b.isoformat()} PRLV SEPA EDF 84,20"
        if kinds(line):
            flagged.append(line)
    assert flagged == []


def test_spaced_number_lists_are_not_cards():
    assert kinds("Scores : 12 34 56 78 90 11 22 33") == []


def test_card_in_4_6_5_grouping_found():
    assert kinds("Amex 3714 496353 98431") == ["card"]


def test_nir_key():
    assert nir_is_valid("185057800608491")  # 1850578006084, key = 97 - (n mod 97) = 91
    assert not nir_is_valid("185057800608492")


def test_nir_found_in_text():
    assert kinds("N° de sécurité sociale : 1 85 05 78 006 084 91") == ["fr_nir"]


def test_random_15_digits_not_flagged():
    assert kinds("Ref 123456789012345") == []


def test_password_lines():
    assert kinds("login: bob\nmot de passe : hunter2") == ["password"]
    assert kinds("password=abc") == ["password"]
    assert kinds("Mot de passe oublié ? Cliquez ici") == []


def test_span_points_at_the_value():
    text = "IBAN FR7630006000011234567890189 fin"
    (s,) = find_sensitive(text)
    assert text[s.start : s.end].replace(" ", "") == "FR7630006000011234567890189"


def test_password_span_covers_value_only():
    text = "mdp: hunter2"
    (s,) = find_sensitive(text)
    assert text[s.start : s.end] == "hunter2"


def test_spans_sorted_by_position():
    text = "mdp: x1\nIBAN FR7630006000011234567890189"
    assert kinds(text) == ["password", "iban"]
