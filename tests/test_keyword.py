"""Stage 5: from a question to a keyword query, and reciprocal rank fusion (pure)."""

from localmind.keyword import STOP_WORDS, fuse, keyword_query

FTS_SYNTAX = ["l'électricité", 'a "b', "x*", "-y", "NEAR(a b)", "a AND b", "(", "^début", "col:val", "_"]


def test_query_quotes_words_and_drops_stop_words():
    assert keyword_query("Quel est le préavis de mon bail ?") == '"préavis" OR "bail"'


def test_query_keeps_digits_and_lowercases():
    assert keyword_query("Facture 2026-04-5120 EDF") == '"facture" OR "2026" OR "04" OR "5120" OR "edf"'


def test_fts_syntax_is_neutralised():  # Review Focus 1
    for q in FTS_SYNTAX:
        query = keyword_query(q)
        assert query is None or all(
            part.startswith('"') and part.endswith('"') and '"' not in part[1:-1]
            for part in query.split(" OR ")
        ), q


def test_only_stop_words_gives_none():
    assert keyword_query("le de la ?") is None
    assert keyword_query("") is None
    assert {"le", "de", "mon", "quel", "combien"} <= STOP_WORDS


def test_fuse_rewards_both_sides():
    assert fuse([1, 2, 3], [3, 4])[0] == 3


def test_fuse_ties_prefer_vector_rank():
    assert fuse([1, 2], [2, 1]) == [1, 2]


def test_fuse_ties_without_vector_rank_prefer_smaller_id():
    assert fuse([], [5, 4]) == [5, 4]
    assert fuse([9], [7, 8])[:1] == [9]


def test_fuse_keeps_one_sided_hits():
    assert set(fuse([1], [2])) == {1, 2}
    assert fuse([], []) == []


def test_decomposed_accents_are_recomposed():
    import unicodedata

    nfd = unicodedata.normalize("NFD", "électricité")  # macOS paste, some speech-to-text
    assert keyword_query(nfd) == keyword_query("électricité") == '"électricité"'
