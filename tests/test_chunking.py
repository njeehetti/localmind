"""Stage 4: cutting pages into chunks (pure, no I/O)."""

from localmind.chunking import Chunk, ChunkConfig, chunk_pages, embed_text


def test_short_page_is_one_chunk():
    assert chunk_pages([(1, "Facture EDF. Montant : 84,20 €")]) == [
        Chunk(1, 0, "Facture EDF. Montant : 84,20 €")
    ]


def test_chunks_never_span_pages():
    chunks = chunk_pages([(1, "a " * 600), (2, "b " * 600)])
    assert {c.page_no for c in chunks if "a" in c.text} == {1}
    assert {c.page_no for c in chunks if "b" in c.text} == {2}
    assert [c.seq for c in chunks] == list(range(len(chunks)))


def test_cuts_on_paragraphs_then_lines_then_sentences():
    para = "Premier paragraphe. " * 25  # ~500 chars
    chunks = chunk_pages([(1, para + "\n\n" + para)], ChunkConfig(chunk_chars=600, overlap_chars=0))
    assert chunks[0].text.strip() == para.strip()
    ocr = "\n".join(f"Ligne {i} du relevé de compte" for i in range(60))  # no blank line
    chunks = chunk_pages([(1, ocr)])
    assert len(chunks) > 1
    assert all(c.text.split("\n")[-1].startswith("Ligne") for c in chunks)
    assert all(c.text.endswith("compte") for c in chunks)


def test_never_cuts_inside_a_word():
    text = " ".join(["anticonstitutionnellement"] * 100)
    chunks = chunk_pages([(1, text)])
    assert len(chunks) > 1
    for c in chunks:
        assert set(c.text.split()) == {"anticonstitutionnellement"}


def test_size_and_overlap():
    text = ". ".join(f"Phrase numéro {i} du contrat" for i in range(200))
    chunks = chunk_pages([(1, text)])
    assert len(chunks) > 2
    assert all(len(c.text) <= 600 + 120 for c in chunks)
    assert chunks[1].text[:40] in chunks[0].text  # overlap from the previous chunk


def test_no_overlap_across_pages():
    chunks = chunk_pages([(1, "alpha " * 200), (2, "beta " * 200)])
    first_of_page_2 = next(c for c in chunks if c.page_no == 2)
    assert "alpha" not in first_of_page_2.text


def test_small_tail_merges_into_previous_chunk():
    chunks = chunk_pages(
        [(1, "x " * 450)], ChunkConfig(chunk_chars=800, overlap_chars=0, min_chars=200)
    )
    assert len(chunks) == 1
    assert chunks[0].text.count("x") == 450


def test_blank_page_text_gives_no_chunk():
    assert chunk_pages([(1, "   \n\n ")]) == []
    assert chunk_pages([(1, ""), (2, "Texte")]) == [Chunk(2, 0, "Texte")]


def test_version_names_the_parameters():
    assert ChunkConfig().version == "para-v1:600/120/200"


def test_embed_text_has_file_and_page_header():
    assert embed_text("facture.pdf", Chunk(2, 0, "Total")) == "facture.pdf p. 2\nTotal"
