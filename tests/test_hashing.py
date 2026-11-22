from localmind.hashing import content_hash, file_sha256, normalize_text


def test_file_sha256_matches_known_value(tmp_path):
    f = tmp_path / "a.txt"
    f.write_bytes(b"abc")
    # Reference SHA-256 of b"abc" (FIPS 180-2 test vector)
    assert file_sha256(f) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_file_sha256_streams_large_files(tmp_path):
    f = tmp_path / "big.bin"
    f.write_bytes(b"x" * 5_000_000)
    assert len(file_sha256(f)) == 64


def test_normalize_collapses_whitespace_and_line_endings():
    assert normalize_text("  Hello\r\n\r\n  world\t !  ") == "Hello world !"


def test_normalize_unifies_unicode_forms():
    composed = "café"  # é as one code point
    decomposed = "café"  # e + combining accent
    assert composed != decomposed
    assert normalize_text(composed) == normalize_text(decomposed)


def test_content_hash_ignores_cosmetic_differences():
    assert content_hash("Total : 42 €\n") == content_hash("Total :  42 €\r\n\r\n")


def test_content_hash_keeps_case_differences():
    # Case can carry meaning (acronyms, names): we do not fold it.
    assert content_hash("Apple") != content_hash("apple")
