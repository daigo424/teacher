import hashlib

import pytest

from packages.core.services import text as text_service


def test_read_text_file_reads_utf8_content(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("hello", encoding="utf-8")

    assert text_service.read_text_file(str(path)) == "hello"


def test_estimate_tokens_matches_encoding_length():
    sample = "hello world"

    assert text_service.estimate_tokens(sample) == len(text_service.ENCODING.encode(sample))


def test_sha256_text_returns_expected_digest():
    assert text_service.sha256_text("abc") == hashlib.sha256(b"abc").hexdigest()


def test_chunk_text_splits_using_overlap():
    assert text_service.chunk_text("abcdefghij", chunk_size=4, overlap=1) == [
        "abcd",
        "defg",
        "ghij",
    ]


@pytest.mark.parametrize(
    ("chunk_size", "overlap", "message"),
    [
        (0, 0, "chunk_size must be > 0"),
        (4, -1, "overlap must be >= 0"),
        (4, 4, "overlap must be smaller than chunk_size"),
    ],
)
def test_chunk_text_validates_arguments(chunk_size, overlap, message):
    with pytest.raises(ValueError, match=message):
        text_service.chunk_text("abcdef", chunk_size=chunk_size, overlap=overlap)
