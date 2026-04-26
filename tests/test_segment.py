"""Sentence segmentation tests."""
from halluguard.segment import split_sentences


def test_empty():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_single_sentence():
    assert split_sentences("Hello world.") == ["Hello world."]


def test_two_sentences():
    out = split_sentences("Hello world. This is fine.")
    assert out == ["Hello world.", "This is fine."]


def test_question_then_statement():
    assert split_sentences("Is this OK? Yes it is.") == ["Is this OK?", "Yes it is."]


def test_abbreviation_kept():
    # "e.g." should not split the sentence
    out = split_sentences("Use indexes, e.g. B-tree. Cosine works too.")
    assert len(out) == 2
    assert out[0].startswith("Use indexes")
    assert out[1] == "Cosine works too."


def test_turkish():
    # Turkish capital letters with diacritics
    out = split_sentences("Bu doğru. Ama dikkat et.")
    assert out == ["Bu doğru.", "Ama dikkat et."]
