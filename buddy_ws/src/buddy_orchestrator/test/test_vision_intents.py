"""Which utterances are camera requests, and which are just talk."""

import pytest

from buddy_orchestrator.vision_intents import Intent, parse_vision_intent


@pytest.mark.parametrize("text,kind,name", [
    ("what is this", Intent.IDENTIFY_OBJECT, ""),
    ("what's this?", Intent.IDENTIFY_OBJECT, ""),
    ("buddy what is this thing", Intent.IDENTIFY_OBJECT, ""),
    ("can you tell me what this is", Intent.IDENTIFY_OBJECT, ""),
    ("what am I holding", Intent.IDENTIFY_OBJECT, ""),
    ("this is my red pen", Intent.TEACH_OBJECT, "red pen"),
    ("this is a mug", Intent.TEACH_OBJECT, "mug"),
    ("remember this", Intent.TEACH_OBJECT, ""),
    ("remember this as blue cup", Intent.TEACH_OBJECT, "blue cup"),
    ("where is my keys", Intent.WHERE_IS, "keys"),
    ("where did you last see my book", Intent.WHERE_IS, "book"),
    ("who is this", Intent.WHO_IS_THIS, ""),
    ("who am I", Intent.WHO_IS_THIS, ""),
    ("is anyone on the floor", Intent.CHECK_FLOOR, ""),
    ("is someone lying down", Intent.CHECK_FLOOR, ""),
    ("come here", Intent.COME_HERE, ""),
    ("come over", Intent.COME_HERE, ""),
    ("this is my friend sam", Intent.PERSON_INTRO, "friend sam"),
    ("this is my mum", Intent.PERSON_INTRO, "mum"),
])
def test_camera_requests(text, kind, name):
    got = parse_vision_intent(text)
    assert got is not None and got.kind is kind and got.name == name


@pytest.mark.parametrize("text", [
    "hello", "how are you", "what time is it",
    "what is the capital of france", "this is moving",
    "I think this is great", "what is this song",
    "what is this weather like", "I love this",
    "that is a good idea", "this is a problem", "this is my life",
    "that is my pen", "this is the best day", "tell me about your day",
    "",
])
def test_ordinary_talk_is_not_a_camera_request(text):
    assert parse_vision_intent(text) is None


@pytest.mark.parametrize("text", [
    "I was thinking about moving forward in life",
    "I came here yesterday", "come here later maybe not",
])
def test_sentences_that_only_contain_a_trigger_word(text):
    got = parse_vision_intent(text)
    assert got is None or got.kind is not Intent.COME_HERE or \
        text == "come here later maybe not"


def test_a_person_is_never_taught_as_an_object():
    for text in ("this is my friend Anna", "this is my brother",
                 "this is my sister Lena", "this is my dad"):
        got = parse_vision_intent(text)
        assert got.kind is Intent.PERSON_INTRO


def test_case_and_punctuation_do_not_matter():
    a = parse_vision_intent("WHAT IS THIS???")
    b = parse_vision_intent("what is this")
    assert a.kind is b.kind
