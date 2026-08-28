from accesstrap.connectives import is_connective_token
from accesstrap.entropy import (
    choose_entropy_rule,
    entropy_from_logits,
    lexical_entropies,
    topk_entropies,
)


def test_connective_match():
    assert is_connective_token("Therefore")
    assert is_connective_token(" therefore,")
    assert not is_connective_token("apple")


def test_entropy_uniform_two_class():
    # logits [0, 0] → 2-way uniform → ln 2
    ent = entropy_from_logits([0.0, 0.0])
    assert abs(ent - 0.693147) < 1e-4


def test_lexical_filters_internal():
    toks = ["therefore", "the", "24"]
    ents = [1.0, 0.2, 0.1]
    # "24" appears in access block; "therefore" does not
    lex = lexical_entropies(toks, ents, access_block="48/2 = 24")
    assert lex == [1.0]


def test_top20():
    ents = [0.1, 0.2, 0.9, 0.8, 0.05]
    top = topk_entropies(ents, 0.2)
    assert top == [0.9]  # ceil(1)


def test_fallback_rule():
    assert choose_entropy_rule({"A": 100, "B": 100, "C": 100}) == "lexical"
    assert choose_entropy_rule({"A": 100, "B": 5, "C": 100}) == "top20"
