from weighttraces.cot import extract_cot_text, prefix_ids, prefix_token_count
from weighttraces.score import answer_span, is_correct


def test_think_cot_and_answer_span():
    text = "<think>1+1 is 2, wait no 3</think>\n#### 2"
    assert extract_cot_text(text, "think") == "1+1 is 2, wait no 3"
    assert "3" not in answer_span(text)
    assert is_correct(text, "2")
    assert not is_correct(text, "3")


def test_think_cot_prompt_already_open():
    text = "add 1 and 1\n</think>\n#### 2"
    assert extract_cot_text(text, "think") == "add 1 and 1\n"


def test_base_cot_before_hash():
    text = "Let's add 3 and 4. Therefore #### 7"
    assert extract_cot_text(text, "base") == "Let's add 3 and 4. Therefore "
    assert is_correct(text, "7")


def test_prefix_floor_half_and_empty():
    assert prefix_token_count(0) == 0
    assert prefix_token_count(1) == 0
    assert prefix_token_count(5) == 2
    assert prefix_ids([10, 11, 12, 13, 14]) == [10, 11]
    assert prefix_ids([]) == []
