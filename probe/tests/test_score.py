from accesstrap.score import (
    extract_predicted_number,
    gold_intermediates,
    gsm8k_correct,
    parse_gsm8k_calcs,
    perturb_value,
    qa_correct,
)


GOLD_SOL = (
    "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
    "Natalia sold 48+24 = <<48+24=72>>72 clips altogether.\n"
    "#### 72"
)


def test_parse_and_drop_last():
    calcs = parse_gsm8k_calcs(GOLD_SOL)
    assert calcs == [("48/2", "24"), ("48+24", "72")]
    assert gold_intermediates(GOLD_SOL) == [("48/2", "24")]


def test_single_calc_not_leaked_as_gold_access():
    sol = "2+2 = <<2+2=4>>4\n#### 4"
    assert gold_intermediates(sol) == []


def test_perturb_never_equals_gold():
    assert perturb_value("24") != "24"
    assert perturb_value("24.5") != "24.5"


def test_gsm8k_extract():
    assert extract_predicted_number("blah #### 72\n") == "72"
    assert extract_predicted_number("the answer is 72") == "72"
    assert gsm8k_correct("#### 72", "72")
    assert not gsm8k_correct("#### 71", "72")


def test_qa_correct_end_line():
    assert qa_correct("Reasoning here.\n#### Paris", "Paris")
    assert qa_correct("The capital is Paris.", "Paris")
    assert not qa_correct("#### London", "Paris")
