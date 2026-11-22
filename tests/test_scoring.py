import pytest

from localmind.scoring import (
    ClassifyConfig,
    Example,
    cosine,
    decide,
    folder_scores,
    from_bytes,
    normalize,
    to_bytes,
)


def test_bytes_round_trip_and_cosine():
    v = normalize([3.0, 4.0])
    assert from_bytes(to_bytes(v)) == pytest.approx([0.6, 0.8])
    assert cosine(v, v) == pytest.approx(1.0)


def test_normalize_zero_vector_stays_zero():
    assert normalize([0.0, 0.0]) == [0.0, 0.0]


def test_folder_score_is_mean_of_top_k_and_user_examples_count_twice():
    doc = normalize([1.0, 0.0])
    near, far = normalize([1.0, 0.1]), normalize([1.0, 1.0])
    examples = [Example(1, near, 1), Example(1, far, 1), Example(2, far, 2), Example(2, near, 1)]
    scores = folder_scores(doc, examples, top_k=2)
    assert scores[1][0] == pytest.approx((cosine(doc, near) + cosine(doc, far)) / 2)
    # far counts twice in folder 2: its top 2 is near + far
    assert scores[2][0] == pytest.approx((cosine(doc, near) + cosine(doc, far)) / 2)
    assert scores[1][1] == 2 and scores[2][1] == 2


def test_user_weight_can_lift_a_folder():
    doc = normalize([1.0, 0.0])
    near, far = normalize([1.0, 0.1]), normalize([1.0, 1.0])
    scores = folder_scores(doc, [Example(1, near, 2), Example(1, far, 1)], top_k=2)
    assert scores[1][0] == pytest.approx(cosine(doc, near))  # near twice beats far
    assert scores[1][1] == 2


def test_decide_clear_needs_score_margin_and_size():
    cfg = ClassifyConfig()
    assert decide({1: (0.93, 3), 2: (0.80, 5)}, cfg) == ("clear", [1, 2])
    assert decide({1: (0.93, 2), 2: (0.80, 5)}, cfg)[0] == "ask_llm"  # too few documents
    assert decide({1: (0.93, 3), 2: (0.90, 5)}, cfg)[0] == "ask_llm"  # margin < 0.05
    assert decide({1: (0.79, 3)}, cfg)[0] == "ask_llm"  # score < 0.80
    assert decide({1: (0.85, 3)}, cfg)[0] == "clear"  # single folder: second = 0
    assert decide({}, cfg) == ("ask_llm", [])


def test_config_defaults_match_the_spec():
    cfg = ClassifyConfig()
    assert (cfg.min_score, cfg.min_margin, cfg.min_folder_docs, cfg.top_k, cfg.user_weight) == (
        0.80, 0.05, 3, 3, 2)
    assert (cfg.name_merge, cfg.embed_chars, cfg.llm_chars, cfg.candidates, cfg.examples) == (
        0.90, 2000, 3000, 3, 5)
