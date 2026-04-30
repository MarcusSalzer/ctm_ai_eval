from ctm_ai_eval.utils.hashing import stable_hash


def test_invariant_to_order():
    assert stable_hash({"A": 1, "B": 3}) == stable_hash({"B": 3, "A": 1})
