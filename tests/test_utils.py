import pytest

from ctm_ai_eval.util import infer_model_size


def test_infer_model_size():
    assert infer_model_size("model:e1m") == 1
    assert infer_model_size("model:e2b") == 2000
    assert infer_model_size("model:e3m") == 3
    assert infer_model_size("model:e4b") == 4000

    # Test cases for invalid inputs
    with pytest.raises(ValueError):
        infer_model_size("model:e5x")
    with pytest.raises(ValueError):
        infer_model_size("model:e6y")
