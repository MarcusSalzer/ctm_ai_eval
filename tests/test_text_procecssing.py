from ctm_ai_eval.rag.datamodels import SpanToken
from ctm_ai_eval.rag.text_processing import longest_common_substring, tokenize_words


def test_tokenize_words_simple():

    tokens = tokenize_words("hello world!")
    assert tokens == [SpanToken("hello", 0, 5), SpanToken("world", 6, 11)]


class TestLcs:
    def test_zero(self):
        assert longest_common_substring("hello", "123") == 0

    def test_one(self):
        assert longest_common_substring("hello", "world") == 1

    def test_full(self):
        assert longest_common_substring("world hello", "hello world") == 5
