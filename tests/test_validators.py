import numpy as np
import pandas as pd
import pytest

from src.utils import validators


def test_ensure_not_empty_accepts_nonempty():
    validators.ensure_not_empty([1, 2, 3], name="arr")
    validators.ensure_not_empty(np.array([1]), name="arr2")
    validators.ensure_not_empty(pd.DataFrame({"a": [1]}), name="df")


def test_ensure_not_empty_raises_for_empty_and_none():
    with pytest.raises(ValueError):
        validators.ensure_not_empty([], name="empty")
    with pytest.raises(ValueError):
        validators.ensure_not_empty(None, name="none")


def test_ensure_numeric_array_success_and_failure():
    arr = validators.ensure_numeric_array([1, 2, 3], name="nums")
    assert isinstance(arr, np.ndarray)

    with pytest.raises(TypeError):
        validators.ensure_numeric_array(["a", "b"], name="bad")
