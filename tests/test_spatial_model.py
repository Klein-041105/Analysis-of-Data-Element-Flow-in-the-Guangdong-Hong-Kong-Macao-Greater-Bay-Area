import pandas as pd
import numpy as np
import pytest

from src.models.spatial_model import SpatialModel


def test_moran_i_basic():
    # simple chain weight matrix
    W = pd.DataFrame([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    values = pd.Series([1.0, 2.0, 3.0])

    sm = SpatialModel(weight_matrix=W)
    I = sm.moran_i(values)

    assert isinstance(I, float)
    # Moran's I typically in [-1, 1]
    assert -1.0 <= I <= 1.0


def test_moran_i_no_weight_matrix_raises():
    sm = SpatialModel(weight_matrix=None)
    with pytest.raises(ValueError):
        sm.moran_i(pd.Series([1, 2, 3]))
