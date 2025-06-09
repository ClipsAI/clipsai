import numpy as np
from clipsai.clip.texttiler import smooth

def manual_smooth(x, w):
    window_len = len(w)
    s = np.r_[2 * x[0] - x[window_len:1:-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1 : -window_len + 1]

def test_smooth_string_window():
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    result = smooth(x, window_len=3, window='flat')
    expected = manual_smooth(x, np.ones(3))
    assert np.allclose(result, expected)

def test_smooth_array_window():
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    w = np.array([0.2, 0.6, 0.2])
    result = smooth(x, window_len=3, window=w)
    expected = manual_smooth(x, w)
    assert np.allclose(result, expected)

