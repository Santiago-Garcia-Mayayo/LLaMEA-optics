"""Encoding/decoding: the place where v5 went wrong."""
import numpy as np
import pytest

from harness.space import EncodedSpace
from problems.suite import Variable
from problems.toy import ToyProblem


def space():
    return EncodedSpace(ToyProblem().variables)


def test_bounds_shown_to_algorithm():
    s = space()
    assert s.lb.tolist() == [-1, -1, -0.5, -0.5]
    assert s.ub.tolist() == [1, 1, 4.5, 2.5]          # ordinal 1..5 -> 5 values; categorical 3 values
    assert s.n_values.tolist() == [0, 0, 5, 3]


def test_continuous_decode_endpoints_and_clipping():
    s = space()
    X = s.decode(np.array([[-1, 1, 0, 0], [0, 0, 0, 0], [-7, 9, 0, 0]], float))
    assert X[0, :2].tolist() == [-5, 5]
    assert X[1, :2].tolist() == [0, 0]
    assert X[2, :2].tolist() == [-5, 5]               # clipped


def test_integer_indices_decode_to_themselves():
    s = space()
    for k in range(5):
        assert s.decode(np.array([0, 0, k, 0], float))[0, 2] == 1 + k   # ordinal native values 1..5
    for k in range(3):
        assert s.decode(np.array([0, 0, 0, k], float))[0, 3] == k


@pytest.mark.parametrize("n", [2, 3, 6, 120])
def test_uniform_sampling_over_bounds_hits_every_value_equally(n):
    """The v5 bug produced only 2 of 6 glasses; [0, n-1] bounds would halve the end values."""
    s = EncodedSpace([Variable("categorical", 0, n - 1)])
    rng = np.random.default_rng(0)
    N = 200_000
    z = rng.uniform(s.lb, s.ub, size=(N, 1))
    k = s.decode(z)[:, 0].astype(int)
    counts = np.bincount(k, minlength=n)
    assert counts.size == n and counts.min() > 0
    expected = N / n
    # every value within 5 standard deviations of the uniform expectation (incl. both ends)
    assert np.all(np.abs(counts - expected) < 5 * np.sqrt(expected)), counts


def test_out_of_range_integers_are_clipped():
    s = space()
    X = s.decode(np.array([[0, 0, -3.2, 7.9], [0, 0, 99, -1]], float))
    assert X[0, 2] == 1 and X[0, 3] == 2
    assert X[1, 2] == 5 and X[1, 3] == 0


def test_encode_decode_roundtrip():
    s = space()
    X = np.array([[-5, 3.3, 4, 1], [1, 0, 3, 2]], float)
    assert np.allclose(s.decode(s.encode(X)), X)


@pytest.mark.parametrize("bad", [np.zeros(3), np.zeros((2, 5)), np.array([0, np.nan, 0, 0]),
                                 np.array([np.inf, 0, 0, 0])])
def test_invalid_input_raises(bad):
    with pytest.raises(ValueError):
        space().decode(bad)


def test_real_problems_spaces():
    from problems import get_problem
    for pid in ["cec/RC12", "cec/RC18", "cec/RC26"]:
        p = get_problem(pid)
        s = EncodedSpace(p.variables)
        z = np.random.default_rng(1).uniform(s.lb, s.ub, size=(500, p.dim))
        X = s.decode(z)
        assert np.all(X >= p.lb - 1e-12) and np.all(X <= p.ub + 1e-12)
        ii = p.integer_indices
        assert np.all(X[:, ii] == np.rint(X[:, ii]))
        assert np.allclose(p.seal(X[0]), X[0])          # decoded points are already sealed
