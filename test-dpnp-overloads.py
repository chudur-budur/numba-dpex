import faulthandler

import dpctl
import numpy as np
from numba import njit

from numba_dpex.tests._helper import dpnp_debug, is_gen12
from numba_dpex.tests.njit_tests.dpnp._helper import wrapper_function

faulthandler.enable()

list_of_dtypes = [
    np.int32,
    np.int64,
    np.float32,
    np.float64,
]

list_of_shape = [
    (10),
    (5, 2),
]

list_of_unary_ops = [
    "sum",
    "prod",
    "max",
    "min",
    "mean",
    "argmax",
    "argmin",
    "argsort",
    "copy",
    "cumsum",
    "cumprod",
]


def test_unary_ops(filter_str, op_name, dtype, shape):
    N = 10
    a = np.array(np.random.random(N), dtype)
    op = wrapper_function("a", f"a.{op_name}()", globals())

    actual = np.empty(shape=a.shape, dtype=a.dtype)
    expected = np.empty(shape=a.shape, dtype=a.dtype)

    f = njit(op)
    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device), dpnp_debug():
        actual = f(a)
        # captured = capfd.readouterr()
        # assert "dpnp implementation" in captured.out

    expected = op(a)

    print("actual =", actual)
    print("expected =", expected)
    # np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=0)


if __name__ == "__main__":
    test_unary_ops("level_zero:gpu:0", "copy", np.int32, (10))
