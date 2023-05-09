# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for dpnp ndarray constructors."""

import dpctl
import dpnp
import pytest
from numba import errors

from numba_dpex import dpjit

shapes = [11, (3, 7)]
dtypes = [dpnp.int32, dpnp.int64, dpnp.float32, dpnp.float64]
usm_types = ["device", "shared", "host"]


@pytest.mark.parametrize("shape", shapes)
def test_dpnp_zeros_default(shape):
    """Test dpnp.zeros() with default parameters inside dpjit."""

    @dpjit
    def func(shape):
        c = dpnp.zeros(shape)
        return c

    try:
        c = func(shape)
    except Exception:
        pytest.fail("Calling dpnp.zeros() inside dpjit failed.")

    if len(c.shape) == 1:
        assert c.shape[0] == shape
    else:
        assert c.shape == shape

    assert not c.asnumpy().any()

    dummy = dpnp.zeros(shape)

    assert c.dtype == dummy.dtype
    assert c.usm_type == dummy.usm_type
    assert c.sycl_device == dummy.sycl_device


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("usm_type", usm_types)
def test_dpnp_zeros_from_device(shape, dtype, usm_type):
    """ "Use device only in dpnp.zeros() inside dpjit."""
    device = dpctl.SyclDevice().filter_string

    @dpjit
    def func(shape):
        c = dpnp.zeros(shape, dtype=dtype, usm_type=usm_type, device=device)
        return c

    try:
        c = func(shape)
    except Exception:
        pytest.fail("Calling dpnp.zeros() inside dpjit failed.")

    if len(c.shape) == 1:
        assert c.shape[0] == shape
    else:
        assert c.shape == shape

    assert c.dtype == dtype
    assert c.usm_type == usm_type
    assert c.sycl_device.filter_string == device
    assert not c.asnumpy().any()


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("usm_type", usm_types)
def test_dpnp_zeros_from_queue(shape, dtype, usm_type):
    """ "Use queue only in dpnp.zeros() inside dpjit."""

    @dpjit
    def func(shape, queue):
        c = dpnp.zeros(shape, dtype=dtype, usm_type=usm_type, sycl_queue=queue)
        return c

    queue = dpctl.SyclQueue()

    try:
        c = func(shape, queue)
    except Exception:
        pytest.fail("Calling dpnp.zeros() inside dpjit failed.")

    if len(c.shape) == 1:
        assert c.shape[0] == shape
    else:
        assert c.shape == shape

    assert c.dtype == dtype
    assert c.usm_type == usm_type
    assert c.sycl_device == queue.sycl_device
    assert not c.asnumpy().any()

    if c.sycl_queue != queue:
        pytest.xfail(
            "Returned queue does not have the queue passed to the dpnp function."
        )


def test_dpnp_zeros_exceptions():
    """Test if exception is raised when both queue and device are specified."""
    device = dpctl.SyclDevice().filter_string

    @dpjit
    def func(shape, queue):
        c = dpnp.zeros(shape, sycl_queue=queue, device=device)
        return c

    queue = dpctl.SyclQueue()

    try:
        func(10, queue)
    except Exception as e:
        assert isinstance(e, errors.TypingError)
