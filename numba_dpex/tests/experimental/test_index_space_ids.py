# SPDX-FileCopyrightText: 2022 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Tests different input array type support for the kernel."""

import dpnp
import numpy as np
import pytest
from numba.core.errors import TypingError

import numba_dpex as dpex
import numba_dpex.experimental as dpex_exp
from numba_dpex.kernel_api import Item, NdItem, NdRange
from numba_dpex.kernel_api import call_kernel as kapi_call_kernel

_SIZE = 16
_GROUP_SIZE = 4


@dpex_exp.kernel
def set_ones_no_item(a):
    a[0] = 1


@dpex_exp.kernel
def set_ones_item(item: Item, a):
    i = item.get_id(0)
    a[i] = 1


@dpex_exp.kernel
def set_last_one_item(item: Item, a):
    i = item.get_range(0) - 1
    a[i] = 1


@dpex_exp.kernel
def set_last_one_linear_item(item: Item, a):
    i = item.get_linear_range() - 1
    a[i] = 1


@dpex_exp.kernel
def set_last_one_linear_nd_item(nd_item: NdItem, a):
    i = nd_item.get_global_linear_range() - 1
    a[0] = i
    a[i] = 1


@dpex_exp.kernel
def set_last_one_nd_item(item: NdItem, a):
    if item.get_global_id(0) == 0:
        i = item.get_global_range(0) - 1
        a[0] = i
        a[i] = 1


@dpex_exp.kernel
def set_last_group_one_linear_nd_item(nd_item: NdItem, a):
    i = nd_item.get_local_linear_range() - 1
    a[0] = i
    a[i] = 1


@dpex_exp.kernel
def set_last_group_one_group_linear_nd_item(nd_item: NdItem, a):
    i = nd_item.get_group().get_local_linear_range() - 1
    a[0] = i
    a[i] = 1


@dpex_exp.kernel
def set_last_group_one_nd_item(item: NdItem, a):
    if item.get_global_id(0) == 0:
        i = item.get_local_range(0) - 1
        a[0] = i
        a[i] = 1


@dpex_exp.kernel
def set_ones_nd_item(nd_item: NdItem, a):
    i = nd_item.get_global_id(0)
    a[i] = 1


@dpex_exp.kernel
def set_local_ones_nd_item(nd_item: NdItem, a):
    i = nd_item.get_local_id(0)
    a[i] = 1


@dpex_exp.kernel
def set_dimensions_item(item: Item, a):
    i = item.get_id(0)
    a[i] = item.dimensions


@dpex_exp.kernel
def set_dimensions_nd_item(nd_item: NdItem, a):
    i = nd_item.get_global_id(0)
    a[i] = nd_item.dimensions


@dpex_exp.kernel
def set_dimensions_group(nd_item: NdItem, a):
    i = nd_item.get_global_id(0)
    a[i] = nd_item.get_group().dimensions


def _get_group_id_driver(nditem: NdItem, a):
    i = nditem.get_global_id(0)
    g = nditem.get_group()
    a[i] = g.get_group_id(0)


def _get_group_linear_id_driver(nditem: NdItem, a):
    i = nditem.get_global_linear_id()
    g = nditem.get_group()
    a[i] = g.get_group_linear_id()


def _get_group_range_driver(nditem: NdItem, a):
    i = nditem.get_global_id(0)
    g = nditem.get_group()
    a[i] = g.get_group_range(0)


def _get_group_linear_range_driver(nditem: NdItem, a):
    i = nditem.get_global_linear_id()
    g = nditem.get_group()
    a[i] = g.get_group_linear_range()


def _get_group_local_range_driver(nditem: NdItem, a):
    i = nditem.get_global_id(0)
    g = nditem.get_group()
    a[i] = g.get_local_range(0)


def test_item_get_id():
    a = dpnp.zeros(_SIZE, dtype=dpnp.float32)
    dpex_exp.call_kernel(set_ones_item, dpex.Range(a.size), a)

    assert np.array_equal(a.asnumpy(), np.ones(a.size, dtype=np.float32))


def test_item_get_range():
    a = dpnp.zeros(_SIZE, dtype=dpnp.float32)
    dpex_exp.call_kernel(set_last_one_item, dpex.Range(a.size), a)

    want = np.zeros(a.size, dtype=np.float32)
    want[-1] = 1

    assert np.array_equal(a.asnumpy(), want)


@pytest.mark.parametrize(
    "rng",
    [dpex.Range(_SIZE), dpex.Range(1, _GROUP_SIZE, int(_SIZE / _GROUP_SIZE))],
)
def test_item_get_linear_range(rng):
    a = dpnp.zeros(_SIZE, dtype=dpnp.float32)
    dpex_exp.call_kernel(set_last_one_linear_item, rng, a)

    want = np.zeros(a.size, dtype=np.float32)
    want[-1] = 1

    assert np.array_equal(a.asnumpy(), want)


@pytest.mark.parametrize(
    "kernel,rng",
    [
        (set_last_one_nd_item, dpex.NdRange((_SIZE,), (_GROUP_SIZE,))),
        (set_last_one_linear_nd_item, dpex.NdRange((_SIZE,), (_GROUP_SIZE,))),
        (
            set_last_one_linear_nd_item,
            dpex.NdRange((1, 1, _SIZE), (1, 1, _GROUP_SIZE)),
        ),
    ],
)
def test_nd_item_get_global_range(kernel, rng):
    a = dpnp.zeros(_SIZE, dtype=dpnp.float32)
    dpex_exp.call_kernel(kernel, rng, a)

    want = np.zeros(a.size, dtype=np.float32)
    want[-1] = 1
    want[0] = a.size - 1

    assert np.array_equal(a.asnumpy(), want)


@pytest.mark.parametrize(
    "kernel,rng",
    [
        (set_last_group_one_nd_item, dpex.NdRange((_SIZE,), (_GROUP_SIZE,))),
        (
            set_last_group_one_linear_nd_item,
            dpex.NdRange((_SIZE,), (_GROUP_SIZE,)),
        ),
        (
            set_last_group_one_linear_nd_item,
            dpex.NdRange((1, 1, _SIZE), (1, 1, _GROUP_SIZE)),
        ),
        (
            set_last_group_one_group_linear_nd_item,
            dpex.NdRange((_SIZE,), (_GROUP_SIZE,)),
        ),
        (
            set_last_group_one_group_linear_nd_item,
            dpex.NdRange((1, 1, _SIZE), (1, 1, _GROUP_SIZE)),
        ),
    ],
)
def test_nd_item_get_local_range(kernel, rng):
    a = dpnp.zeros(_SIZE, dtype=dpnp.float32)
    dpex_exp.call_kernel(kernel, rng, a)

    want = np.zeros(a.size, dtype=np.float32)
    want[_GROUP_SIZE - 1] = 1
    want[0] = _GROUP_SIZE - 1

    assert np.array_equal(a.asnumpy(), want)


def test_nd_item_get_global_id():
    a = dpnp.zeros(_SIZE, dtype=dpnp.float32)
    dpex_exp.call_kernel(
        set_ones_nd_item, dpex.NdRange((a.size,), (_GROUP_SIZE,)), a
    )

    assert np.array_equal(a.asnumpy(), np.ones(a.size, dtype=np.float32))


def test_nd_item_get_local_id():
    a = dpnp.zeros(_SIZE, dtype=dpnp.float32)

    dpex_exp.call_kernel(
        set_local_ones_nd_item, dpex.NdRange((a.size,), (_GROUP_SIZE,)), a
    )

    assert np.array_equal(
        a.asnumpy(),
        np.array(
            [1] * _GROUP_SIZE + [0] * (a.size - _GROUP_SIZE),
            dtype=np.float32,
        ),
    )


@pytest.mark.parametrize("dims", [1, 2, 3])
def test_item_dimensions(dims):
    a = dpnp.zeros(_SIZE, dtype=dpnp.float32)
    rng = [1] * dims
    rng[0] = a.size
    dpex_exp.call_kernel(set_dimensions_item, dpex.Range(*rng), a)

    assert np.array_equal(a.asnumpy(), dims * np.ones(a.size, dtype=np.float32))


@pytest.mark.parametrize("dims", [1, 2, 3])
@pytest.mark.parametrize(
    "kernel", [set_dimensions_nd_item, set_dimensions_group]
)
def test_nd_item_dimensions(dims, kernel):
    a = dpnp.zeros(_SIZE, dtype=dpnp.float32)
    rng, grp = [1] * dims, [1] * dims
    rng[0], grp[0] = a.size, _GROUP_SIZE
    dpex_exp.call_kernel(kernel, dpex.NdRange(rng, grp), a)

    assert np.array_equal(a.asnumpy(), dims * np.ones(a.size, dtype=np.float32))


def test_error_item_get_global_id():
    a = dpnp.zeros(_SIZE, dtype=dpnp.float32)

    with pytest.raises(TypingError):
        dpex_exp.call_kernel(set_ones_nd_item, dpex.Range(a.size), a)


def test_no_item():
    a = dpnp.zeros(_SIZE, dtype=dpnp.float32)
    dpex_exp.call_kernel(set_ones_no_item, dpex.Range(a.size), a)

    assert np.array_equal(
        a.asnumpy(), np.array([1] + [0] * (a.size - 1), dtype=np.float32)
    )


@pytest.mark.parametrize(
    "driver,rng",
    [
        (_get_group_id_driver, dpex.NdRange((_SIZE,), (_GROUP_SIZE,))),
        (_get_group_linear_id_driver, dpex.NdRange((_SIZE,), (_GROUP_SIZE,))),
        (
            _get_group_linear_id_driver,
            dpex.NdRange((1, 1, _SIZE), (1, 1, _GROUP_SIZE)),
        ),
    ],
)
def test_get_group_id(driver, rng):
    num_groups = _SIZE // _GROUP_SIZE

    a = dpnp.empty(_SIZE, dtype=dpnp.int32)
    ka = dpnp.empty(_SIZE, dtype=dpnp.int32)
    expected = np.empty(_SIZE, dtype=np.int32)
    dpex_exp.call_kernel(dpex_exp.kernel(driver), rng, a)
    kapi_call_kernel(driver, rng, ka)

    for gid in range(num_groups):
        for lid in range(_GROUP_SIZE):
            expected[gid * _GROUP_SIZE + lid] = gid

    assert np.array_equal(a.asnumpy(), expected)
    assert np.array_equal(ka.asnumpy(), expected)


@pytest.mark.parametrize(
    "driver,rng",
    [
        (_get_group_range_driver, dpex.NdRange((_SIZE,), (_GROUP_SIZE,))),
        (
            _get_group_linear_range_driver,
            dpex.NdRange((_SIZE,), (_GROUP_SIZE,)),
        ),
        (
            _get_group_linear_range_driver,
            dpex.NdRange((1, 1, _SIZE), (1, 1, _GROUP_SIZE)),
        ),
    ],
)
def test_get_group_range(driver, rng):
    num_groups = _SIZE // _GROUP_SIZE

    a = dpnp.empty(_SIZE, dtype=dpnp.int32)
    ka = dpnp.empty(_SIZE, dtype=dpnp.int32)
    expected = np.empty(_SIZE, dtype=np.int32)
    dpex_exp.call_kernel(dpex_exp.kernel(driver), rng, a)
    kapi_call_kernel(driver, rng, ka)

    for gid in range(num_groups):
        for lid in range(_GROUP_SIZE):
            expected[gid * _GROUP_SIZE + lid] = num_groups

    assert np.array_equal(a.asnumpy(), expected)
    assert np.array_equal(ka.asnumpy(), expected)


def test_get_group_local_range():
    global_size = 100
    group_size = 20
    num_groups = global_size // group_size

    a = dpnp.empty(global_size, dtype=dpnp.int32)
    ka = dpnp.empty(global_size, dtype=dpnp.int32)
    expected = np.empty(global_size, dtype=np.int32)
    ndrange = NdRange((global_size,), (group_size,))
    dpex_exp.call_kernel(
        dpex_exp.kernel(_get_group_local_range_driver), ndrange, a
    )
    kapi_call_kernel(_get_group_local_range_driver, ndrange, ka)

    for gid in range(num_groups):
        for lid in range(group_size):
            expected[gid * group_size + lid] = group_size

    assert np.array_equal(a.asnumpy(), expected)
    assert np.array_equal(ka.asnumpy(), expected)


I_SIZE, J_SIZE, K_SIZE = 2, 3, 4


@dpex_exp.kernel
def set_3d_ones_item(item: Item, a):
    i = item.get_id(0)
    j = item.get_id(1)
    k = item.get_id(2)

    # Since we have different sizes for each dimension, wrong order will result
    # that some indexes will be set twice and some won't be set.
    index = i + I_SIZE * (j + J_SIZE * k)

    a[index] = 1


@dpex_exp.kernel
def set_3d_ones_item_linear(item: Item, a):
    # Since we have different sizes for each dimension, wrong order will result
    # that some indexes will be set twice and some won't be set.
    index = item.get_linear_id()

    a[index] = 1


@dpex_exp.kernel
def set_3d_ones_nd_item_linear(nd_item: NdItem, a):
    # Since we have different sizes for each dimension, wrong order will result
    # that some indexes will be set twice and some won't be set.
    index = nd_item.get_global_linear_id()

    a[index] = 1


@dpex_exp.kernel
def set_local_3d_ones_nd_item_linear(nd_item: NdItem, a):
    # Since we have different sizes for each dimension, wrong order will result
    # that some indexes will be set twice and some won't be set.
    index = nd_item.get_local_linear_id()

    a[index] = 1


@pytest.mark.parametrize("kernel", [set_3d_ones_item, set_3d_ones_item_linear])
def test_item_index_order(kernel):
    a = dpnp.zeros(I_SIZE * J_SIZE * K_SIZE, dtype=dpnp.int32)

    dpex_exp.call_kernel(kernel, dpex.Range(I_SIZE, J_SIZE, K_SIZE), a)

    assert np.array_equal(a.asnumpy(), np.ones(a.size, dtype=np.int32))


def test_nd_item_index_order():
    a = dpnp.zeros(I_SIZE * J_SIZE * K_SIZE, dtype=dpnp.int32)

    dpex_exp.call_kernel(
        set_3d_ones_nd_item_linear,
        dpex.NdRange((I_SIZE, J_SIZE, K_SIZE), (1, 1, K_SIZE)),
        a,
    )

    assert np.array_equal(a.asnumpy(), np.ones(a.size, dtype=np.int32))


def test_nd_item_local_linear_id():
    a = dpnp.zeros(I_SIZE * J_SIZE * K_SIZE, dtype=dpnp.int32)

    dpex_exp.call_kernel(
        set_local_3d_ones_nd_item_linear,
        dpex.NdRange((I_SIZE, J_SIZE, K_SIZE), (1, 1, K_SIZE)),
        a,
    )

    assert np.array_equal(
        a.asnumpy(),
        np.array([1] * K_SIZE + [0] * (a.size - K_SIZE), dtype=np.int32),
    )
