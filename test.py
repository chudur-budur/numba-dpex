# import sklearn_numba_dpex
import dpctl.tensor as dpt
import numpy as np

import numba_dpex as dpex

square_block_side = 2
work_group_size = (square_block_side, square_block_side)
dtype = np.float32

result_idx = np.array(
    [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 2],
        [1, 2],
        [0, 3],
        [1, 3],
        [0, 4],
        [1, 4],
        [0, 5],
        [1, 5],
        [4, 2],
        [5, 2],
        [4, 3],
        [5, 3],
        [4, 0],
        [5, 0],
        [4, 1],
        [5, 1],
        [4, 4],
        [5, 4],
        [4, 5],
        [5, 5],
        [2, 0],
        [3, 0],
        [2, 1],
        [3, 1],
        [2, 2],
        [3, 2],
        [2, 3],
        [3, 3],
        [2, 4],
        [3, 4],
        [2, 5],
        [3, 5],
    ]
)


local_idx = np.array(
    [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
    ]
)


@dpex.kernel
def matmul(
    X,  # IN READ-ONLY    (X_n_rows, n_cols)
    Y,  # IN READ-ONLY    (n_cols, y_n_rows),
    result,  # OUT             (X_n_rows, y_n_rows)
):

    X_n_rows = X.shape[0]
    Y_n_cols = Y.shape[1]
    n_cols = X.shape[1]

    result_row_idx = dpex.get_global_id(0)
    result_col_idx = dpex.get_global_id(1)
    # print(result_row_idx, ",", result_col_idx)

    local_row_idx = dpex.get_local_id(0)
    local_col_idx = dpex.get_local_id(1)
    # print(local_row_idx, ",", local_col_idx)

    n_blocks_for_cols = n_cols // square_block_side
    # print("n_blocks_for_cols =", n_blocks_for_cols, "\n")
    if (n_cols % square_block_side) > 0:
        n_blocks_for_cols += 1
    # print("n_blocks_for_cols =", n_blocks_for_cols)

    X_sliding_window = dpex.local.array(shape=work_group_size, dtype=dtype)
    Y_sliding_window = dpex.local.array(shape=work_group_size, dtype=dtype)

    output = dtype(0)

    for block_idx in range(n_blocks_for_cols):
        X_sliding_window[local_row_idx, local_col_idx] = dtype(0)
        Y_sliding_window[local_row_idx, local_col_idx] = dtype(0)
        if (result_row_idx < X_n_rows) and (
            (local_col_idx + (square_block_side * block_idx)) < n_cols
        ):
            X_sliding_window[local_row_idx, local_col_idx] = X[
                result_row_idx, local_col_idx + (square_block_side * block_idx)
            ]

        if (result_col_idx < Y_n_cols) and (
            (local_row_idx + (square_block_side * block_idx)) < n_cols
        ):
            Y_sliding_window[local_row_idx, local_col_idx] = Y[
                local_row_idx + (square_block_side * block_idx), result_col_idx
            ]

        # dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)
        dpex.barrier(dpex.LOCAL_MEM_FENCE)  # new

        for idx in range(square_block_side):
            output += (
                X_sliding_window[local_row_idx, idx]
                * Y_sliding_window[idx, local_col_idx]
            )

        # dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)
        dpex.barrier(dpex.LOCAL_MEM_FENCE)  # new

    if (result_row_idx < X_n_rows) and (result_col_idx < Y_n_cols):
        result[result_row_idx, result_col_idx] = output


def matmul_no_kernel(X, Y):
    X_n_rows = X.shape[0]
    Y_n_cols = Y.shape[1]
    n_cols = X.shape[1]

    n_blocks_for_cols = n_cols // square_block_side
    if (n_cols % square_block_side) > 0:
        n_blocks_for_cols += 1

    X_sliding_window = np.zeros(shape=work_group_size, dtype=dtype)
    Y_sliding_window = np.zeros(shape=work_group_size, dtype=dtype)

    result = np.zeros((X_n_rows, Y_n_cols), dtype=dtype)

    for i in range(result_idx.shape[0]):
        result_row_idx, result_col_idx = result_idx[i]
        local_row_idx, local_col_idx = local_idx[i]

        output = 0.0  # dtype(0)
        for block_idx in range(n_blocks_for_cols):
            if (result_row_idx < X_n_rows) and (
                (local_col_idx + (square_block_side * block_idx)) < n_cols
            ):
                X_sliding_window[local_row_idx, local_col_idx] = X[
                    result_row_idx,
                    local_col_idx + (square_block_side * block_idx),
                ]

            if (result_col_idx < Y_n_cols) and (
                (local_row_idx + (square_block_side * block_idx)) < n_cols
            ):
                Y_sliding_window[local_row_idx, local_col_idx] = Y[
                    local_row_idx + (square_block_side * block_idx),
                    result_col_idx,
                ]

            for idx in range(square_block_side):
                output += (
                    X_sliding_window[local_row_idx, idx]
                    * Y_sliding_window[idx, local_col_idx]
                )

        if (result_row_idx < X_n_rows) and (result_col_idx < Y_n_cols):
            result[result_row_idx, result_col_idx] = output
    return result


def _arange_reshaped(shape, dtype):
    n_items = shape[0] * shape[1]
    return np.arange(n_items, dtype=dtype).reshape(shape)


if __name__ == "__main__":
    X = _arange_reshaped((5, 5), dtype)
    Y = _arange_reshaped((5, 5), dtype)

    print(np.matmul(X, Y))

    X_ = dpt.asarray(X)
    Y_ = dpt.asarray(Y)

    device = X_.device.sycl_device
    result = dpt.zeros((5, 5), dtype, device=device)

    matmul[(6, 6), (2, 2)](X_, Y_, result)
    print(dpt.asnumpy(result))

    print(result_idx.shape)
    print(local_idx.shape)
    result = matmul_no_kernel(X, Y)
    print(result)
