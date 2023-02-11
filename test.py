import dpctl
import dpctl.tensor as dpt
import numpy as np

import numba_dpex as ndpx

square_block_side = 2
work_group_size = (square_block_side, square_block_side)
dtype = np.float32


@ndpx.kernel
def matmul(
    X,  # IN READ-ONLY    (X_n_rows, n_cols)
    Y,  # IN READ-ONLY    (n_cols, y_n_rows),
    result,  # OUT             (X_n_rows, y_n_rows)
):
    X_n_rows = X.shape[0]
    Y_n_cols = Y.shape[1]
    n_cols = X.shape[1]

    result_row_idx = ndpx.get_global_id(0)
    result_col_idx = ndpx.get_global_id(1)

    local_row_idx = ndpx.get_local_id(0)
    local_col_idx = ndpx.get_local_id(1)

    n_blocks_for_cols = n_cols // square_block_side
    if (n_cols % square_block_side) > 0:
        n_blocks_for_cols += 1

    X_sliding_window = ndpx.local.array(shape=work_group_size, dtype=dtype)
    Y_sliding_window = ndpx.local.array(shape=work_group_size, dtype=dtype)

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

        ndpx.barrier(ndpx.LOCAL_MEM_FENCE)

        for idx in range(square_block_side):
            output += (
                X_sliding_window[local_row_idx, idx]
                * Y_sliding_window[idx, local_col_idx]
            )

        ndpx.barrier(ndpx.LOCAL_MEM_FENCE)

    if (result_row_idx < X_n_rows) and (result_col_idx < Y_n_cols):
        result[result_row_idx, result_col_idx] = output


def _arange_reshaped(shape, dtype):
    n_items = shape[0] * shape[1]
    return np.arange(n_items, dtype=dtype).reshape(shape)


if __name__ == "__main__":
    X = _arange_reshaped((5, 5), dtype)
    Y = _arange_reshaped((5, 5), dtype)

    answer = np.matmul(X, Y)
    print(answer)

    device = dpctl.SyclDevice("opencl:cpu:0")
    queue = dpctl.SyclQueue(device)

    X_ = dpt.usm_ndarray(
        X.shape,
        dtype=X.dtype,
        buffer="device",
        buffer_ctor_kwargs={"queue": queue},
    )
    X_.usm_data.copy_from_host(X.reshape((-1)).view("|u1"))

    Y_ = dpt.usm_ndarray(
        Y.shape,
        dtype=Y.dtype,
        buffer="device",
        buffer_ctor_kwargs={"queue": queue},
    )
    Y_.usm_data.copy_from_host(Y.reshape((-1)).view("|u1"))

    Z = np.zeros((5, 5), dtype=dtype)
    Z_ = dpt.usm_ndarray(
        Z.shape,
        dtype=Z.dtype,
        buffer="device",
        buffer_ctor_kwargs={"queue": queue},
    )
    Z_.usm_data.copy_from_host(Z.reshape((-1)).view("|u1"))

    matmul[ndpx.NdRange(ndpx.Range(6, 6), ndpx.Range(2, 2))](X_, Y_, Z_)

    Z_.usm_data.copy_to_host(Z.reshape((-1)).view("|u1"))

    print(Z)
