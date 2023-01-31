#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

template <typename T>
sycl::event gemm(sycl::queue q,
                 const T *X_ptr,
                 const T *Y_ptr,
                 T *R_ptr,
                 size_t X_n_rows,
                 size_t X_n_cols,
                 size_t Y_n_cols,
                 size_t gws0_blocks,
                 size_t gws1_blocks,
                 size_t lws0,
                 size_t lws1,
                 size_t square_block_size,
                 const std::vector<sycl::event> &depends = {})
{
    sycl::event comp_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        // sycl::local_accessor<T, 2> X_sliding_window(
        //      sycl::range<2>(lws0, lws1), cgh);
        // sycl::local_accessor<T, 2> Y_sliding_window(
        //      sycl::range<2>(lws0, lws1), cgh);

        sycl::accessor<T, 2, sycl::access::mode::read_write,
                       sycl::access::target::local>
            X_sliding_window(sycl::range<2>(lws0, lws1), cgh);
        sycl::accessor<T, 2, sycl::access::mode::read_write,
                       sycl::access::target::local>
            Y_sliding_window(sycl::range<2>(lws0, lws1), cgh);

        auto gwsRange = sycl::range<2>(gws0_blocks * lws0, gws1_blocks * lws1);
        auto lwsRange = sycl::range<2>(lws0, lws1);
        size_t n_cols = X_n_cols;
        cgh.parallel_for(
            sycl::nd_range<2>(gwsRange, lwsRange), [=](sycl::nd_item<2> ndit) {
                size_t result_row_idx = ndit.get_global_id(0);
                size_t result_col_idx = ndit.get_global_id(1);
                size_t local_row_idx = ndit.get_local_id(0);
                size_t local_col_idx = ndit.get_local_id(1);

                size_t n_blocks_for_cols =
                    (n_cols + square_block_size - 1) / square_block_size;

                T output(0);
                for (size_t block_idx = 0; block_idx < n_blocks_for_cols;
                     ++block_idx) {
                    size_t col_idx =
                        local_col_idx + square_block_size * block_idx;
                    T X_v = (result_row_idx < X_n_rows && col_idx < n_cols)
                                ? X_ptr[result_row_idx * X_n_cols + col_idx]
                                : T(0);
                    X_sliding_window[ndit.get_local_id()] = X_v;

                    size_t row_idx =
                        local_row_idx + square_block_size * block_idx;
                    T Y_v = (result_col_idx < Y_n_cols && row_idx < n_cols)
                                ? Y_ptr[row_idx * Y_n_cols + result_col_idx]
                                : T(0);

                    Y_sliding_window[ndit.get_local_id()] = Y_v;

                    ndit.barrier(sycl::access::fence_space::local_space);

                    for (size_t idx = 0; idx < square_block_size; ++idx) {
                        output +=
                            X_sliding_window[sycl::id<2>(local_row_idx, idx)] *
                            Y_sliding_window[sycl::id<2>(idx, local_col_idx)];
                    }

                    ndit.barrier(sycl::access::fence_space::local_space);
                }
                if (result_row_idx < X_n_rows && result_col_idx < Y_n_cols) {
                    R_ptr[result_row_idx * Y_n_cols + result_col_idx] = output;
                }
            });
    });

    return comp_ev;
}

int main(void)
{
    using T = float;
    size_t n = 5;

    size_t square_block_side = 2;

    sycl::queue q; // q{ sycl::default_selector_v };

    size_t n_elems = n * n;
    T *X_ptr = sycl::malloc_device<T>(n_elems, q);
    T *Y_ptr = sycl::malloc_device<T>(n_elems, q);
    T *R_ptr = sycl::malloc_device<T>(n_elems, q);

    sycl::event pop_ev = q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for({n_elems}, [=](sycl::id<1> id) {
            T v(id[0]);
            X_ptr[id] = v;
            Y_ptr[id] = v;
            R_ptr[id] = T(-7);
        });
    });

    size_t lws0 = 2;
    size_t lws1 = 2;
    size_t gws0_blocks = (n + lws0 - 1) / lws0;
    size_t gws1_blocks = (n + lws1 - 1) / lws1;

    sycl::event comp_ev = gemm(q, X_ptr, Y_ptr, R_ptr, n, n, n, gws0_blocks,
                               gws1_blocks, lws0, lws1, lws0, {pop_ev});

    T *host_res = new T[n_elems];

    sycl::event copy_ev = q.copy<T>(R_ptr, host_res, n_elems, {comp_ev});

    copy_ev.wait();

    for (size_t i0 = 0; i0 < n; ++i0) {
        std::cout << "[ ";
        for (size_t i1 = 0; i1 < n; ++i1) {
            std::cout << host_res[i0 * n + i1] << " ";
        }
        std::cout << "]" << std::endl;
    }

    delete[] host_res;

    sycl::free(X_ptr, q);
    sycl::free(Y_ptr, q);
    sycl::free(R_ptr, q);

    return 0;
}
