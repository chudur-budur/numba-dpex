import numpy as np, numba_dpex

@numba_dpex.kernel
def div_kernel(dst, src, m):
    i = numba_dpex.get_global_id(0)
    dst[i] = src[i] // m

import dpctl
with dpctl.device_context(dpctl.SyclQueue()):
    X = np.arange(10)
    Y = np.arange(10)
    div_kernel[10, numba_dpex.DEFAULT_LOCAL_SIZE](Y, X, 5)
    D = X//5
    print(Y, D)
