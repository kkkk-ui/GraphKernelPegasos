import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

# 高速枝刈り関数
def prune_basis(np.ndarray[np.float64_t, ndim=1] alpha,
                np.ndarray[np.float64_t, ndim=2] kernel_cache,
                np.ndarray[np.int32_t, ndim=1] support_indices,
                int i_t,
                int t,
                double delta_c):

    cdef int j, coh_index = -1
    cdef double coh, coh_max = 0.0

    for j in support_indices:
        if j == i_t:
            continue
        if t - kernel_cache[j][1] < 100:
            continue
        if alpha[j] > 1.0:
            continue
        coh = kernel_cache[j][0] 
        if coh > coh_max:
            coh_max = coh
            coh_index = j

    if coh_max > delta_c and coh_index >= 0:
        alpha[coh_index] = 0.0
