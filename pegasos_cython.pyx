import numpy as np
cimport numpy as np
import GraphKernelFunc as gkf
from sklearn.model_selection import train_test_split
from libc.stdlib cimport malloc, free, rand, srand
from libc.time cimport time

cpdef np.ndarray[np.float64_t, ndim=1] train(np.ndarray[object] G_train,
          np.ndarray[np.int32_t, ndim=1] y_train,
          int iter,
          double lamda,
          double delta_c):

    cdef int n = G_train.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] alpha = np.zeros(n)
    cdef np.ndarray[np.float64_t, ndim=2] kernel_cache = np.zeros((n, 2))
    cdef np.ndarray[np.float64_t, ndim=1] kernel_vec
    cdef int i_t, j, coh_index
    cdef double sigma_loss, coh, coh_max
    cdef int t = 1
    srand(<unsigned int>time(NULL))

    while t <= iter:    
        i_t = rand() % n  
        support_indices = np.where(alpha > 0)[0].astype(np.int32)
        sigma_loss = 0.0

        if support_indices.shape[0] > 0:
            kernel_vec = gkf.GraphkernelFunc.k_vec_wl(G_train[i_t], G_train[support_indices], 2)

        for idx, j in enumerate(support_indices):
            kernel_cache[j, 0] = kernel_vec[idx]
            sigma_loss += alpha[j] * y_train[j] * kernel_cache[j, 0]

        if y_train[i_t] / (lamda * t) * sigma_loss < 1:
            alpha[i_t] += 1
            kernel_cache[i_t, 0] = 1.0
            kernel_cache[i_t, 1] = t

            if t > 100:
                coh_max = 0
                coh_index = -1
                for j in support_indices:
                    if j == i_t: continue
                    if t - kernel_cache[j, 1] < 100: continue
                    if alpha[j] > 1: continue
                    coh = kernel_cache[j, 0]
                    if coh > coh_max:
                        coh_max = coh
                        coh_index = j

                if coh_max > delta_c and coh_index >= 0:
                    alpha[coh_index] = 0
        t += 1

    return alpha
