import numpy as np
import matplotlib.pyplot as plt

# ==== 各回のデータ ====
times = np.array([
    [9.8444, 7.6917, 12.3169, 44.3889, 70.1796, 52.8072, 72.4535, 61.2284],
    [11.7267, 10.1615, 7.8137, 52.8662, 79.2477, 77.995, 72.6661, 106.7571],
    [8.1115, 6.8002, 7.6522, 53.0229, 99.5668, 74.9785, 89.4791, 87.9774],
    [14.1913, 13.8512, 11.0844, 46.4148, 81.3375, 101.8621, 79.3229, 66.9801]
])

accs = np.array([
    [0.9282, 0.9811, 0.8960, 0.9490, 0.9452, 0.9433, 0.9546, 0.9225],
    [0.8526, 0.9471, 0.9471, 0.9754, 0.9754, 0.8658, 0.9868, 0.9622],
    [0.8563, 0.9149, 0.9792, 0.9792, 0.9168, 0.9641, 0.9471, 0.9527],
    [0.9187, 0.9282, 0.9282, 0.9735, 0.9244, 0.9282, 0.9206, 0.9301]
])

bases = np.array([
    [14, 5, 5, 84, 110, 96, 104, 104],
    [4, 3, 6, 90, 115, 109, 114, 130],
    [3, 3, 6, 94, 121, 104, 118, 110],
    [5, 4, 4, 99, 118, 126, 116, 110]
])

# ==== 統計量 ====
time_mean, time_std = times.mean(axis=0), times.std(axis=0)
acc_mean, acc_std = accs.mean(axis=0), accs.std(axis=0)
base_mean, base_std = bases.mean(axis=0), bases.std(axis=0)

lamdas = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]

# ==== 比較データ ====
time_no_prune = [9.4188, 19.0890, 43.0868, 80.8836, 201.6332, 239.7608, 258.5186, 256.3522]
acc_no_prune  = [0.9981, 0.9981, 0.9943, 0.9338, 0.7656, 0.6446, 0.6087, 0.6276]
base_no_prune = [18, 28, 68, 171, 400, 478, 472, 472]

time_nc_prune = [15.7494, 12.2922, 14.0270, 33.4471, 73.4886, 87.7436, 98.4539, 87.0964]
acc_nc_prune  = [0.9282, 0.9112, 0.9376, 0.9376, 0.7902, 0.6352, 0.6106, 0.6295]
base_nc_prune = [31, 27, 36, 92, 231, 265, 271, 260]

time_proj = [315.9307, 109.3227, 2076.8985, 2905.0784, 3229.0533, 2815.9781, 2776.0530, 2877.6462]
acc_proj  = [0.8809073724007561, 0.8657844990548205, 0.9886578449905482, 0.9659735349716446, 0.9149338374291115, 0.9848771266540642, 0.8733459357277883, 0.8998109640831758]
base_proj = [57, 14, 297, 327, 340, 304, 311, 302]

# ========== 1) Training Time ==========
plt.figure(figsize=(6, 4))
plt.errorbar(lamdas, time_mean, yerr=time_std, fmt='-o', capsize=5, label='Sparse Dict Pruning')
plt.plot(lamdas, time_no_prune, '--', marker='s', label='Pegasos')
plt.plot(lamdas, time_nc_prune, '-.', marker='^', label='Sparse Dict Growth')
# plt.plot(lamdas, time_proj, '-', marker='o', label='Sparse Dict Pruning(iter=5000)')
plt.xscale("log")
plt.ylabel("Time [s]")
plt.title("Training Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_time.png", dpi=600)
plt.show()

# ========== 2) Accuracy ==========
plt.figure(figsize=(6, 4))
plt.errorbar(lamdas, acc_mean, yerr=acc_std, fmt='-o', capsize=5, label='Sparse Dict Pruning')
plt.plot(lamdas, acc_no_prune, '--', marker='s', label='Pegasos')
plt.plot(lamdas, acc_nc_prune, '-.', marker='^', label='Sparse Dict Growth')
# plt.plot(lamdas, acc_proj, '-', marker='o', label='Sparse Dict Pruning(iter=5000)')
plt.xscale("log")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")
plt.legend()
plt.grid(True)
plt.ylim(0.5, 1.0)
plt.tight_layout()
plt.savefig("accuracy.png", dpi=600)
plt.show()

# ========== 3) Number of Basis ==========
plt.figure(figsize=(6, 4))
plt.errorbar(lamdas, base_mean, yerr=base_std, fmt='-o', capsize=5, label='Sparse Dict Pruning')
plt.plot(lamdas, base_no_prune, '--', marker='s', label='Pegasos')
plt.plot(lamdas, base_nc_prune, '-.', marker='^', label='Sparse Dict Growth')
# plt.plot(lamdas, base_proj, '-', marker='o', label='Sparse Dict Pruning(iter=5000)')
plt.xscale("log")
plt.ylabel("Support Vectors")
plt.xlabel("Lambda")
plt.title("Number of Basis (Support Vectors)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("basis_support_vectors.png", dpi=600)
plt.show()
